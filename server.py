import asyncio
import websockets
import cv2
import numpy as np
import base64
import threading
import queue
import time
import atexit

# Global server state
latest_frame = None
consumers = set()
server_started = False
server_thread = None

def start_server():
    """Start the WebSocket server in background"""
    global server_started, server_thread
    
    if server_started:
        return
    
    def run_server():
        global server_started
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def handle_connection(websocket):
            global latest_frame, consumers  # ADD consumers here!
            print(f"New connection from {websocket.remote_address}")
            
            try:
                # Wait a bit to see if this connection sends data (browser) or waits (consumer)
                first_message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                
                # This is the browser sending frames
                latest_frame = first_message
                print("✓ Browser connected and sending frames")
                
                frame_count = 1
                async for message in websocket:
                    latest_frame = message
                    frame_count += 1
                    if frame_count % 30 == 0:  # Log every 30 frames
                        print(f"Received {frame_count} frames from browser")
                    
                    # Send to all consumers immediately
                    disconnected = set()
                    for consumer in consumers.copy():
                        try:
                            await consumer.send(message)
                        except Exception as e:
                            print(f"Failed to send to consumer: {e}")
                            disconnected.add(consumer)
                    
                    # Remove disconnected consumers
                    consumers -= disconnected
                    
                print("Browser disconnected")
                            
            except asyncio.TimeoutError:
                # This is a consumer waiting for frames
                consumers.add(websocket)
                print(f"✓ Consumer connected (total: {len(consumers)})")
                
                try:
                    # Send current frame if available
                    if latest_frame:
                        await websocket.send(latest_frame)
                        print("Sent current frame to new consumer")
                    
                    # Keep connection alive
                    while True:
                        await asyncio.sleep(1)
                        
                except websockets.exceptions.ConnectionClosed:
                    print("Consumer disconnected")
                except Exception as e:
                    print(f"Consumer error: {e}")
                finally:
                    consumers.discard(websocket)
                    print(f"Consumer removed (remaining: {len(consumers)})")
        
        async def main():
            print("WebSocket server starting on ws://localhost:8765")
            async with websockets.serve(handle_connection, "localhost", 8765):
                await asyncio.Future()
        
        server_started = True
        loop.run_until_complete(main())
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    while not server_started:
        time.sleep(0.1)

def stop_server():
    """Clean shutdown"""
    global server_started
    server_started = False

# Auto-start server on import and clean shutdown
start_server()
atexit.register(stop_server)

class WebSocketVideoCapture:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.frame_queue = queue.Queue(maxsize=10)
        self.is_opened = False
        self.loop = None
        self.websocket = None
        self.thread = None
        
        # Video properties
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30
        self.current_frame = None
        
    def open(self):
        """Start the websocket connection in a separate thread"""
        print("=== OPEN() CALLED ===")
        
        # Ensure server is running
        print("Starting server...")
        start_server()
        print(f"Server started: {server_started}")
        
        print("Starting WebSocket client...")
        self.thread = threading.Thread(target=self._run_websocket, daemon=True)
        self.thread.start()
        print("WebSocket thread started")
        
        # Wait for connection and first frame
        timeout = 10  # seconds
        start_time = time.time()
        print("Waiting for connection and first frame...")
        
        while (not self.is_opened or self.current_frame is None) and (time.time() - start_time) < timeout:
            elapsed = time.time() - start_time
            print(f"Waiting... is_opened={self.is_opened}, has_frame={self.current_frame is not None}, elapsed={elapsed:.1f}s")
            time.sleep(0.5)
        
        final_status = self.is_opened and self.current_frame is not None
        print(f"=== OPEN() RESULT: {final_status} ===")
        print(f"is_opened: {self.is_opened}")
        print(f"has_frame: {self.current_frame is not None}")
        
        if final_status:
            print(f"✓ Successfully connected! Frame size: {self.frame_width}x{self.frame_height}")
        else:
            print("✗ Failed to connect or receive first frame")
            print("Make sure:")
            print("1. HTML page is open in browser")
            print("2. You clicked 'Start' button")
            print("3. Browser granted camera permission")
        
        return final_status
    
    def _run_websocket(self):
        """Run websocket in separate thread with its own event loop"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._websocket_handler())
    
    async def _websocket_handler(self):
        """Handle websocket connection and frame reception"""
        try:
            uri = f"ws://{self.host}:{self.port}"
            print(f"Connecting to {uri}...")
            async with websockets.connect(uri) as websocket:
                self.websocket = websocket
                self.is_opened = True
                print(f"✓ Connected to {uri}")
                
                frame_count = 0
                async for message in websocket:
                    try:
                        # Decode base64 image
                        img_data = base64.b64decode(message)
                        nparr = np.frombuffer(img_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            frame_count += 1
                            if frame_count == 1:
                                print(f"✓ First frame received! Size: {frame.shape}")
                            elif frame_count % 30 == 0:
                                print(f"Received {frame_count} frames")
                            
                            # Update frame dimensions
                            self.frame_height, self.frame_width = frame.shape[:2]
                            self.current_frame = frame
                            
                            # Add frame to queue (remove old frames if queue is full)
                            if self.frame_queue.full():
                                try:
                                    self.frame_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            self.frame_queue.put(frame)
                        else:
                            print("⚠ Failed to decode frame")
                            
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        
        except Exception as e:
            print(f"WebSocket connection error: {e}")
        finally:
            self.is_opened = False
            print("WebSocket connection closed")
    
    def read(self):
        """Read a frame (mimics cv2.VideoCapture.read())"""
        if not self.is_opened:
            return False, None
            
        try:
            # Get latest frame with timeout
            frame = self.frame_queue.get(timeout=1.0)
            return True, frame
        except queue.Empty:
            return False, None
    
    def isOpened(self):
        """Check if capture is opened (mimics cv2.VideoCapture.isOpened())"""
        return self.is_opened
    
    def release(self):
        """Release the capture (mimics cv2.VideoCapture.release())"""
        self.is_opened = False
        if self.websocket:
            asyncio.run_coroutine_threadsafe(self.websocket.close(), self.loop)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
    
    def get(self, prop):
        """Get property (mimics cv2.VideoCapture.get())"""
        if prop == cv2.CAP_PROP_FRAME_WIDTH or prop == 3:
            return float(self.frame_width)
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT or prop == 4:
            return float(self.frame_height)
        elif prop == cv2.CAP_PROP_FPS or prop == 5:
            return float(self.fps)
        elif prop == cv2.CAP_PROP_FRAME_COUNT:
            return -1.0  # Unknown for live stream
        elif prop == cv2.CAP_PROP_POS_FRAMES:
            return 0.0  # Live stream
        elif prop == cv2.CAP_PROP_FOURCC:
            return cv2.VideoWriter_fourcc(*'MJPG')
        elif prop == cv2.CAP_PROP_BRIGHTNESS:
            return 0.0
        elif prop == cv2.CAP_PROP_CONTRAST:
            return 0.0
        elif prop == cv2.CAP_PROP_SATURATION:
            return 0.0
        elif prop == cv2.CAP_PROP_HUE:
            return 0.0
        else:
            return 0.0
    
    def set(self, prop, value):
        """Set property (mimics cv2.VideoCapture.set())"""
        # Most properties can't be changed for websocket stream
        # but we return True to maintain compatibility
        if prop == cv2.CAP_PROP_FRAME_WIDTH or prop == 3:
            self.frame_width = int(value)
            return True
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT or prop == 4:
            self.frame_height = int(value)
            return True
        elif prop == cv2.CAP_PROP_FPS or prop == 5:
            self.fps = int(value)
            return True
        else:
            return True  # Pretend we set it successfully

# Example usage
if __name__ == "__main__":
    print("WebSocket server is running...")
    print("Open the HTML file in your browser and click 'Start'")
    print("Then run your main script")
    
    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        stop_server()
