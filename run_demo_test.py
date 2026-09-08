import os

import cv2
import pkg_resources
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import time
import json
from collections import deque, defaultdict
import requests
# My libs
import spiga.demo.analyze.track.get_tracker as tr
import spiga.demo.analyze.extract.spiga_processor as pr_spiga
from spiga.demo.analyze.analyzer import VideoAnalyzer
from spiga.demo.visualize.viewer import Viewer

from approach.ResEmoteNet import ResEmoteNet
from server import WebSocketVideoCapture
# Gaze integration
from gaze import init_gaze_model, run_gaze_calibration, evaluate_gaze_attention, detect_phone, get_device_id, fetch_calibration_data, post_calibration_data
import sys
print(f"[Python] Started run_demo_test.py for user {sys.argv}")


# Load ResEmoteNet model
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = ResEmoteNet().to(device)
checkpoint = torch.load("models/best_model.pth", map_location=device)
resnet_model.load_state_dict(checkpoint['model_state_dict'])
resnet_model.eval()

# ResEmoteNet transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def notify_ai_agent(user_id, level):
    url = "http://localhost:4040/ai-agent/event"  ### AI agent server should be there!
    data = {"user_id": user_id, "status": level}
    try:
        requests.post(url, json=data)
    except Exception as e:
        print(f"[AI Agent Notify Error] {e}")

# Paths
video_out_path_dft = pkg_resources.resource_filename('spiga', 'demo/outputs')
if not os.path.exists(video_out_path_dft):
    os.makedirs(video_out_path_dft)

left_eye_idx = [62,66]
right_eye_idx = [70,74]
left_pupil_idx = 96
right_pupil_idx = 97

def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "1")

def run_integrated_demo():
    import argparse
    pars = argparse.ArgumentParser(description='SPIGA + ResEmoteNet + Gaze Demo')
    pars.add_argument('--calibration_only', type=str2bool, default=False, help='Run only gaze calibration')
    pars.add_argument('--user_id', type=str, default='xfy', required=True, help='Logged-in user ID')
    pars.add_argument('-i', '--input', type=str, default='0', help='Video input')
    pars.add_argument('-d', '--dataset', type=str, default='wflw',
                      choices=['wflw', '300wpublic', '300wprivate', 'merlrav'],
                      help='SPIGA pretrained weights per dataset')
    pars.add_argument('--window', type=int, default=180, help='Tracking window in seconds')
    pars.add_argument('-t', '--tracker', type=str, default='RetinaSort',
                      choices=['RetinaSort', 'RetinaSort_Res50'], help='Tracker name')
    pars.add_argument('--fps', type=int, default=30, help='Frames per second')
    pars.add_argument('--thresh1', type=float, default=0.10, help='Low threshold for inattentiveness')
    pars.add_argument('--thresh2', type=float, default=0.30, help='Medium threshold for inattentiveness')
    #pars.add_argument('--thresh3', type=float, default=0.25, help='High threshold for inattentiveness')

    args = pars.parse_args()

    #capture = cv2.VideoCapture(int(args.input))
    capture = WebSocketVideoCapture()
    if not capture.open():  # ADD THIS CHECK!
        print("Failed to connect to websocket camera")
        exit(1)  # STOP HERE IF OPEN FAILS

    vid_w, vid_h = capture.get(3), capture.get(4)
    viewer = Viewer('Integrated Demo', width=vid_w, height=vid_h, fps=args.fps)
    viewer.start_view()

    # Init gaze model and calibration
    gaze_detector, gaze_face_detector, gaze_device, idx_tensor, config = init_gaze_model()

    #pitch_range, yaw_range = run_gaze_calibration(capture, gaze_detector, gaze_face_detector, gaze_device, idx_tensor, config)
    device_id = get_device_id()
    if args.calibration_only:
        pitch_range, yaw_range = run_gaze_calibration(capture, gaze_detector, gaze_face_detector, gaze_device, idx_tensor, config)
        post_calibration_data(device_id, pitch_range, yaw_range)
        print("[Calibration Only Mode] Calibration completed and data posted.")
        return
    saved_data = fetch_calibration_data(device_id)

    if saved_data:
        print("Loaded previous calibration")
        pitch_range = tuple(saved_data["pitch_range"])
        yaw_range = tuple(saved_data["yaw_range"])
    else:
        pitch_range, yaw_range = run_gaze_calibration(capture, gaze_detector, gaze_face_detector, gaze_device, idx_tensor, config)
        post_calibration_data(device_id, pitch_range, yaw_range)

    faces_tracker = tr.get_tracker(args.tracker)
    faces_tracker.detector.set_input_shape(vid_h, vid_w)
    processor = pr_spiga.SPIGAProcessor(dataset=args.dataset)
    faces_analyzer = VideoAnalyzer(faces_tracker, processor=processor)

    prev_left_eye_center = None
    prev_right_eye_center = None
    prev_left_pupil = None
    prev_right_pupil = None

    inattentive_per_second = defaultdict(bool) 
    #emotion_log = []
    emotion_records = []
    emotion_window = deque()

    segments = []
    prev_sec_state = False

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        
        current_time = time.time()
        current_sec = int(current_time)
        headpose_log = []
        phone_log = []

        faces_analyzer.process_frame(frame)
        inattentive = False
        #Rule 1: No face detected
        if not faces_analyzer.tracked_obj:
            inattentive = True
        else:
            for face in faces_analyzer.tracked_obj:
                if face.bbox is not None:
                    bbox = np.array(face.bbox).astype(int)
                    x_min, y_min, x_max, y_max = bbox[:4]
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(frame.shape[1], x_max)
                    y_max = min(frame.shape[0], y_max)
                    face_crop = frame[y_min:y_max, x_min:x_max]

                    #Rule2: Headpose inattentive
                    if face.headpose is not None and len(face.headpose) == 6:
                        yaw, pitch, roll = face.headpose[:3]
                        headpose_log.append({
                            "timestamp": current_time,
                            "yaw": float(yaw),
                            "pitch": float(pitch),
                            "roll": float(roll)
                        })
                        
                        if abs(yaw) > 20 or abs(pitch) > 20:
                            inattentive = True

                    #Rule3: Emotion inattentive
                    if not inattentive and face_crop.size != 0:
                        face_pil = Image.fromarray(face_crop)
                        face_tensor = transform(face_pil).unsqueeze(0).to(device)
                        with torch.no_grad():
                            outputs = resnet_model(face_tensor)
                            probabilities = F.softmax(outputs, dim=1).cpu().numpy().flatten()
                        max_index = np.argmax(probabilities)
                        max_emotion = emotions[max_index]
                        confidence = probabilities[max_index]
                        if confidence > 0.85:
                            emotion_window.append((current_time, max_emotion))
                            emotion_records.append({
                                "timestamp": current_time,
                                "emotion": max_emotion,
                                "confidence": float(confidence)
                            })
                        if max_emotion != 'neutral' and max_emotion != 'surprise' and confidence > 0.90:
                            inattentive = True
                        label = f"{max_emotion}: {probabilities[max_index]:.2f}"
                        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    #Rule4: Gaze inattentive
                    gaze_inattentive, micro_flag = evaluate_gaze_attention(
                        frame, bbox, gaze_detector, gaze_device, idx_tensor, config, pitch_range, yaw_range)
                    if gaze_inattentive: #or micro_flag:
                        inattentive = True

              
                    # Rule 5: Phone detection inattentive
                    phone_inattentive, phone_boxes = detect_phone(frame)
                    if phone_inattentive:
                        inattentive = True
                        for (x1, y1, x2, y2) in phone_boxes:
                            phone_log.append({
                                "timestamp": current_time,
                                "box": [int(x1), int(y1), int(x2), int(y2)]
                            })
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, "Phone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        #Calculate inattentive time
        if current_sec not in inattentive_per_second:
            inattentive_per_second[current_sec] = inattentive
        else:
            inattentive_per_second[current_sec] |= inattentive


        min_allowed_sec = current_sec - args.window
        inattentive_per_second = {sec: val for sec, val in inattentive_per_second.items() if sec >= min_allowed_sec}
        
        if current_sec - 1 in inattentive_per_second:
            prev = inattentive_per_second[current_sec - 1]
            curr = inattentive_per_second[current_sec]
            if curr and not prev:
                segment_start = current_sec
            elif not curr and prev:
                segment_end = current_sec
                segments.append((segment_start, segment_end))   

        segment_start = None
        if current_sec - 1 in inattentive_per_second:
            prev = inattentive_per_second[current_sec - 1]
            curr = inattentive_per_second[current_sec]
            if curr and not prev:
                segment_start = current_sec
            elif not curr and prev and segment_start is not None:
                segment_end = current_sec
                segments.append((segment_start, segment_end))
                

        # === Per-second evaluation of past 1 sec ===
        #while emotion_window and current_time - emotion_window[0][0] > 1.0:
            #emotion_window.popleft()

        #if emotion_window:
            #probs_in_window = [vec for ts, vec in emotion_window]
            #avg_probs = np.mean(probs_in_window, axis=0)
            #avg_emotion = emotions[np.argmax(avg_probs)]
            #emotion_log.append({
                #"timestamp": current_time,
                #"mean_emotion": avg_emotion
            #})
    
        inattentive_count = sum(inattentive_per_second.values())
        ratio = inattentive_count / args.window

        if ratio > args.thresh1:
            if ratio <= args.thresh2:
                tag = "slightly inattentive"
           
            elif ratio > args.thresh2:
                tag = "heavily inattentive"

        
            print("3-min Status: ",tag) 
            print("Inattentive Ratio: ",ratio) 
            notify_ai_agent(args.user_id, tag)
            #cv2.putText(frame, f"3-min Status: {tag}",
                        #(50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            inattentive_per_second = defaultdict(bool)
        
        #########       
        if inattentive:
            cv2.putText(frame, "INATTENTIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        key = viewer.process_image(frame, drawers=[faces_analyzer], show_attributes=['landmarks', 'headpose', 'face_id', 'fps'])
        if key:
            break

    #print("\nInattentive segments:")
    #for start, end in segments:
        #print(f"From {time.strftime('%H:%M:%S', time.localtime(start))} to {time.strftime('%H:%M:%S', time.localtime(end))}")

    capture.release()
    viewer.close()
    with open("emotion_log.json", "w") as f:
        json.dump(emotion_records, f, indent=2)
    with open("headpose_log.json", "w") as f:
        json.dump(headpose_log, f, indent=2)
    with open("phone_log.json", "w") as f:
        json.dump(phone_log, f, indent=2)

if __name__ == '__main__':
    run_integrated_demo()
