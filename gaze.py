import cv2
import logging
import warnings
import numpy as np
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
from config import data_config
from utils.helpers import get_model, draw_bbox_gaze
import uniface
import requests
# phone_detector.py
from ultralytics import YOLO

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Gaze module integration for run_demo
# Call these before main loop:
#   gaze_detector, face_detector, device, idx_tensor, config = init_gaze_model()
#   pitch_range, yaw_range = run_gaze_calibration(cap, gaze_detector, face_detector, device, idx_tensor, config)
# Then in loop:
#   inattentive_flag, microsaccade_flag = evaluate_gaze_attention(frame, bbox, gaze_detector, device, idx_tensor, config, pitch_range, yaw_range)


API_URL = "http://localhost:4040/api/v1/calibration" # Replace with your actual endpoint

def get_device_id():
    import platform, hashlib
    device_info = platform.node() + platform.system() + platform.processor()
    return hashlib.sha256(device_info.encode()).hexdigest()

def fetch_calibration_data(device_id):
    try:
        response = requests.get(f"{API_URL}?device_id={device_id}")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"[Fetch Calibration Error] {e}")
    return None

def post_calibration_data(device_id, pitch_range, yaw_range):
    try:
        data = {
            "device_id": device_id,
            "pitch_range": pitch_range,
            "yaw_range": yaw_range
        }
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            print("[Calibration] Data successfully uploaded")
    except Exception as e:
        print(f"[Post Calibration Error] {e}")


gaze_history = []
delta_log = []

def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image_batch = image.unsqueeze(0)
    return image_batch

def detect_microsaccade(gaze_history, angle_thresh_deg=13, duration_ms=1000):
    if len(gaze_history) < 2:
        return False
    now = time.time()
    recent_entries = [entry for entry in gaze_history if now - entry[0] <= duration_ms / 1000]
    for i in range(1, len(recent_entries)):
        t0, p0, y0 = recent_entries[i - 1]
        t1, p1, y1 = recent_entries[i]
        delta = np.sqrt((p1 - p0) ** 2 + (y1 - y0) ** 2)
        delta_deg = np.degrees(delta)
        delta_log.append((time.time(), delta_deg))
        if delta_deg > angle_thresh_deg:
            return True
    return False

def init_gaze_model(model_name="resnet34", weight_path="resnet34.pt", dataset="gaze360"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gaze_detector = get_model(model_name, data_config[dataset]["bins"], inference_mode=True)
    state_dict = torch.load(weight_path, map_location=device)
    gaze_detector.load_state_dict(state_dict)
    gaze_detector.to(device)
    gaze_detector.eval()
    face_detector = uniface.RetinaFace()
    idx_tensor = torch.arange(data_config[dataset]["bins"], device=device, dtype=torch.float32)
    return gaze_detector, face_detector, device, idx_tensor, data_config[dataset]

def run_gaze_calibration(cap, gaze_detector, face_detector, device, idx_tensor, config):
    import time
    print("=" * 60)
    print("SCREEN BOUNDARY CALIBRATION")
    print("=" * 60)
    print("This calibration will capture your actual viewing limits.")
    print("You'll be asked to look at 5 positions: CENTER, TOP, BOTTOM, LEFT, RIGHT")
    print("This creates personalized attention boundaries for your setup.")
    print("\nYou have 5 seconds to make this window FULLSCREEN for best results!")

    # Phase 0: Fullscreen countdown
    print("\nPreparing calibration - make window fullscreen now...")
    
    prep_start = time.time()
    while time.time() - prep_start < 1:
        ret, frame = cap.read()
        if not ret:
            continue
        height, width = frame.shape[:2]
        remaining = 5 - int(time.time() - prep_start)
        center_x = width // 2
        center_y = height // 2
        cv2.putText(frame, f"Starting calibration in: {remaining}", (center_x - 180, center_y - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, str(remaining), (center_x - 20, center_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)
        cv2.putText(frame, "Make window fullscreen (F11 or maximize)", (center_x - 250, center_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.namedWindow("Gaze Calibration", cv2.WINDOW_NORMAL)
        cv2.imshow('Gaze Calibration', frame)
        cv2.waitKey(1)

    positions = [
        ("CENTER", "center of your screen", (0.5, 0.5)),
        ("TOP", "top edge of your screen", (0.5, 0.1)),
        ("BOTTOM", "bottom edge of your screen", (0.5, 0.9)),
        ("LEFT", "left edge of your screen", (0.1, 0.5)),
        ("RIGHT", "right edge of your screen", (0.9, 0.5))
    ]
    results = {}
    all_pitch = []
    all_yaw = []
    for pos_name, instruction, (rel_x, rel_y) in positions:
        print(f"\n--- Calibrating {pos_name} ---")
        print(f"Look at the {instruction}")

        # Phase 1: Setup countdown
        setup_start = time.time()
        while True:
            elapsed = time.time() - setup_start
            remaining = int(1 - elapsed)
            if remaining < 0:
                break

            ret, frame = cap.read()
            if not ret: continue
            height, width = frame.shape[:2]
            tx, ty = int(rel_x * width), int(rel_y * height)
            cv2.circle(frame, (tx, ty), 20, (0, 255, 255), 3)
            cv2.putText(frame, f"SETUP - Look at {pos_name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Starting in: {remaining}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
            cv2.putText(frame, f"Look at the {instruction}", 
                           (20, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow('Gaze Calibration', frame)
            key = cv2.waitKey(1)
            if remaining == 0:
                time.sleep(0.3)  

        # Phase 2: Record gaze
        print(f"Recording {pos_name}...")
        record_start = time.time()
        pitch_list, yaw_list = [], []
        while time.time() - record_start < 3:
            ret, frame = cap.read()
            if not ret: continue
            bboxes, keypoints = face_detector.detect(frame)
            for bbox in bboxes:
                x1, y1, x2, y2 = map(int, bbox[:4])
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                face_tensor = pre_process(face_img).to(device)
                pitch_logits, yaw_logits = gaze_detector(face_tensor)
                pitch = torch.sum(F.softmax(pitch_logits, dim=1) * idx_tensor, dim=1) * config["binwidth"] - config["angle"]
                yaw = torch.sum(F.softmax(yaw_logits, dim=1) * idx_tensor, dim=1) * config["binwidth"] - config["angle"]
                if pitch.item() is not None and yaw.item() is not None:
                  pitch_list.append(pitch.item())
                  yaw_list.append(yaw.item())
        if len(pitch_list) < 5:
            print(f"❌ Not enough samples for {pos_name}")
            return None, None
        results[pos_name] = {
            "pitch_mean": np.mean(pitch_list),
            "pitch_std": np.std(pitch_list),
            "yaw_mean": np.mean(yaw_list),
            "yaw_std": np.std(yaw_list)
        }
        all_pitch.extend(pitch_list)
        all_yaw.extend(yaw_list)
        print(f"✓ {pos_name}: pitch={results[pos_name]['pitch_mean']:.2f}, yaw={results[pos_name]['yaw_mean']:.2f}")

    
    pitch_min, pitch_max = min(all_pitch), max(all_pitch)
    yaw_min, yaw_max = min(all_yaw), max(all_yaw)
    pitch_range_span = pitch_max - pitch_min
    yaw_range_span = yaw_max - yaw_min

    tolerance_ratio = 0.1  # 10%

    pitch_range = (
        pitch_min - pitch_range_span * tolerance_ratio,
        pitch_max + pitch_range_span * 0.05
    )
    yaw_range = (
        yaw_min,# - yaw_range_span * tolerance_ratio,
        yaw_max + yaw_range_span * tolerance_ratio
    )

 
    print("\n✅ Calibration complete.")
    print(f"Final PITCH_RANGE: {pitch_range}")
    print(f"Final YAW_RANGE:   {yaw_range}")
    cv2.destroyWindow("Gaze Calibration")
    return pitch_range, yaw_range


def evaluate_gaze_attention(frame, bbox, gaze_detector, device, idx_tensor, config, pitch_range, yaw_range):
    global gaze_history
    x_min, y_min, x_max, y_max = map(int, bbox[:4])
    image = frame[y_min:y_max, x_min:x_max]
    image = pre_process(image).to(device)
    pitch, yaw = gaze_detector(image)
    pitch_predicted = torch.sum(F.softmax(pitch, dim=1) * idx_tensor, dim=1) * config["binwidth"] - config["angle"]
    yaw_predicted = torch.sum(F.softmax(yaw, dim=1) * idx_tensor, dim=1) * config["binwidth"] - config["angle"]
    pitch_deg = pitch_predicted.cpu().item()
    yaw_deg = yaw_predicted.cpu().item()

    inattentive_flag = not (pitch_range[0] <= pitch_deg <= pitch_range[1] and yaw_range[0] <= yaw_deg <= yaw_range[1])
    
    pitch_predicted = np.radians(pitch_deg)
    yaw_predicted = np.radians(yaw_deg)
    now = time.time()
    gaze_history.append((now, pitch_predicted, yaw_predicted))
    gaze_history = [entry for entry in gaze_history if now - entry[0] <= 1.0]
    draw_bbox_gaze(frame, bbox, pitch_predicted, yaw_predicted)
    micro_flag = detect_microsaccade(gaze_history)
    return inattentive_flag, micro_flag

# Load model only once
yolo_model = YOLO("./models/yolov8n.pt")  # or your preferred model
PHONE_CLASS_ID = 67

def detect_phone(frame, confidence_threshold=0.5):
    """
    Detect phones in the frame using YOLOv8.
    Returns (bool, list of bounding boxes).
    """
    phone_boxes = []
    try:
        results = yolo_model(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                if class_id == PHONE_CLASS_ID and conf > confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    phone_boxes.append((x1, y1, x2, y2))
        phone_detected = len(phone_boxes) > 0
        return phone_detected, phone_boxes
    except Exception as e:
        print(f"[Phone Detection Error] {e}")
        return False, []
