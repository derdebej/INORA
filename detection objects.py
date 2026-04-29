import os, sys

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
METRIC_DIR = os.path.join(BASE_DIR, 'Depth-Anything-V2', 'metric_depth')
sys.path.insert(0, METRIC_DIR)

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
import time

# ── Device ────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ── Paths ─────────────────────────────────────────────────────
WEIGHTS_PATH  = os.path.join(BASE_DIR, 'best.pt')
DEPTH_WEIGHTS = os.path.join(METRIC_DIR, 'checkpoints', 'depth_anything_v2_metric_vkitti_vits.pth')

# ── Constants ─────────────────────────────────────────────────
CONF_THRESHOLD = 0.4
MAX_DIST       = 20.0
IGNORE_CLASSES = ['']
SKIP_FRAMES    = 4          # run YOLO+depth every (SKIP_FRAMES+1) frames
INPUT_WIDTH    = 512
INPUT_HEIGHT   = 512

# ── Load Depth-Anything-V2 ────────────────────────────────────
print("Loading Depth-Anything-V2...")
depth_model = DepthAnythingV2(
    encoder='vits',
    features=64,
    out_channels=[48, 96, 192, 384],
    max_depth=80
)
depth_model.load_state_dict(torch.load(DEPTH_WEIGHTS, map_location=DEVICE))
depth_model.eval().to(DEVICE)
print("Depth model loaded.")

# ── Load YOLO ─────────────────────────────────────────────────
print("Loading YOLO...")
det_model = YOLO(WEIGHTS_PATH)
#det_model.fuse()
print("YOLO loaded.")

# ── Camera setup ──────────────────────────────────────────────
cap     = cv2.VideoCapture(0)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera: {frame_w}x{frame_h}")

# ── Depth inference ───────────────────────────────────────────
def get_depth_map(frame):
    small = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    with torch.no_grad():
        depth = depth_model.infer_image(small)
    depth = cv2.resize(depth, (frame_w, frame_h))
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    return depth

# ── Main loop ─────────────────────────────────────────────────
frame_count     = 0
depth_map       = None
last_detections = []   # frozen (x1,y1,x2,y2,label,color) from last inference
t_start         = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    run_inference = (frame_count % (SKIP_FRAMES + 1) == 0) or depth_map is None

    if run_inference:
        depth_map = get_depth_map(frame)

        results = det_model.predict(
            frame, conf=CONF_THRESHOLD,
            device=0 if DEVICE == 'cuda' else 'cpu',
            verbose=False, imgsz=640
        )[0]

        last_detections = []
        for box in results.boxes:
            cls_id = int(box.cls)
            if det_model.names[cls_id] in IGNORE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy          = (x1 + x2) // 2, (y1 + y2) // 2

            patch = depth_map[max(0, cy-10):cy+10, max(0, cx-10):cx+10]
            if patch.size == 0:
                continue
            dist = round(float(np.median(patch)), 1)

            if dist > MAX_DIST:
                continue

            conf  = float(box.conf)
            dist  = round(dist / 6, 1)
            label = f"{det_model.names[cls_id]} {dist}m ({conf:.2f})"
            color = (0, 0, 255) if dist <= 2.0 else (0, 255, 0)
            last_detections.append((x1, y1, x2, y2, label, color))

    for (x1, y1, x2, y2, label, color) in last_detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1-th-10), (x1+tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    elapsed  = time.time() - t_start
    fps_real = frame_count / elapsed if elapsed > 0 else 0
    cv2.putText(frame, f"FPS: {fps_real:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

del depth_model, det_model
if DEVICE == 'cuda':
    torch.cuda.empty_cache()
print("Models unloaded.")
