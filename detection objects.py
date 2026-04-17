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
VIDEO_PATH    = os.path.join(BASE_DIR, 'test.mp4')
OUTPUT_PATH   = os.path.join(BASE_DIR, 'output.mp4')
DEPTH_WEIGHTS = os.path.join(METRIC_DIR, 'checkpoints', 'depth_anything_v2_metric_vkitti_vits.pth')

# ── Constants ─────────────────────────────────────────────────
CONF_THRESHOLD = 0.4
MAX_DIST       = 20.0
IGNORE_CLASSES = ['']
DEPTH_EVERY_N  = 5
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
print("YOLO loaded.")

# ── Video setup ───────────────────────────────────────────────
cap     = cv2.VideoCapture(VIDEO_PATH)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps     = cap.get(cv2.CAP_PROP_FPS)
total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
writer  = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps, (frame_w, frame_h)
)
print(f"{frame_w}x{frame_h} @ {fps}fps — {total} frames")

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
frame_count = 0
depth_map   = None
t_start     = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % DEPTH_EVERY_N == 0 or depth_map is None:
        depth_map = get_depth_map(frame)

    results = det_model.predict(
        frame, conf=CONF_THRESHOLD,
        device=0 if DEVICE == 'cuda' else 'cpu',
        verbose=False, imgsz=640
    )[0]

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
        label = f"{det_model.names[cls_id]} {dist}m ({conf:.2f})"
        color = (0, 0, 255) if dist <= 2.0 else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1-th-10), (x1+tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    if frame_count % 50 == 0:
        elapsed   = time.time() - t_start
        fps_real  = frame_count / elapsed
        remaining = (total - frame_count) / fps_real if fps_real > 0 else 0
        print(f"Frame {frame_count}/{total} | {fps_real:.1f} fps | ~{remaining/60:.1f} min remaining")

    writer.write(frame)

cap.release()
writer.release()
total_time = time.time() - t_start
print(f"Done! {frame_count} frames in {total_time/60:.1f} minutes")
print(f"Output saved to: {OUTPUT_PATH}")
