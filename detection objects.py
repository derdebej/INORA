# ============================================================
# Cell 1 — Install + download
# ============================================================
!git clone https://github.com/DepthAnything/Depth-Anything-V2 -q
%cd Depth-Anything-V2/metric_depth
!pip install -r requirements.txt -q
!pip install ultralytics huggingface_hub -q

import os, shutil
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="depth-anything/Depth-Anything-V2-Metric-VKITTI-Small",
    filename="depth_anything_v2_metric_vkitti_vits.pth",
)
os.makedirs('/content/Depth-Anything-V2/metric_depth/checkpoints', exist_ok=True)
shutil.copy(path, '/content/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vits.pth')

size = os.path.getsize('/content/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vits.pth')
print(f"✅ Downloaded! File size: {size/1e6:.1f} MB")

# ============================================================
# Run this ONCE to patch the root dpt.py
# ============================================================
import subprocess
result = subprocess.run(
    ['python', '-c', 'from depth_anything_v2.dpt import DepthAnythingV2; import inspect; print(inspect.getfile(DepthAnythingV2))'],
    capture_output=True, text=True,
    cwd='/content/Depth-Anything-V2/metric_depth'
)
print(result.stdout)

# Copy metric_depth dpt.py over root dpt.py
import shutil
shutil.copy(
    '/content/Depth-Anything-V2/metric_depth/depth_anything_v2/dpt.py',
    '/content/Depth-Anything-V2/depth_anything_v2/dpt.py'
)
print("✅ Patched!")

import sys
# Remove any cached depth_anything_v2 modules
keys_to_remove = [k for k in sys.modules if 'depth_anything' in k]
for k in keys_to_remove:
    del sys.modules[k]

sys.path.insert(0, '/content/Depth-Anything-V2/metric_depth')

# ============================================================
# Cell 2 — Full optimized video code
# ============================================================
import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import sys
# ← critical: import from metric_depth subfolder, not root
sys.path.insert(0, '/content/Depth-Anything-V2/metric_depth')

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
from IPython.display import HTML
from base64 import b64encode
import time

# ── Paths ─────────────────────────────────────────────────────
WEIGHTS_PATH = '/content/best.pt'
VIDEO_PATH   = '/content/test.mp4'
OUTPUT_PATH  = '/content/output.mp4'

# ── Constants ─────────────────────────────────────────────────
CONF_THRESHOLD = 0.4
MAX_DIST       = 20.0
IGNORE_CLASSES = ['']
DEPTH_EVERY_N  = 5
INPUT_WIDTH    = 512
INPUT_HEIGHT   =512

# ── Load Depth-Anything-V2 vits on GPU ────────────────────────
print("⏳ Loading Depth-Anything-V2 vits...")
depth_model = DepthAnythingV2(
    encoder='vits',
    features=64,
    out_channels=[48, 96, 192, 384],
    max_depth=80  # 80 for outdoor (vkitti)
)
depth_model.load_state_dict(torch.load(
    '/content/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vits.pth',
    map_location='cuda'
))
depth_model.eval().cuda()
print(f"✅ Depth model loaded | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# ── Load YOLO on GPU ──────────────────────────────────────────
print("⏳ Loading YOLO...")
det_model = YOLO(WEIGHTS_PATH)
print(f"✅ YOLO loaded | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

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
print(f"📹 {frame_w}x{frame_h} @ {fps}fps — {total} frames")

# ── Depth inference ───────────────────────────────────────────
def get_depth_map(frame):
    small = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    with torch.no_grad():
        depth = depth_model.infer_image(small)
    depth = cv2.resize(depth, (frame_w, frame_h))
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
        device=0, verbose=False,
        imgsz=640
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
        print(f"⏳ Frame {frame_count}/{total} | "
            f"{fps_real:.1f} fps | "
            f"~{remaining/60:.1f} min remaining | "
            f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    writer.write(frame)

cap.release()
writer.release()
total_time = time.time() - t_start
print(f"✅ Done! {frame_count} frames in {total_time/60:.1f} minutes")