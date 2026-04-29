import os, sys

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
METRIC_DIR = os.path.join(BASE_DIR, 'Depth-Anything-V2', 'metric_depth')
sys.path.insert(0, METRIC_DIR)

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2

WEIGHTS_PATH  = os.path.join(BASE_DIR, 'best.pt')
DEPTH_WEIGHTS = os.path.join(METRIC_DIR, 'checkpoints', 'depth_anything_v2_metric_vkitti_vits.pth')


class InoraDetection:

    def __init__(
        self,
        conf_threshold=0.4,
        max_dist=20.0,
        skip_frames=4,
        input_size=512,
        ignore_classes=None,
    ):
        self.conf_threshold = conf_threshold
        self.max_dist       = max_dist
        self.skip_frames    = skip_frames
        self.input_size     = input_size
        self.ignore_classes = ignore_classes or ['']
        self.device         = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        print("Loading Depth-Anything-V2...")
        self.depth_model = DepthAnythingV2(
            encoder='vits', features=64,
            out_channels=[48, 96, 192, 384], max_depth=80
        )
        self.depth_model.load_state_dict(torch.load(DEPTH_WEIGHTS, map_location=self.device))
        self.depth_model.eval().to(self.device)
        print("Depth model loaded.")

        print("Loading YOLO...")
        self.det_model = YOLO(WEIGHTS_PATH)
        print("YOLO loaded.")

        self._frame_count     = 0
        self._depth_map       = None
        self._last_detections = []
        self._frame_w         = None
        self._frame_h         = None

    def _get_depth_map(self, frame):
        small = cv2.resize(frame, (self.input_size, self.input_size))
        with torch.no_grad():
            depth = self.depth_model.infer_image(small)
        depth = cv2.resize(depth, (self._frame_w, self._frame_h))
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        return depth

    def process(self, frame):
        self._frame_h, self._frame_w = frame.shape[:2]
        self._frame_count += 1
        run_inference = (self._frame_count % (self.skip_frames + 1) == 0) or self._depth_map is None

        if run_inference:
            self._depth_map = self._get_depth_map(frame)

            results = self.det_model.predict(
                frame, conf=self.conf_threshold,
                device=0 if self.device == 'cuda' else 'cpu',
                verbose=False, imgsz=640
            )[0]

            self._last_detections = []
            for box in results.boxes:
                cls_id = int(box.cls)
                if self.det_model.names[cls_id] in self.ignore_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy          = (x1 + x2) // 2, (y1 + y2) // 2

                patch = self._depth_map[max(0, cy-10):cy+10, max(0, cx-10):cx+10]
                if patch.size == 0:
                    continue
                dist = round(float(np.median(patch)) / 6, 1)

                if dist > self.max_dist:
                    continue

                conf  = float(box.conf)
                label = f"{self.det_model.names[cls_id]} {dist}m ({conf:.2f})"
                color = (0, 0, 255) if dist <= 2.0 else (0, 255, 0)
                self._last_detections.append((x1, y1, x2, y2, label, color))

        for (x1, y1, x2, y2, label, color) in self._last_detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1-th-10), (x1+tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return frame

    def release(self):
        del self.depth_model, self.det_model
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        print("Models unloaded.")
