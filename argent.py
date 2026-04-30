import cv2
import time
from collections import defaultdict
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════
MODEL_PATH = "best (15).pt"   # ← path to your downloaded best.pt
SOURCE     = 0            # ← 0 for default webcam, 1 for external camera
CONF       = 0.35         # ← confidence threshold
IOU        = 0.45         # ← NMS threshold
# ═══════════════════════════════════════════════════════

PALETTE = [
    (0, 200, 100),  (0, 120, 255),  (255, 80, 0),   (180, 0, 255),
    (0, 220, 220),  (255, 200, 0),  (100, 255, 80),  (255, 0, 120),
    (80, 80, 255),
]

def get_color(cls_id):
    return PALETTE[cls_id % len(PALETTE)]

def draw_box(frame, box, cls_id, cls_name, conf):
    x1, y1, x2, y2 = map(int, box)
    color = get_color(cls_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{cls_name}  {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

def draw_overlay(frame, fps, counts):
    h, w = frame.shape[:2]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 100), 2, cv2.LINE_AA)

    # Detections summary
    summary = "  |  ".join(f"{n}: {c}" for n, c in sorted(counts.items()))
    cv2.putText(frame, summary if summary else "No detections", (130, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # Bottom hint
    cv2.putText(frame, "Q / ESC = quit   S = screenshot   P = pause",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (160, 160, 160), 1, cv2.LINE_AA)


def run():
    print(f"\nLoading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    names = model.names
    print(f"Classes: {list(names.values())}")

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source: {SOURCE}")

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    paused       = False
    frame_count  = 0
    fps          = 0.0
    t_fps        = time.time()
    screenshot_n = 0
    frame        = None

    print("Running! Press Q or ESC to quit, S to screenshot, P to pause.\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed.")
                break

            frame_count += 1

            # ── Inference ──────────────────────────────
            results = model.predict(
                source=frame,
                conf=CONF,
                iou=IOU,
                imgsz=640,
                verbose=False,
            )[0]

            # ── Draw detections ────────────────────────
            counts = defaultdict(int)
            if results.boxes is not None:
                for box_data in results.boxes:
                    cls_id   = int(box_data.cls[0])
                    cls_name = names[cls_id]
                    conf     = float(box_data.conf[0])
                    xyxy     = box_data.xyxy[0].tolist()
                    draw_box(frame, xyxy, cls_id, cls_name, conf)
                    counts[cls_name] += 1

            # ── FPS ────────────────────────────────────
            if frame_count % 30 == 0:
                fps   = 30 / (time.time() - t_fps)
                t_fps = time.time()

            draw_overlay(frame, fps, counts)

        # ── Show ───────────────────────────────────────
        cv2.imshow("Money Detector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("s") and frame is not None:
            path = f"screenshot_{screenshot_n:04d}.jpg"
            cv2.imwrite(path, frame)
            print(f"Screenshot saved → {path}")
            screenshot_n += 1
        elif key == ord("p"):
            paused = not paused
            print("Paused." if paused else "Resumed.")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    run()
