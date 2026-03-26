from ultralytics import YOLO
import cv2

# ─────────────────────────────────────────
# Load Your Trained Model
# ─────────────────────────────────────────
model = YOLO("25031246.pt")

# ─────────────────────────────────────────
# Open Webcam
# ─────────────────────────────────────────
cap = cv2.VideoCapture(0)  # 0 = default camera

if not cap.isOpened():
    print("❌ Camera not found!")
    exit()

print("✅ Camera opened! Press 'Q' to quit.")

# ─────────────────────────────────────────
# Live Detection Loop
# ─────────────────────────────────────────
while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    # Run detection on current frame
    results = model.predict(
        source=frame,
        conf=0.5,
        verbose=False   # hides per-frame output in terminal
    )

    # Draw boxes on frame
    annotated_frame = results[0].plot()

    # Print detections in terminal
    for box in results[0].boxes:
        cls  = int(box.cls)
        conf = float(box.conf)
        name = model.names[cls]
        print(f"✅ {name} dinars — {conf:.1%}")

    # Show the live window
    cv2.imshow("💰 Dinar Detector — Press Q to quit", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Quitting...")
        break

# ─────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()