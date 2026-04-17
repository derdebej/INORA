import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["GLOG_minloglevel"] = "3"

import cv2
from paddleocr import PaddleOCR
import tempfile
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(name)s] %(levelname)s: %(message)s"
)
log = logging.getLogger("TEST_OCR")

ocr = PaddleOCR(lang='en', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # PaddleOCR 3.x requires PIL or file path, so save frame to temp file
    tmp_path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
    cv2.imwrite(tmp_path, frame)

    results = ocr.predict(tmp_path)

    # Save annotated image to another temp file
    annotated_path = tmp_path + "_out.jpg"
    results[0].save_to_img(annotated_path)

    if results:
        i = 0  
        for res in results:
            data       = res.json.get("res", {})
            rec_texts  = data.get("rec_texts", [])

            areas = [polygon_area(rect) for rect in data["dt_polys"]]
            if rec_texts and i == 0:
                log.debug(areas)
                log.debug(rec_texts)
                i = i+1

    # Load annotated image with OpenCV
    frame_out = cv2.imread(annotated_path)

    cv2.imshow("Real-Time OCR", frame_out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Cleanup temp files
    os.remove(tmp_path)
    os.remove(annotated_path)

cap.release()
cv2.destroyAllWindows()