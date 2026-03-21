"""
INORA — Diagnostic PaddleOCR
=============================
Lance ce script pour voir exactement ce que PaddleOCR retourne
et corriger le parsing dans inora_ocr.py

  python debug_ocr.py
"""

import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["GLOG_minloglevel"] = "3"

import cv2
import tempfile
from paddleocr import PaddleOCR

print("=" * 50)
print("  Diagnostic PaddleOCR — INORA")
print("=" * 50)

ocr = PaddleOCR(
    lang='en',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

cap = cv2.VideoCapture(0)
print("\nCaméra ouverte. Montre un texte devant la caméra.")
print("Appuie sur ESPACE pour capturer et analyser.")
print("Appuie sur Q pour quitter.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Debug OCR — ESPACE pour capturer", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        print("\n── Capture en cours... ──────────────────────")

        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
        cv2.imwrite(tmp, frame)
        results = ocr.predict(tmp)
        os.remove(tmp)

        print(f"\nType results : {type(results)}")
        print(f"Longueur     : {len(results)}")

        res = results[0]
        print(f"\nType res     : {type(res)}")
        print(f"dir(res)     : {[a for a in dir(res) if not a.startswith('_')]}")

        # Affiche tout ce que contient l'objet
        for attr in dir(res):
            if not attr.startswith('_'):
                try:
                    val = getattr(res, attr)
                    if not callable(val):
                        print(f"\nres.{attr} = {val}")
                except:
                    pass
        os.remove(tmp)

        print(f"\nType de results      : {type(results)}")
        print(f"Longueur de results  : {len(results)}")

        for i, res in enumerate(results):
            print(f"\n── results[{i}] ──────────────────────────────")
            print(f"  Type : {type(res)}")
            print(f"  Attributs disponibles : {[a for a in dir(res) if not a.startswith('_')]}")

            # Test des attributs les plus courants
            for attr in ["rec_texts", "rec_scores", "dt_polys", "text", "texts", "boxes"]:
                if hasattr(res, attr):
                    val = getattr(res, attr)
                    print(f"\n  .{attr} ({type(val)}) = {val}")

            # Test de l'accès dict
            try:
                print(f"\n  Accès dict : {dict(res)}")
            except Exception:
                pass

            # Affiche tout l'objet
            print(f"\n  str(res) = {str(res)[:300]}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nDiagnostic terminé.")