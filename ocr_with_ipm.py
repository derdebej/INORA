"""
ocr_with_ipm.py
===============
Module OCR temps réel avec estimation de distance IPM intégrée.
Basé sur PaddleOCR 3.x + OpenCV.

Avant de lancer : ajuster les paramètres de calibration dans CameraConfig
selon votre matériel.
"""

import cv2
import tempfile
import os
import logging
from paddleocr import PaddleOCR

from ipm_module import (
    IPMEstimator,
    ReadabilityChecker,
    ObjectType,
    BBox,
    paddle_poly_to_bbox,
    get_global_text_bbox,
    annotate_frame,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(name)s] %(levelname)s: %(message)s"
)
log = logging.getLogger("OCR_IPM")


# ─────────────────────────────────────────────
#  Calibration caméra — À MODIFIER selon votre matériel
# ─────────────────────────────────────────────

class CameraConfig:
    CAM_INDEX       = 1       # index caméra (0 = webcam intégrée, 1 = externe)
    IMG_W           = 640     # résolution largeur
    IMG_H           = 480     # résolution hauteur
    CAM_HEIGHT_M    = 1.5     # hauteur de la caméra/tête au-dessus du sol (mètres)
    VFOV_DEG        = 70.0    # champ de vue vertical (degrés) — voir spec de votre caméra
    HFOV_DEG        = 90.0    # champ de vue horizontal (degrés)
    TILT_DEG        = 10.0    # inclinaison vers le bas (0 = horizontal)
    DRAW_ZONES      = True    # afficher les lignes de niveau IPM


# ─────────────────────────────────────────────
#  Utilitaires
# ─────────────────────────────────────────────

def polygon_area(points) -> float:
    """Aire d'un polygone (formule du lacet de Shoelace)."""
    area = 0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


# ─────────────────────────────────────────────
#  Boucle principale
# ─────────────────────────────────────────────

def main():
    cfg = CameraConfig()

    # ── Init IPM ──────────────────────────────────────────────────────────
    ipm = IPMEstimator(
        cam_height = cfg.CAM_HEIGHT_M,
        vfov       = cfg.VFOV_DEG,
        hfov       = cfg.HFOV_DEG,
        img_w      = cfg.IMG_W,
        img_h      = cfg.IMG_H,
        tilt_deg   = cfg.TILT_DEG,
    )
    checker = ReadabilityChecker(
        ipm                   = ipm,
        min_char_height_px    = 15,   # seuil OCR minimal
        target_char_height_px = 22,   # seuil confortable
        step_length_m         = 0.75,
        avg_lines_in_sign     = 4,
    )

    # ── Init PaddleOCR ────────────────────────────────────────────────────
    ocr = PaddleOCR(
        lang                       = 'en',
        use_doc_orientation_classify = False,
        use_doc_unwarping          = False,
        use_textline_orientation   = False,
    )

    # ── Init caméra ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(cfg.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.IMG_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.IMG_H)

    log.info("Démarrage — appuyez sur 'q' pour quitter, 'c' pour recalibrer l'horizon")

    frame_count     = 0
    last_voice_msg  = ""  # anti-spam : ne pas répéter le même message

    while True:
        ret, frame = cap.read()
        if not ret:
            log.error("Impossible de lire la caméra")
            break

        frame_count += 1

        # ── Recalibration horizon automatique toutes les 60 frames ────────
        if frame_count % 60 == 0:
            ipm.calibrate_horizon(frame)

        # ── Dessin des zones IPM (debug) ───────────────────────────────────
        display_frame = ipm.draw_distance_zones(frame) if cfg.DRAW_ZONES else frame.copy()

        # ── OCR via fichier temporaire (requis par PaddleOCR 3.x) ─────────
        tmp_path      = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
        annotated_path = tmp_path + "_out.jpg"
        cv2.imwrite(tmp_path, frame)

        try:
            results = ocr.predict(tmp_path)
        except Exception as e:
            log.warning(f"Erreur OCR : {e}")
            os.remove(tmp_path)
            continue

        # ── Sauvegarde image annotée PaddleOCR ────────────────────────────
        results[0].save_to_img(annotated_path)

        # ── Traitement des résultats ───────────────────────────────────────
        if results:
            for res in results:
                data      = res.json.get("res", {})
                rec_texts = data.get("rec_texts", [])
                dt_polys  = data.get("dt_polys",  [])
                scores    = data.get("rec_scores", [])

                if not rec_texts or not dt_polys:
                    continue

                # ── Calcul des aires (votre logique originale conservée) ───
                areas = [polygon_area(poly) for poly in dt_polys]
                log.debug(f"Aires: {[f'{a:.0f}' for a in areas]}")
                log.debug(f"Textes: {rec_texts}")

                # ── Décision IPM sur la bbox globale du bloc de texte ──────
                global_bbox = get_global_text_bbox(dt_polys)

                if global_bbox:
                    decision = checker.check(
                        bbox        = global_bbox,
                        frame       = frame,
                        object_type = ObjectType.VERTICAL,
                    )

                    # ── Affichage de la décision ───────────────────────────
                    log.debug(
                        f"IPM → dist={decision.distance_result.distance_m:.2f}m "
                        f"zone={decision.distance_result.zone} "
                        f"char_h={decision.char_height_px:.1f}px "
                        f"ocr={decision.should_ocr} "
                        f"steps={decision.steps_to_walk}"
                    )

                    # ── Message vocal (anti-spam) ──────────────────────────
                    if decision.voice_message != last_voice_msg:
                        log.info(f"[TTS] → {decision.voice_message}")
                        last_voice_msg = decision.voice_message
                        # TODO : passer decision.voice_message à votre module TTS
                        # tts_module.speak(decision.voice_message)

                    # ── Annotation visuelle sur la frame ──────────────────
                    display_frame = annotate_frame(display_frame, decision, global_bbox)

                    # ── Lancer l'OCR seulement si lisible ─────────────────
                    if decision.should_ocr:
                        log.info(f"[OCR] Texte lu : {rec_texts}")
                        # TODO : passer les textes au module TTS
                        # tts_module.speak(" ".join(rec_texts))

                        # Afficher chaque ligne avec sa bbox individuelle
                        for poly, text, score in zip(dt_polys, rec_texts, scores):
                            if score < 0.6:
                                continue  # ignorer les détections peu fiables
                            line_bbox = paddle_poly_to_bbox(poly)

                            # Distance individuelle de cette ligne
                            line_dist = ipm.estimate_ground_distance(line_bbox.y_max)
                            log.debug(
                                f"  '{text}' (conf={score:.2f}) "
                                f"dist_estimée={line_dist.distance_m:.1f}m"
                            )

        # ── Affichage ──────────────────────────────────────────────────────
        # Charger l'image annotée PaddleOCR et la combiner
        paddle_out = cv2.imread(annotated_path)
        if paddle_out is not None:
            # Forcer la même taille que display_frame (PaddleOCR peut redimensionner)
            if paddle_out.shape != display_frame.shape:
                paddle_out = cv2.resize(paddle_out, (display_frame.shape[1], display_frame.shape[0]))
            alpha = 0.6
            combined = cv2.addWeighted(paddle_out, alpha, display_frame, 1 - alpha, 0)
        else:
            combined = display_frame

        cv2.imshow("OCR + IPM", combined)

        # ── Touches de contrôle ────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            ipm.calibrate_horizon(frame)
            log.info(f"Horizon recalibré manuellement : {ipm.horizon_y}px")
        elif key == ord('d'):
            cfg.DRAW_ZONES = not cfg.DRAW_ZONES
            log.info(f"Zones IPM : {'ON' if cfg.DRAW_ZONES else 'OFF'}")

        # ── Nettoyage ──────────────────────────────────────────────────────
        for path in (tmp_path, annotated_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()