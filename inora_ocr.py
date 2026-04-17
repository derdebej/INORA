"""
INORA - Module OCR
===================
Projet : Lunettes intelligentes pour non-voyants
Basé sur : PaddleOCR 3.4.0

Stabilisation :
  - Le texte doit apparaître dans N frames consécutives avant d'être validé
  - Seuil de confiance configurable
  - Anti-spam : ne répète pas le même texte avant X secondes

Installation :
  pip install paddleocr opencv-python
  pip install paddlepaddle-gpu
"""

import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["GLOG_minloglevel"] = "3"

import cv2
import tempfile
import logging
import time
from collections import Counter

log = logging.getLogger("INORA_OCR")


class INORAOcr:
    """
    Wrapper PaddleOCR stabilisé pour le projet INORA.

    Stabilisation par vote majoritaire :
      Le texte doit être détecté de manière cohérente pendant
      `stability_frames` frames consécutives avant d'être envoyé au TTS.
      Cela élimine les lectures parasites et les variations frame-à-frame.

    Usage depuis main.py :
        ocr = INORAOcr(lang="en")
        text, annotated_frame = ocr.process(frame)
        if text:
            tts.say(...)
    """

    def __init__(
        self,
        lang: str = "en",
        confidence_threshold: float = 0.9,   # seuil de confiance élevé
        stability_frames: int = 5,            # frames consécutives requises
        stability_duration: float = 2.0,     # secondes de stabilité requises
        repeat_delay: float = 10.0,           # secondes avant de répéter
    ):
        """
        Paramètres
        ----------
        lang                 : langue OCR ("en" ou "fr")
        confidence_threshold : score minimum (0-1) — 0.9 = très strict
        stability_frames     : nombre de frames similaires requises
        stability_duration   : secondes pendant lesquelles le texte doit
                               rester stable avant d'être envoyé au TTS
        repeat_delay         : secondes avant de répéter le même texte
        """
        self.lang                 = lang
        self.confidence_threshold = confidence_threshold
        self.stability_frames     = stability_frames
        self.stability_duration   = stability_duration
        self.repeat_delay         = repeat_delay

        # Buffer de stabilisation — dernières N détections
        self._frame_buffer  = []    # textes détectés récemment
        self._stable_text   = ""    # texte stable en cours de validation
        self._stable_since  = None  # timestamp début de stabilité
        self._last_sent     = ""    # dernier texte envoyé au TTS
        self._last_time     = 0.0   # timestamp du dernier envoi

        log.info(f"Chargement PaddleOCR (lang={lang})...")
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        log.info("PaddleOCR prêt.")

    # ── API publique ─────────────────────────────────────────────────────────

    def process(self, frame) -> tuple:
        """
        Analyse une frame et retourne le texte stabilisé + la frame annotée.

        Retourne
        --------
        (text, annotated_frame)
          text            : str — texte validé et prêt pour le TTS, ou ""
          annotated_frame : numpy array — frame avec boîtes de détection
        """
        tmp_in  = None
        tmp_out = None

        try:
            # PaddleOCR 3.x nécessite un chemin fichier
            tmp_in = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
            cv2.imwrite(tmp_in, frame)
            results = self._ocr.predict(tmp_in)

            # Frame annotée
            tmp_out = tmp_in + "_annotated.jpg"
            results[0].save_to_img(tmp_out)
            annotated_frame = cv2.imread(tmp_out)
            if annotated_frame is None:
                annotated_frame = frame

            # Extraction brute du texte de cette frame
            raw_text = self._extract_text(results)

            # Stabilisation — retourne le texte seulement si stable
            validated_text = self._stabilize(raw_text)

            return validated_text, annotated_frame

        except Exception as e:
            log.error(f"Erreur OCR : {e}")
            return "", frame

        finally:
            for path in (tmp_in, tmp_out):
                if path:
                    try:
                        os.remove(path)
                    except Exception:
                        pass

    # ── Extraction ───────────────────────────────────────────────────────────

    def _extract_text(self, results) -> str:
        """
        Extrait le texte des résultats PaddleOCR 3.4.0.
        Structure : results[0].json['res']['rec_texts'] / ['rec_scores']
        """
        words = []
        try:
            for res in results:
                data       = res.json.get("res", {})
                rec_texts  = data.get("rec_texts", [])
                rec_scores = data.get("rec_scores", [])
                for txt, score in zip(rec_texts, rec_scores):
                    if score >= self.confidence_threshold and txt.strip():
                        words.append(txt.strip())
        except Exception as e:
            log.warning(f"Erreur parsing OCR : {e}")
        return " ".join(words)

    # ── Stabilisation ────────────────────────────────────────────────────────

    def _similarity(self, a: str, b: str) -> float:
        """
        Calcule la similarité entre deux textes (0.0 à 1.0).
        Basé sur les mots communs — robuste aux petites variations OCR.
        """
        if not a or not b:
            return 0.0
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        common  = words_a & words_b
        return len(common) / max(len(words_a), len(words_b))

    def _stabilize(self, raw_text: str) -> str:
        """
        Collecte tous les résultats OCR pendant stability_duration secondes,
        puis choisit et envoie le meilleur au TTS.

        Algorithme :
          1. Collecte toutes les détections non-vides dans _frame_buffer
          2. Après stability_duration secondes → choisit le meilleur texte :
               - Trouve le groupe de textes les plus similaires entre eux
               - Dans ce groupe, prend le plus long (le plus complet)
          3. Anti-spam : ne répète pas avant repeat_delay secondes
          4. Reset le timer et recommence la collecte
        """
        now = time.time()

        # Initialise le timer au premier appel
        if self._stable_since is None:
            self._stable_since = now

        elapsed = now - self._stable_since

        # ── 3 premières secondes ignorées (stabilisation caméra) ─────────────
        WARMUP = 3.0
        if elapsed < WARMUP:
            log.debug(f"Warmup {elapsed:.1f}s/{WARMUP}s — en attente...")
            return ""

        # Collecte les détections non-vides (après warmup seulement)
        if raw_text:
            self._frame_buffer.append(raw_text)

        log.debug(f"Collecte {elapsed:.1f}s/{self.stability_duration + WARMUP}s — buffer: {len(self._frame_buffer)} détections")

        # Pas encore atteint la durée de collecte
        if elapsed < self.stability_duration + WARMUP:
            return ""

        # ── Fin de la fenêtre de collecte → choisit le meilleur ─────────────
        if not self._frame_buffer:
            # Rien détecté pendant la fenêtre → reset et recommence
            self._stable_since = now
            return ""

        best = self._pick_best(self._frame_buffer)

        # Reset pour la prochaine fenêtre
        self._frame_buffer = []
        self._stable_since = now

        if not best:
            return ""

        # Anti-spam
        if (self._similarity(best, self._last_sent) > 0.85 and
                now - self._last_time < self.repeat_delay):
            log.debug(f"Anti-spam : texte déjà lu récemment : {best!r}")
            return ""

        self._last_sent = best
        self._last_time = now
        log.info(f"Meilleur texte après {self.stability_duration}s : {best!r}")
        return best

    def _pick_best(self, candidates: list) -> str:
        """
        Parmi tous les textes collectés, choisit le meilleur.

        Stratégie :
          1. Groupe les textes similaires (similarité > 70%)
          2. Prend le groupe le plus fréquent (le plus fiable)
          3. Dans ce groupe, retourne le texte le plus long (le plus complet)
        """
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]

        # Clustering simple par similarité
        groups = []
        for text in candidates:
            placed = False
            for group in groups:
                if self._similarity(text, group[0]) >= 0.7:
                    group.append(text)
                    placed = True
                    break
            if not placed:
                groups.append([text])

        # Groupe le plus fréquent
        best_group = max(groups, key=len)

        # Texte le plus long dans ce groupe = le plus complet
        best_text = max(best_group, key=len)
        log.debug(f"Groupe majoritaire : {len(best_group)}/{len(candidates)} détections → {best_text!r}")
        return best_text

    def reset(self):
        """Réinitialise le buffer de stabilisation (utile au changement de scène)."""
        self._frame_buffer  = []
        self._stable_text   = ""
        self._stable_since  = None
        self._last_sent     = ""
        self._last_time     = 0.0