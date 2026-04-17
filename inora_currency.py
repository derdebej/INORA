"""
INORA - Module Currency Detection
===================================
Projet : Lunettes intelligentes pour non-voyants
Auteur  : (camarade) — adapté pour intégration INORA
Basé sur : YOLOv8 (ultralytics)

Installation :
  pip install ultralytics

Modèle :
  Placer banknotes.pt dans le dossier INORA

Usage depuis main.py :
    currency = INORACurrency()
    result = currency.process(frame)
    if result:
        tts.say(result["tts_message"], priority="high")
"""

import cv2
import logging
import time

log = logging.getLogger("INORA_CURRENCY")

# Couleurs par coupure (pour l'affichage)
COLORS = {
    "5":  (0, 200, 0),
    "10": (255, 100, 0),
    "20": (0, 165, 255),
    "50": (0, 0, 220),
}

# Noms des coupures pour le TTS
NAMES_FR = {
    "5":  "cinq dinars",
    "10": "dix dinars",
    "20": "vingt dinars",
    "50": "cinquante dinars",
}
NAMES_EN = {
    "5":  "five dinars",
    "10": "ten dinars",
    "20": "twenty dinars",
    "50": "fifty dinars",
}


class INORACurrency:
    """
    Détection et identification des billets de banque.

    Retourne un dict avec :
      counts      : { "10": 2, "20": 1 }
      total       : 40.0
      tts_message : "Deux billets de dix dinars, un billet de vingt dinars. Total : quarante dinars."
      frame       : frame annotée avec les boîtes de détection
    """

    def __init__(
        self,
        model_path: str = "banknotes.pt",
        confidence: float = 0.5,
        lang: str = "fr",
        repeat_delay: float = 3.0,
    ):
        """
        Paramètres
        ----------
        model_path  : chemin vers le fichier .pt du modèle YOLO
        confidence  : seuil de confiance (0.0 à 1.0)
        lang        : langue pour le message TTS ("fr" ou "en")
        repeat_delay: secondes avant de répéter le même résultat
        """
        self.confidence   = confidence
        self.lang         = lang
        self.repeat_delay = repeat_delay

        self._last_result = {}    # dernier résultat envoyé au TTS
        self._last_time   = 0.0

        log.info(f"Chargement modèle YOLO : {model_path}")
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        log.info("Modèle currency prêt.")

    # ── API publique ─────────────────────────────────────────────────────────

    def process(self, frame) -> dict | None:
        """
        Analyse une frame et retourne les billets détectés.

        Retourne None si :
          - Aucun billet détecté
          - Même résultat qu'il y a moins de repeat_delay secondes

        Retourne un dict sinon :
          {
            "counts":      { "10": 2, "20": 1 },
            "total":       40.0,
            "tts_message": "...",
            "frame":       frame annotée
          }
        """
        results = self._model(frame, conf=self.confidence, verbose=False)
        result  = results[0]

        counts = {}
        total  = 0.0

        for box in result.boxes:
            cls_name = self._model.names[int(box.cls[0])]
            conf     = float(box.conf[0])
            value    = float(cls_name)

            counts[cls_name] = counts.get(cls_name, 0) + 1
            total += value

            # Annotation frame
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            color = COLORS.get(cls_name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} dt {conf:.0%}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + w + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Barre total en haut
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(frame, f"TOTAL: {total:.0f} dt  |  Notes: {len(result.boxes)}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 180), 2)

        # Rien détecté
        if not counts:
            return None

        # Anti-spam
        now = time.time()
        if counts == self._last_result and now - self._last_time < self.repeat_delay:
            return None

        self._last_result = counts.copy()
        self._last_time   = now

        tts_message = self._build_tts_message(counts, total)
        log.info(f"Currency détectée : {counts} — total {total:.0f} dt")

        return {
            "counts":      counts,
            "total":       total,
            "tts_message": tts_message,
            "frame":       frame,
        }

    # ── Construction du message TTS ──────────────────────────────────────────

    def _build_tts_message(self, counts: dict, total: float) -> str:
        """
        Construit un message naturel pour le TTS.

        Exemple FR : "Deux billets de dix dinars, un billet de vingt dinars. Total : quarante dinars."
        Exemple EN : "Two ten dinar notes, one twenty dinar note. Total: forty dinars."
        """
        names   = NAMES_FR if self.lang == "fr" else NAMES_EN
        numbers = self._number_words()
        parts   = []

        for note, qty in sorted(counts.items(), key=lambda x: float(x[0])):
            name   = names.get(note, f"{note} dinars")
            qty_word = numbers.get(qty, str(qty))

            if self.lang == "fr":
                billet = "billet" if qty == 1 else "billets"
                parts.append(f"{qty_word} {billet} de {name}")
            else:
                note_word = "note" if qty == 1 else "notes"
                parts.append(f"{qty_word} {name} {note_word}")

        detail = ", ".join(parts)
        total_word = self._total_words(total)

        if self.lang == "fr":
            return f"{detail}. Total : {total_word} dinars."
        else:
            return f"{detail}. Total: {total_word} dinars."

    def _number_words(self) -> dict:
        if self.lang == "fr":
            return {1: "un", 2: "deux", 3: "trois", 4: "quatre", 5: "cinq",
                    6: "six", 7: "sept", 8: "huit", 9: "neuf", 10: "dix"}
        else:
            return {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

    def _total_words(self, total: float) -> str:
        """Convertit le total numérique en mot."""
        words_fr = {5: "cinq", 10: "dix", 15: "quinze", 20: "vingt",
                    25: "vingt-cinq", 30: "trente", 40: "quarante",
                    50: "cinquante", 60: "soixante", 70: "soixante-dix",
                    80: "quatre-vingts", 100: "cent"}
        words_en = {5: "five", 10: "ten", 15: "fifteen", 20: "twenty",
                    25: "twenty-five", 30: "thirty", 40: "forty",
                    50: "fifty", 60: "sixty", 70: "seventy",
                    80: "eighty", 100: "one hundred"}
        words = words_fr if self.lang == "fr" else words_en
        return words.get(int(total), str(int(total)))