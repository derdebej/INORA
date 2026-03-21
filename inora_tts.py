"""
INORA - Module Text-to-Speech
==============================
Projet : Lunettes intelligentes pour non-voyants
Moteur  : Piper TTS (offline, rapide, neural)
Langues : Français / Anglais

Installation :
  pip install piper-tts sounddevice numpy

Modèles à télécharger et placer dans le dossier INORA :
  FR : https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx
  EN : https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/jenny/medium/en_US-jenny-medium.onnx
"""

import threading
import queue
import time
import logging
import wave
import io
import numpy as np
import sounddevice as sd

logging.basicConfig(level=logging.INFO, format="[TTS] %(levelname)s: %(message)s")
log = logging.getLogger("INORA_TTS")

PRIORITY_URGENT = 0
PRIORITY_HIGH   = 1
PRIORITY_NORMAL = 2
PRIORITY_LOW    = 3

# Chemins des modèles piper — à adapter si placés ailleurs
MODEL_PATHS = {
    "fr": "fr_FR-siwis-medium.onnx",
    "en": "en_US-jenny-medium.onnx",
}


# ─────────────────────────────────────────────────────────────────────────────
# Génération + lecture audio (Piper)
# ─────────────────────────────────────────────────────────────────────────────
def speak_now(text: str, voice):
    """Génère via Piper et joue l'audio immédiatement. Bloquant."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        voice.synthesize_wav(text, wav)

    buf.seek(0)
    with wave.open(buf, "rb") as wav:
        frames = wav.readframes(wav.getnframes())
        rate   = wav.getframerate()

    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(data, rate)
    sd.wait()


# ─────────────────────────────────────────────────────────────────────────────
# INORASpeaker
# ─────────────────────────────────────────────────────────────────────────────
class INORASpeaker:
    """
    Interface TTS centrale du projet INORA.

    - Moteur    : Piper TTS (100% offline, rapide sur Jetson)
    - File      : priorité urgent > high > normal > low
    - Thread    : non-bloquant pour le reste du système
    - Toggle    : active/désactive le TTS + reset OCR
    - Interrupt : stoppe la lecture immédiatement
    """

    PRIORITY_MAP = {
        "urgent": PRIORITY_URGENT,
        "high":   PRIORITY_HIGH,
        "normal": PRIORITY_NORMAL,
        "low":    PRIORITY_LOW,
    }

    def __init__(self, lang: str = "fr"):
        self.lang        = lang
        self._queue      = queue.PriorityQueue()
        self._stop_event = threading.Event()
        self._enabled    = False   # désactivé par défaut — activer avec toggle()
        self._voice      = None

        self._load_model(lang)

        self._worker = threading.Thread(
            target=self._loop, daemon=True, name="TTS-Worker"
        )
        self._worker.start()
        log.info(f"INORASpeaker démarré — langue : {lang.upper()}")

    # ── Chargement modèle ────────────────────────────────────────────────────

    def _load_model(self, lang: str):
        try:
            from piper import PiperVoice
            model_path = MODEL_PATHS.get(lang)
            if not model_path:
                raise FileNotFoundError(f"Pas de modèle défini pour la langue : {lang}")
            log.info(f"Chargement modèle Piper : {model_path}")
            self._voice = PiperVoice.load(model_path)
            log.info("Modèle Piper prêt.")
        except Exception as e:
            log.error(f"Impossible de charger Piper : {e}")
            self._voice = None

    # ── API publique ─────────────────────────────────────────────────────────

    def say(self, text: str, priority: str = "normal"):
        """Ajoute un message à la file. Ignoré si TTS désactivé."""
        if not text or not text.strip():
            return
        if not self._enabled:
            log.debug(f"TTS désactivé — ignoré : {text!r}")
            return
        if self._voice is None:
            log.error("Modèle Piper non chargé.")
            return
        prio = self.PRIORITY_MAP.get(priority, PRIORITY_NORMAL)
        self._queue.put((prio, time.time(), text))

    def say_urgent(self, text: str):
        """Vide la file et met le message en priorité absolue."""
        self._flush_non_urgent()
        self.say(text, priority="urgent")

    def toggle(self, ocr=None):
        """
        Active ou désactive le TTS + OCR ensemble.
        Appelé par bouton physique / commande / geste.

        - Désactivé → Active  : reset OCR, annonce "INORA activé"
        - Activé    → Désactivé : interrompt lecture, annonce "INORA désactivé"

        Paramètres
        ----------
        ocr : INORAOcr — si fourni, reset le buffer au moment de l'activation
        """
        self._enabled = not self._enabled

        if self._enabled:
            if ocr is not None:
                ocr.reset()
                log.info("Buffer OCR réinitialisé.")
            log.info("TTS activé.")
            msg = "INORA activé, collecte en cours" if self.lang == "fr" else "INORA enabled, collecting"
            self._queue.put((PRIORITY_URGENT, time.time(), msg))
        else:
            self.interrupt()
            log.info("TTS désactivé.")
            # Joue l'annonce directement (hors file car _enabled=False)
            if self._voice:
                threading.Thread(
                    target=speak_now,
                    args=("INORA désactivé" if self.lang == "fr" else "INORA disabled", self._voice),
                    daemon=True
                ).start()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def interrupt(self):
        """Interrompt la lecture en cours et vide la file."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        try:
            sd.stop()
        except Exception:
            pass
        log.info("Lecture interrompue.")

    def set_language(self, lang: str):
        """Change la langue à chaud ('fr' ou 'en')."""
        if lang == self.lang:
            return
        self.lang = lang
        self._load_model(lang)
        log.info(f"Langue : {lang.upper()}")

    def stop(self):
        """Arrête proprement le thread TTS."""
        self._stop_event.set()
        self._worker.join(timeout=3)

    # ── Boucle interne ───────────────────────────────────────────────────────

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                _, ts, text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            log.info(f"Lecture : {text!r}")
            try:
                speak_now(text, self._voice)
            except Exception as e:
                log.error(f"Erreur TTS : {e}")
            finally:
                self._queue.task_done()

    def _flush_non_urgent(self):
        saved, n = [], 0
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item[0] == PRIORITY_URGENT:
                    saved.append(item)
                n += 1
                self._queue.task_done()
            except queue.Empty:
                break
        for item in saved:
            self._queue.put(item)
        if n:
            log.info(f"{n} message(s) supprimés.")


# ─────────────────────────────────────────────────────────────────────────────
# Messages standards
# ─────────────────────────────────────────────────────────────────────────────
class INORAMessages:
    FR = {
        "obstacle_near":    "Attention, obstacle à {dist} centimètres devant vous",
        "obstacle_left":    "Obstacle à gauche, à {dist} centimètres",
        "obstacle_right":   "Obstacle à droite, à {dist} centimètres",
        "obstacle_stairs":  "Attention, escalier devant vous",
        "obstacle_vehicle": "Attention, véhicule en approche",
        "clear_path":       "Voie libre",
        "coin_detected":    "{value}",
        "bill_detected":    "{value}",
        "no_currency":      "Aucune monnaie détectée",
        "ocr_reading":      "Texte détecté : {text}",
        "ocr_no_text":      "Aucun texte détecté",
        "system_ready":     "INORA prêt",
        "battery_low":      "Batterie faible, pensez à recharger",
    }

    EN = {
        "obstacle_near":    "Warning, obstacle {dist} centimeters ahead",
        "obstacle_left":    "Obstacle on your left, {dist} centimeters",
        "obstacle_right":   "Obstacle on your right, {dist} centimeters",
        "obstacle_stairs":  "Warning, stairs ahead",
        "obstacle_vehicle": "Warning, vehicle approaching",
        "clear_path":       "Path is clear",
        "coin_detected":    "{value}",
        "bill_detected":    "{value}",
        "no_currency":      "No currency detected",
        "ocr_reading":      "Text detected: {text}",
        "ocr_no_text":      "No text detected",
        "system_ready":     "INORA ready",
        "battery_low":      "Low battery, please recharge",
    }

    @classmethod
    def get(cls, key: str, lang: str = "fr", **kwargs) -> str:
        lib = cls.FR if lang == "fr" else cls.EN
        template = lib.get(key, key)
        return template.format(**kwargs) if kwargs else template


# ─────────────────────────────────────────────────────────────────────────────
# Test interactif
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Test INORA TTS (Piper) ===")
    print("Commandes : on / off / urgent / ocr / monnaie / interrompre / quitter")
    print("Ou tape directement un message\n")

    tts = INORASpeaker(lang="fr")
    msg = INORAMessages

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        elif text == "quitter":
            break
        elif text == "on":
            tts.toggle()
        elif text == "off":
            tts.toggle()
        elif text == "urgent":
            tts.say_urgent(msg.get("obstacle_near", "fr", dist=80))
        elif text == "ocr":
            tts.say(msg.get("ocr_reading", "fr", text="Sortie de secours niveau 2"))
        elif text == "monnaie":
            tts.say(msg.get("coin_detected", "fr", value="Pièce de deux euros"), priority="high")
        elif text == "interrompre":
            tts.interrupt()
        elif text == "en":
            tts.set_language("en")
        elif text == "fr":
            tts.set_language("fr")
        else:
            tts.say(text)

    tts.stop()
    print("=== Fin ===")