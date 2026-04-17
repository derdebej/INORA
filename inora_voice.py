"""
INORA - Module Voice Recognition
==================================
Projet : Lunettes intelligentes pour non-voyants
Moteur  : Whisper (OpenAI, offline, très précis)
Langues : Français / Anglais

Installation :
  pip install openai-whisper sounddevice numpy

Utilisation :
  voice = INORAVoice(lang="fr", command_handler=handle_command)
  voice.start()
"""

import threading
import time
import logging
import numpy as np
import sounddevice as sd
import whisper

log = logging.getLogger("INORA_VOICE")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
RECORD_SECONDS = 3      # durée d'enregistrement par cycle
MIC_DEVICE    = 2       # device ID du micro — changer si nécessaire
MODEL_SIZE    = "small" # tiny / base / small / medium

# ─────────────────────────────────────────────────────────────────────────────
# Commandes reconnues
# ─────────────────────────────────────────────────────────────────────────────
COMMANDS_FR = {
    "inora":        "wake",
    "lis":          "read",
    "stop":         "stop",
    "silence":      "stop",
    "répète":       "repeat",
    "activer":      "toggle_on",
    "démarrer":     "toggle_on",
    "désactiver":   "toggle_off",
    "pause":        "toggle_off",
    "anglais":      "lang_en",
    "français":     "lang_fr",
    "quitter":      "quit",
}

COMMANDS_EN = {
    "inora":        "wake",
    "read":         "read",
    "stop":         "stop",
    "silence":      "stop",
    "repeat":       "repeat",
    "activate":     "toggle_on",
    "start":        "toggle_on",
    "deactivate":   "toggle_off",
    "pause":        "toggle_off",
    "english":      "lang_en",
    "french":       "lang_fr",
    "quit":         "quit",
    "exit":         "quit",
}

# Phrases naturelles reconnues après le mot-clé "INORA"
NATURAL_FR = {
    "que vois tu":           "what_do_you_see",
    "qu est ce que tu vois": "what_do_you_see",
    "lis le texte":          "read",
    "quelle heure est il":   "what_time",
}

NATURAL_EN = {
    "what do you see":       "what_do_you_see",
    "read the text":         "read",
    "what time is it":       "what_time",
}


# ─────────────────────────────────────────────────────────────────────────────
# INORAVoice
# ─────────────────────────────────────────────────────────────────────────────
class INORAVoice:
    """
    Module de reconnaissance vocale INORA basé sur Whisper.

    Fonctionnement :
      - Enregistre RECORD_SECONDS secondes en boucle
      - Transcrit avec Whisper (offline, très précis)
      - Détecte commandes fixes et langage naturel après "INORA"
    """

    WAKE_TIMEOUT = 6.0   # secondes d'écoute naturelle après "INORA"

    def __init__(self, lang: str = "fr", command_handler=None):
        self.lang            = lang
        self.command_handler = command_handler
        self._stop_event     = threading.Event()
        self._wake_until     = 0.0
        self._last_text      = ""
        self._model          = None
        self._ready          = False

        self._load_model()

    # ── Chargement ───────────────────────────────────────────────────────────

    def _load_model(self):
        try:
            log.info(f"Chargement Whisper '{MODEL_SIZE}'...")
            self._model = whisper.load_model(MODEL_SIZE)
            self._ready = True
            log.info("Whisper prêt.")
        except Exception as e:
            log.error(f"Impossible de charger Whisper : {e}")

    # ── API publique ─────────────────────────────────────────────────────────

    def start(self):
        """Démarre l'écoute en arrière-plan."""
        if not self._ready:
            log.error("Whisper non initialisé.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="Voice-Listener"
        )
        self._thread.start()
        log.info(f"Écoute vocale démarrée — langue : {self.lang.upper()}")

    def stop(self):
        """Arrête l'écoute."""
        self._stop_event.set()
        if hasattr(self, "_thread"):
            self._thread.join(timeout=RECORD_SECONDS + 1)
        log.info("Écoute vocale arrêtée.")

    def set_language(self, lang: str):
        if lang != self.lang:
            self.lang = lang
            log.info(f"Langue vocale : {lang.upper()}")

    def set_last_text(self, text: str):
        """Mémorise le dernier texte OCR lu (pour la commande répète)."""
        self._last_text = text

    @property
    def ready(self) -> bool:
        return self._ready

    # ── Boucle d'écoute ──────────────────────────────────────────────────────

    def _listen_loop(self):
        """Enregistre et transcrit en boucle continue."""
        log.info("Micro actif — en écoute...")
        while not self._stop_event.is_set():
            try:
                # Enregistrement
                audio = sd.rec(
                    int(RECORD_SECONDS * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    device=2,
                )
                sd.wait()

                if self._stop_event.is_set():
                    break

                audio = audio.flatten()

                # Ignore si trop silencieux
                if np.abs(audio).mean() < 0.002:
                    continue

                # Transcription Whisper
                result = self._model.transcribe(
                    audio,
                    language=self.lang,
                    fp16=False,
                    condition_on_previous_text=False,
                )
                text = result["text"].strip().lower()

                if text:
                    log.info(f"Reconnu : {text!r}")
                    self._handle_text(text)

            except Exception as e:
                log.error(f"Erreur écoute : {e}")
                time.sleep(0.5)

    # ── Traitement ───────────────────────────────────────────────────────────

    def _handle_text(self, text: str):
        """Analyse le texte et décide de l'action."""
        commands = COMMANDS_FR if self.lang == "fr" else COMMANDS_EN
        natural  = NATURAL_FR  if self.lang == "fr" else NATURAL_EN

        # Cherche une commande fixe mot par mot
        words = text.split()
        for word in words:
            if word in commands:
                action = commands[word]

                if action == "wake":
                    self._wake_until = time.time() + self.WAKE_TIMEOUT
                    log.info(f"Mot-clé INORA — écoute naturelle {self.WAKE_TIMEOUT}s")
                    if self.command_handler:
                        self.command_handler("wake", text)
                    return

                if action == "repeat":
                    if self.command_handler:
                        self.command_handler("repeat", self._last_text)
                    return

                log.info(f"Commande : {action!r}")
                if self.command_handler:
                    self.command_handler(action, text)
                return

        # Mode wake — cherche une phrase naturelle
        if time.time() < self._wake_until:
            for phrase, action in natural.items():
                if phrase in text:
                    log.info(f"Langage naturel : {action!r}")
                    if self.command_handler:
                        self.command_handler(action, text)
                    self._wake_until = 0.0
                    return

            # Phrase non reconnue
            log.info(f"Phrase naturelle non reconnue : {text!r}")
            if self.command_handler:
                self.command_handler("natural", text)
            self._wake_until = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test rapide
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[VOICE] %(levelname)s: %(message)s")

    def on_command(action, phrase):
        print(f"\n  ACTION : {action!r}")
        print(f"  PHRASE : {phrase!r}\n")

    print("=== Test Whisper — parle dans le micro ===")
    print(f"Micro : device {MIC_DEVICE}")
    print("Essaie : 'INORA' / 'lis' / 'stop' / 'activer'")
    print("Ctrl+C pour quitter\n")

    voice = INORAVoice(lang="fr", command_handler=on_command)
    voice.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    voice.stop()
    print("=== Fin ===")