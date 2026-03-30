"""
INORA - Module Voice Recognition
==================================
Projet : Lunettes intelligentes pour non-voyants
Moteur  : Vosk (offline, rapide, Jetson compatible)
Langues : Français / Anglais

Installation :
  pip install vosk sounddevice

Modèles à placer dans le dossier INORA :
  FR : vosk-model-fr/   (téléchargé depuis alphacephei.com)
  EN : vosk-model-en/

Commandes reconnues :
  FR : "INORA" / "lis" / "stop" / "silence" / "répète" / "anglais" / "français"
  EN : "INORA" / "read" / "stop" / "silence" / "repeat" / "french" / "english"
"""

import threading
import queue
import json
import logging
import time

import sounddevice as sd
from vosk import Model, KaldiRecognizer

log = logging.getLogger("INORA_VOICE")

# ─────────────────────────────────────────────────────────────────────────────
# Chemins des modèles Vosk
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATHS = {
    "fr": "vosk-model-fr",
    "en": "vosk-model-en",
}

# ─────────────────────────────────────────────────────────────────────────────
# Commandes fixes reconnues (mots-clés exacts)
# ─────────────────────────────────────────────────────────────────────────────
COMMANDS_FR = {
    # Mot déclencheur
    "inora":        "wake",       # active l'écoute des commandes

    # Contrôle TTS/OCR
    "lis":          "read",       # déclenche la lecture OCR immédiate
    "stop":         "stop",       # arrête la lecture en cours
    "silence":      "stop",       # alias stop
    "répète":       "repeat",     # répète le dernier texte lu

    # Activer / Désactiver INORA
    "activer":      "toggle_on",  # active le TTS + reset OCR
    "démarrer":     "toggle_on",  # alias activer
    "désactiver":   "toggle_off", # désactive le TTS
    "pause":        "toggle_off", # alias désactiver

    # Langue
    "anglais":      "lang_en",    # passe en anglais
    "français":     "lang_fr",    # passe en français

    # Navigation
    "quitter":      "quit",       # arrête INORA
}

COMMANDS_EN = {
    "inora":        "wake",
    "read":         "read",
    "stop":         "stop",
    "silence":      "stop",
    "repeat":       "repeat",

    # Activate / Deactivate
    "activate":     "toggle_on",
    "start":        "toggle_on",
    "deactivate":   "toggle_off",
    "pause":        "toggle_off",

    "french":       "lang_fr",
    "english":      "lang_en",
    "quit":         "quit",
    "exit":         "quit",
}


# ─────────────────────────────────────────────────────────────────────────────
# INORAVoice
# ─────────────────────────────────────────────────────────────────────────────
class INORAVoice:
    """
    Module de reconnaissance vocale pour INORA.

    Fonctionnement :
      - Écoute en continu via le micro
      - Détecte les commandes fixes immédiatement
      - Détecte le mot-clé "INORA" → active l'écoute de langage naturel
      - En mode langage naturel : envoie la phrase complète au handler

    Usage dans main.py :
        voice = INORAVoice(lang="fr", command_handler=handle_command)
        voice.start()
        # ... boucle principale ...
        voice.stop()
    """

    SAMPLE_RATE  = 16000   # Vosk requiert 16kHz
    BLOCK_SIZE   = 8000    # ~500ms par bloc audio
    WAKE_TIMEOUT = 5.0     # secondes d'écoute après le mot-clé

    def __init__(self, lang: str = "fr", command_handler=None):
        """
        Paramètres
        ----------
        lang            : "fr" ou "en"
        command_handler : fonction appelée avec (action, phrase)
                          action = commande fixe ou "natural" pour langage libre
        """
        self.lang            = lang
        self.command_handler = command_handler

        self._stop_event  = threading.Event()
        self._audio_queue = queue.Queue()
        self._wake_until  = 0.0    # timestamp fin d'écoute naturelle
        self._last_text   = ""     # dernier texte reconnu (pour repeat)
        self._model       = None
        self._recognizer  = None
        self._ready       = False

        self._load_model(lang)

    # ── Chargement modèle ────────────────────────────────────────────────────

    # Corrections phonétiques — variantes mal reconnues → mot correct
    PHONETIC_FR = {
        # INORA mal transcrit
        "il n'aura":    "inora",
        "il nora":      "inora",
        "in aura":      "inora",
        "inora":        "inora",
        "il laura":     "inora",
        "il n aura":    "inora",
        "énora":        "inora",
        "ainora":       "inora",
        # Commandes mal transcrites
        "lise":         "lis",
        "lit":         "lis",
        "lys":          "lis",
        "répète":       "répète",
        "répètes":      "répète",
        "répète moi":   "répète",
        "active":       "activer",
        "activez":       "activer",
        "activé":       "activer",
        "désactive":    "désactiver",
        "désactivé":    "désactiver",
    }

    PHONETIC_EN = {
        # INORA mal transcrit
        "in aura":      "inora",
        "il nora":      "inora",
        "a nora":       "inora",
        "enora":        "inora",
        # Commandes
        "reads":        "read",
        "stops":        "stop",
        "repeats":      "repeat",
        "repeated":     "repeat",
        "activated":    "activate",
        "deactivated":  "deactivate",
    }

    def _load_model(self, lang: str):
        try:
            model_path = MODEL_PATHS.get(lang)
            log.info(f"Chargement modèle Vosk : {model_path}")
            self._model      = Model(model_path)
            self._recognizer = KaldiRecognizer(self._model, self.SAMPLE_RATE)
            self._recognizer.SetWords(True)
            self._ready = True
            log.info("Vosk prêt.")
        except Exception as e:
            log.error(f"Impossible de charger Vosk : {e}")
            self._ready = False

    def _correct_phonetics(self, text: str) -> str:
        """
        Corrige les erreurs phonétiques courantes de Vosk.
        Cherche les variantes connues dans le texte et les remplace
        par le mot correct avant l'analyse des commandes.
        """
        corrections = self.PHONETIC_FR if self.lang == "fr" else self.PHONETIC_EN
        corrected = text
        for wrong, right in corrections.items():
            if wrong in corrected:
                corrected = corrected.replace(wrong, right)
                log.debug(f"Correction phonétique : {wrong!r} → {right!r}")
        return corrected

    # ── API publique ─────────────────────────────────────────────────────────

    def start(self):
        """Démarre l'écoute en arrière-plan."""
        if not self._ready:
            log.error("Vosk non initialisé — reconnaissance vocale désactivée.")
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
            self._thread.join(timeout=2)
        log.info("Écoute vocale arrêtée.")

    def set_language(self, lang: str):
        """Change la langue de reconnaissance à chaud."""
        if lang == self.lang:
            return
        self.lang = lang
        self.stop()
        self._load_model(lang)
        self.start()
        log.info(f"Langue vocale : {lang.upper()}")

    def set_last_text(self, text: str):
        """Mémorise le dernier texte lu (pour la commande 'répète')."""
        self._last_text = text

    @property
    def ready(self) -> bool:
        return self._ready

    # ── Boucle d'écoute ──────────────────────────────────────────────────────

    def _listen_loop(self):
        """Capture audio + reconnaissance en continu."""
        def audio_callback(indata, frames, time_info, status):
            self._audio_queue.put(bytes(indata))

        with sd.RawInputStream(
            samplerate=self.SAMPLE_RATE,
            blocksize=self.BLOCK_SIZE,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        ):
            log.info("Micro actif — en écoute...")
            while not self._stop_event.is_set():
                try:
                    data = self._audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if self._recognizer.AcceptWaveform(data):
                    result = json.loads(self._recognizer.Result())
                    text   = result.get("text", "").strip().lower()
                    if text:
                        log.info(f"Reconnu : {text!r}")
                        self._handle_text(text)

    # ── Traitement du texte reconnu ──────────────────────────────────────────

    def _handle_text(self, text: str):
        """
        Analyse le texte reconnu et décide de l'action :
          1. Corrige les erreurs phonétiques connues
          2. Cherche une commande fixe mot par mot
          3. Si en mode wake → envoie comme langage naturel
          4. Si mot-clé "inora" → active le mode wake
        """
        # Correction phonétique avant analyse
        text     = self._correct_phonetics(text)
        commands = COMMANDS_FR if self.lang == "fr" else COMMANDS_EN
        words    = text.split()

        # Cherche une commande fixe dans les mots reconnus
        for word in words:
            if word in commands:
                action = commands[word]

                if action == "wake":
                    # Active l'écoute naturelle pendant WAKE_TIMEOUT secondes
                    self._wake_until = time.time() + self.WAKE_TIMEOUT
                    log.info(f"Mot-clé détecté — écoute naturelle activée ({self.WAKE_TIMEOUT}s)")
                    if self.command_handler:
                        self.command_handler("wake", text)
                    return

                if action == "repeat":
                    # Répète le dernier texte lu
                    if self.command_handler:
                        self.command_handler("repeat", self._last_text)
                    return

                # Toute autre commande fixe
                log.info(f"Commande fixe : {action!r}")
                if self.command_handler:
                    self.command_handler(action, text)
                return

        # Aucune commande fixe — vérifie si on est en mode wake (langage naturel)
        if time.time() < self._wake_until:
            log.info(f"Langage naturel : {text!r}")
            if self.command_handler:
                self.command_handler("natural", text)
            self._wake_until = 0.0   # désactive le mode wake après une phrase


# ─────────────────────────────────────────────────────────────────────────────
# Test rapide
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[TTS] %(levelname)s: %(message)s")

    def on_command(action, phrase):
        print(f"\n  ACTION : {action!r}")
        print(f"  PHRASE : {phrase!r}\n")

    print("=== Test Vosk — parle dans le micro ===")
    print("Essaie : 'INORA' puis une phrase")
    print("         'lis' / 'stop' / 'répète'")
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