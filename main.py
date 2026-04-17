"""
INORA - Main
=============
Point d'entrée principal du projet.
Orchestre la caméra, le module OCR, TTS et reconnaissance vocale.

Lancement :
  python main.py

Contrôles :
  Clavier : Q → quitter | S → interrompre TTS | T → toggle TTS
  Voix    : "INORA" → écoute naturelle
            "lis"   → lecture OCR immédiate
            "stop"  → interrompt TTS
            "répète"→ répète le dernier texte
"""

import cv2
import logging
import time
import threading

from inora_ocr   import INORAOcr
from inora_tts   import INORASpeaker, INORAMessages
from inora_voice import INORAVoice

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(levelname)s: %(message)s"
)
log = logging.getLogger("INORA_MAIN")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration centrale
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "lang":             "fr",    # langue OCR + TTS + voix : "fr" ou "en"
    "camera_index":     0,       # 0 = webcam par défaut
    "ocr_confidence":   0.8,     # seuil de confiance OCR (0.0 à 1.0)
    "ocr_repeat_delay": 10.0,    # secondes avant de répéter le même texte
    "show_window":      True,    # afficher la fenêtre caméra annotée
    "interrupt_key":    "s",     # touche S → interrompt TTS
    "toggle_key":       "t",     # touche T → active/désactive TTS
}


# ─────────────────────────────────────────────────────────────────────────────
# Gestionnaire de commandes vocales
# ─────────────────────────────────────────────────────────────────────────────
def make_command_handler(tts: INORASpeaker, ocr: INORAOcr, voice: INORAVoice):
    """
    Retourne la fonction qui traite les commandes reconnues par Vosk.

    Actions possibles :
      wake    → confirme l'écoute ("Je t'écoute")
      read    → force la lecture du dernier texte OCR collecté
      stop    → interrompt le TTS
      repeat  → répète le dernier texte lu
      lang_fr → passe en français
      lang_en → passe en anglais
      quit    → arrête INORA
      natural → langage naturel (phrase complète après "INORA")
    """
    lang = CONFIG["lang"]

    # Réponses aux commandes naturelles courantes
    NATURAL_RESPONSES_FR = {
        "que vois tu":          lambda: tts.say(ocr._last_sent or "Je ne vois rien pour le moment"),
        "qu est ce que tu vois":lambda: tts.say(ocr._last_sent or "Je ne vois rien pour le moment"),
        "lis le texte":         lambda: tts.say(ocr._last_sent or "Aucun texte détecté"),
        "quelle heure est il":  lambda: tts.say(time.strftime("Il est %H heures %M")),
        "what time is it":      lambda: tts.say(time.strftime("It is %H:%M")),
    }

    def handle(action: str, phrase: str):
        nonlocal lang
        msg = INORAMessages

        if action == "wake":
            response = "Je t'écoute" if lang == "fr" else "I'm listening"
            tts._enabled = True
            tts.say(response, priority="urgent")

        elif action == "read":
            last = ocr._last_sent
            if last:
                tts._enabled = True
                tts.say(msg.get("ocr_reading", lang, text=last), priority="high")
            else:
                tts.say("Aucun texte disponible" if lang == "fr" else "No text available")

        elif action == "stop":
            tts.interrupt()

        elif action == "toggle_on":
            if not tts.enabled:
                tts.toggle(ocr=ocr)
                log.info("TTS activé par commande vocale.")
            else:
                tts.say("INORA est déjà actif" if lang == "fr" else "INORA is already active")

        elif action == "toggle_off":
            if tts.enabled:
                tts.toggle()
                log.info("TTS désactivé par commande vocale.")
            else:
                # Joue directement car TTS désactivé
                threading.Thread(
                    target=lambda: __import__('inora_tts').speak_now(
                        "INORA est déjà désactivé" if lang == "fr" else "INORA is already inactive",
                        tts.lang
                    ), daemon=True
                ).start()

        elif action == "repeat":
            if phrase:
                tts._enabled = True
                tts.say(phrase, priority="high")
            else:
                tts.say("Rien à répéter" if lang == "fr" else "Nothing to repeat")

        elif action == "lang_fr":
            lang = "fr"
            CONFIG["lang"] = "fr"
            tts.set_language("fr")
            ocr.lang = "fr"
            voice.set_language("fr")
            tts.say("Langue changée en français", priority="urgent")

        elif action == "lang_en":
            lang = "en"
            CONFIG["lang"] = "en"
            tts.set_language("en")
            ocr.lang = "en"
            voice.set_language("en")
            tts.say("Language switched to English", priority="urgent")

        elif action == "quit":
            tts.say("Au revoir" if lang == "fr" else "Goodbye", priority="urgent")
            time.sleep(2)
            import os; os._exit(0)

        elif action == "what_do_you_see":
            last = ocr._last_sent
            tts._enabled = True
            if last:
                tts.say(INORAMessages.get("ocr_reading", lang, text=last), priority="high")
            else:
                tts.say("Je ne vois rien pour le moment" if lang == "fr" else "I don't see anything", priority="normal")

        elif action == "what_time":
            tts._enabled = True
            tts.say(time.strftime("Il est %H heures %M") if lang == "fr" else time.strftime("It is %H:%M"), priority="normal")

        elif action == "natural":
            # Phrase non reconnue après mot-clé INORA
            response = "Je n'ai pas compris" if lang == "fr" else "I didn't understand"
            tts._enabled = True
            tts.say(response, priority="normal")

    return handle


# ─────────────────────────────────────────────────────────────────────────────
# Thread clavier (interruption / toggle)
# ─────────────────────────────────────────────────────────────────────────────
def listen_keyboard(tts: INORASpeaker, ocr: INORAOcr, stop_event: threading.Event):
    import sys
    try:
        import msvcrt
        while not stop_event.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                if key == CONFIG["interrupt_key"]:
                    tts.interrupt()
                elif key == CONFIG["toggle_key"]:
                    tts.toggle(ocr=ocr)
                    log.info(f"TTS {'activé' if tts.enabled else 'désactivé'}.")
            time.sleep(0.05)
    except ImportError:
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not stop_event.is_set():
                ch = sys.stdin.read(1).lower()
                if ch == CONFIG["interrupt_key"]:
                    tts.interrupt()
                elif ch == CONFIG["toggle_key"]:
                    tts.toggle(ocr=ocr)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("Démarrage INORA...")
    lang = CONFIG["lang"]

    # ── Initialisation TTS ────────────────────────────────────────────────────
    tts = INORASpeaker(lang=lang)
    tts._enabled = True
    tts.say(INORAMessages.get("system_ready", lang), priority="urgent")
    tts._enabled = False   # désactivé — l'utilisateur active avec T ou la voix

    # ── Initialisation OCR ────────────────────────────────────────────────────
    ocr = INORAOcr(
        lang=lang,
        confidence_threshold=CONFIG["ocr_confidence"],
        repeat_delay=CONFIG["ocr_repeat_delay"],
    )

    # ── Initialisation Voice ──────────────────────────────────────────────────
    voice = INORAVoice(lang=lang)
    handler = make_command_handler(tts, ocr, voice)
    voice.command_handler = handler
    voice.start()

    # ── Caméra ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    if not cap.isOpened():
        log.error("Impossible d'ouvrir la caméra.")
        tts.stop()
        voice.stop()
        return

    log.info("INORA prêt — commandes : Q=quitter | T=toggle | S=stop | Voix='INORA'")

    # ── Thread clavier ────────────────────────────────────────────────────────
    stop_event = threading.Event()
    threading.Thread(
        target=listen_keyboard,
        args=(tts, ocr, stop_event),
        daemon=True, name="Keyboard-Listener"
    ).start()

    # ── Boucle caméra ─────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # OCR
        text, annotated_frame = ocr.process(frame)
        if text:
            log.info(f"Texte détecté : {text!r}")
            voice.set_last_text(text)   # mémorise pour la commande "répète"
            message = INORAMessages.get("ocr_reading", lang, text=text)
            tts.say(message, priority="normal")

        # Affichage
        if CONFIG["show_window"]:
            cv2.imshow("INORA — Q:quitter | T:toggle | S:stop | Voix:'INORA'", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("Arrêt demandé.")
                break

    # ── Nettoyage ─────────────────────────────────────────────────────────────
    stop_event.set()
    voice.stop()
    cap.release()
    cv2.destroyAllWindows()
    tts.stop()
    log.info("INORA arrêté proprement.")


# ─────────────────────────────────────────────────────────────────────────────
# Intégration modules camarades
# ─────────────────────────────────────────────────────────────────────────────
# from inora_obstacle import INORAObstacle
# from inora_currency import INORACurrency   ← décommenter quand prêt
#
# Dans __init__ :
#   currency = INORACurrency(lang=lang)
#
# Dans la boucle while, après le bloc OCR :
#   result = currency.process(frame)
#   if result:
#       tts.say(result["tts_message"], priority="high")
#       annotated_frame = result["frame"]   # affiche les boîtes de détection
#
# Obstacle :
#   direction, dist = obstacle.process(frame)
#   if direction:
#       tts.say_urgent(INORAMessages.get(f"obstacle_{direction}", lang, dist=dist))


if __name__ == "__main__":
    main()