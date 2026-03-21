"""
INORA - Main
=============
Point d'entrée principal du projet.
Orchestre la caméra, le module OCR et le module TTS ensemble.

Lancement :
  python main.py

Quitter : appuie sur  Q  dans la fenêtre caméra
"""

import cv2
import logging
import time
import threading

from inora_ocr import INORAOcr
from inora_tts import INORASpeaker, INORAMessages

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(levelname)s: %(message)s"
)
log = logging.getLogger("INORA_MAIN")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration centrale — modifie ici pour adapter le comportement
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "lang":                 "fr",   # langue OCR + TTS : "en" ou "fr"
    "camera_index":         1,      # 0 = webcam par défaut
    "ocr_confidence":       0.8,    # seuil de confiance OCR (0.0 à 1.0)
    "ocr_repeat_delay":     10.0,   # secondes avant de répéter le même texte
    "tts_repeat_delay":     3.0,    # secondes avant de répéter le même message
    "show_window":          True,   # afficher la fenêtre caméra annotée
    "interrupt_key":        "s",    # touche S → interrompt la lecture
    "toggle_key":           "t",    # touche T → active/désactive le TTS
                                    # Sur Jetson : remplacer par bouton GPIO
}


# ─────────────────────────────────────────────────────────────────────────────
# Boucle principale
# ─────────────────────────────────────────────────────────────────────────────
def listen_for_interrupt(tts: INORASpeaker, ocr: "INORAOcr", stop_event: threading.Event):
    """
    Thread qui écoute la touche d'interruption clavier.
    Sur Jetson : remplacer par détection GPIO.

    Touche S → interrompt la lecture TTS en cours
    """
    import sys
    try:
        import msvcrt   # Windows uniquement
        while not stop_event.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                if key == CONFIG["interrupt_key"]:
                    log.info("Interruption TTS demandée.")
                    tts.interrupt()
                elif key == CONFIG["toggle_key"]:
                    tts.toggle(ocr=ocr)
                    status = "activé" if tts.enabled else "désactivé"
                    log.info(f"TTS {status}.")
            time.sleep(0.05)
    except ImportError:
        # Linux/Jetson — utilise termios
        import tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not stop_event.is_set():
                ch = sys.stdin.read(1).lower()
                if ch == CONFIG["interrupt_key"]:
                    log.info("Interruption TTS demandée.")
                    tts.interrupt()
                elif ch == CONFIG["toggle_key"]:
                    tts.toggle(ocr=ocr)
                    status = "activé" if tts.enabled else "désactivé"
                    log.info(f"TTS {status}.")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    log.info("Démarrage INORA...")

    # ── Initialisation des modules ────────────────────────────────────────────
    lang = CONFIG["lang"]

    tts = INORASpeaker(lang=lang)
    # Annonce de démarrage — forcée même si TTS désactivé par défaut
    tts._enabled = True
    tts.say(INORAMessages.get("system_ready", lang), priority="urgent")
    tts._enabled = False   # remet en attente — l'utilisateur active avec T

    ocr = INORAOcr(
        lang=lang,
        confidence_threshold=CONFIG["ocr_confidence"],
        repeat_delay=CONFIG["ocr_repeat_delay"],
    )

    cap = cv2.VideoCapture(CONFIG["camera_index"])
    if not cap.isOpened():
        log.error("Impossible d'ouvrir la caméra.")
        tts.stop()
        return

    log.info(f"Caméra ouverte. Appuie sur Q pour quitter, {CONFIG['interrupt_key'].upper()} pour interrompre le TTS.")

    # ── Thread d'interruption clavier ────────────────────────────────────────
    stop_event = threading.Event()
    interrupt_thread = threading.Thread(
        target=listen_for_interrupt,
        args=(tts, ocr, stop_event),
        daemon=True,
        name="Interrupt-Listener"
    )
    interrupt_thread.start()

    # ── Boucle caméra ─────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame caméra non reçue, nouvelle tentative...")
            time.sleep(0.1)
            continue

        # ── OCR ───────────────────────────────────────────────────────────────
        text, annotated_frame = ocr.process(frame)

        if text:
            log.info(f"Texte détecté : {text!r}")
            message = INORAMessages.get("ocr_reading", lang, text=text)
            tts.say(message, priority="normal")

        # ── Affichage ─────────────────────────────────────────────────────────
        if CONFIG["show_window"]:
            cv2.imshow("INORA — OCR temps réel  (Q pour quitter)", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("Arrêt demandé par l'utilisateur.")
                break

    # ── Nettoyage ─────────────────────────────────────────────────────────────
    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    tts.stop()
    log.info("INORA arrêté proprement.")


# ─────────────────────────────────────────────────────────────────────────────
# Intégration future — exemple pour tes camarades
# ─────────────────────────────────────────────────────────────────────────────
# Quand les modules de tes camarades seront prêts, ils s'intègrent ici :
#
#   from inora_obstacle import INORAObstacle
#   from inora_currency import INORACurrency
#
#   obstacle = INORAObstacle()
#   currency = INORACurrency()
#
#   # Dans la boucle while :
#   direction, dist = obstacle.process(frame)
#   if direction:
#       tts.say_urgent(INORAMessages.get(f"obstacle_{direction}", lang, dist=dist))
#
#   coin = currency.process(frame)
#   if coin:
#       tts.say(INORAMessages.get("coin_detected", lang, value=coin), priority="high")


if __name__ == "__main__":
    main()