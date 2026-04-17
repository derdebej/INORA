"""
INORA - Test Whisper
=====================
Teste la reconnaissance vocale Whisper avant intégration.

Installation :
  pip install openai-whisper

Lancement :
  python test_whisper.py
"""

import sounddevice as sd
import numpy as np
import whisper
import time

SAMPLE_RATE  = 16000
DURATION     = 4       # secondes d'enregistrement
MODEL_SIZE   = "small" # tiny / base / small / medium / large
LANGUAGE     = "fr"    # "fr" ou "en" ou None (détection auto)


DEVICE = 2   # ← change ce numéro si ton micro est différent
             # Lance : python -c "import sounddevice as sd; print(sd.query_devices())"
             # pour voir la liste des périphériques

def record(duration: int) -> np.ndarray:
    """Enregistre le micro pendant `duration` secondes."""
    print(f"  Enregistrement ({duration}s)... Parle maintenant !")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=DEVICE,
    )
    sd.wait()
    return audio.flatten()


def main():
    print("=" * 45)
    print("       Test Whisper — INORA")
    print("=" * 45)

    # Chargement du modèle
    print(f"\nChargement modèle Whisper '{MODEL_SIZE}'...")
    print("(Premier lancement = téléchargement ~250MB, patiente...)\n")
    model = whisper.load_model(MODEL_SIZE)
    print("Modèle prêt.\n")

    # Boucle de test
    while True:
        print("-" * 45)
        input("  Appuie sur ENTREE puis parle...")
        print()

        audio = record(DURATION)

        t0 = time.time()
        result = model.transcribe(audio, language=LANGUAGE, fp16=False)
        elapsed = (time.time() - t0) * 1000

        text = result["text"].strip()
        print(f"\n  Reconnu  : {text!r}")
        print(f"  Temps    : {elapsed:.0f} ms")
        print(f"  Langue   : {result.get('language', '?')}")

        again = input("\n  Autre test ? (o/n) : ").strip().lower()
        if again != "o":
            break

    print("\n=== Fin du test ===")


if __name__ == "__main__":
    main()