from piper import PiperVoice
import sounddevice as sd
import numpy as np
import wave
import io

voice = PiperVoice.load('fr_FR-siwis-medium.onnx')

# synthesize_wav écrit dans un objet WAV
buf = io.BytesIO()
with wave.open(buf, 'wb') as wav:
    voice.synthesize_wav('Bonjour je suis INORA', wav)

# Lecture audio
buf.seek(0)
with wave.open(buf, 'rb') as wav:
    frames = wav.readframes(wav.getnframes())
    rate   = wav.getframerate()

data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
sd.play(data, rate)
sd.wait()
print('OK !')