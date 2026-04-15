import logging
import time
import sounddevice as sd
from inora_voice import INORAVoice

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")

# Show available microphones
print("=== Available audio devices ===")
for i, dev in enumerate(sd.query_devices()):
    if dev["max_input_channels"] > 0:
        name = dev["name"]
        ch   = dev["max_input_channels"]
        print(f"  [{i}] {name} ({ch} in)")
print()

def on_command(action, phrase):
    print(f"ACTION : {action!r}")
    print(f"  PHRASE : {phrase!r}")

print("=== Test INORA Voice (Vosk) ===")
print("Language: FR")
print("Commands to try: inora / lis / stop / repete / anglais / quitter")
print("Press Ctrl+C to exit")

voice = INORAVoice(lang="en", command_handler=on_command)

if not voice.ready:
    print("ERROR: Vosk model not loaded. Check that vosk-model-fr/ exists.")
    exit(1)

voice.start()
print("Microphone active — speak now...")

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass

voice.stop()
print("=== Done ===")
