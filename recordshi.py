import sounddevice as sd
import scipy.io.wavfile as wav
import sys

DURATION = 2
SAMPLE_RATE = 16000

try:
    print("Recording... speak now", flush=True)
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    wav.write("judge_clip.wav", SAMPLE_RATE, audio)
    print("Saved judge_clip.wav", flush=True)
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    sys.exit(1)