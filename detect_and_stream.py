import asyncio
import websockets
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import numpy as np
import tensorflow as tf

SAMPLE_RATE = 16000
DURATION = 2
SAMPLE_LENGTH = SAMPLE_RATE * DURATION
THRESHOLD = 0.57
KEYWORD = "kilo bravo"  # change to your actual keyword

interpreter = tf.lite.Interpreter(model_path="model_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
is_quantized = input_details[0]['dtype'] == np.int8

def predict(audio):
    if len(audio) > SAMPLE_LENGTH:
        audio = audio[:SAMPLE_LENGTH]
    else:
        audio = np.pad(audio, (0, SAMPLE_LENGTH - len(audio)))
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13).T
    mfcc = np.expand_dims(np.expand_dims(mfcc, 0), -1)
    if is_quantized:
        scale, zp = input_details[0]['quantization']
        mfcc_in = (mfcc / scale + zp).astype(np.int8)
    else:
        mfcc_in = mfcc.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], mfcc_in)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    if is_quantized:
        oscale, ozp = output_details[0]['quantization']
        output = (output.astype(np.float32) - ozp) * oscale
    return float(output[0][1])

async def send_detection():
    async with websockets.connect("ws://localhost:8765") as ws:
        await ws.send(KEYWORD)
        print(f"Sent '{KEYWORD}' to server.")

def record():
    print("Recording... speak now")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

if __name__ == "__main__":
    audio = record()
    score = predict(audio)
    print(f"Score: {score:.4f}")
    if score > THRESHOLD:
        print("✅ DETECTED — streaming to server")
        asyncio.run(send_detection())
    else:
        print("❌ Not detected — nothing sent")