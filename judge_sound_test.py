import librosa
import numpy as np
import tensorflow as tf
import sys

interpreter = tf.lite.Interpreter(model_path="model_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
is_quantized = input_details[0]['dtype'] == np.int8

SAMPLE_LENGTH = 32000
THRESHOLD = 0.5

def predict_keyword(file_path):
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    if len(audio) > SAMPLE_LENGTH:
        audio = audio[:SAMPLE_LENGTH]
    else:
        audio = np.pad(audio, (0, SAMPLE_LENGTH - len(audio)))
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T
    mfcc = np.expand_dims(mfcc, axis=(0, -1)) if False else np.expand_dims(np.expand_dims(mfcc, 0), -1)

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

if __name__ == "__main__":
    path ="C:/Users/shree/OneDrive/Desktop/soundsz/judge_clip.wav"
    score = predict_keyword(path)
    status = "✅ DETECTED" if score > THRESHOLD else "❌ REJECTED"
    print(f"{status} | {path} | Score: {score:.4f}")