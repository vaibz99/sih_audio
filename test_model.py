import librosa
import numpy as np
import tensorflow as tf
import os

# Load the INT8 model
print("Loading INT8 model...")
interpreter = tf.lite.Interpreter(model_path="model_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

is_quantized = input_details[0]['dtype'] == np.int8

# 2.0s * 16000 SAMPLE_RATE — must match your training DURATION
SAMPLE_LENGTH = 32000  

# THRESHOLD: Raised to 0.57 to eliminate false positives clustering at 0.50-0.57
THRESHOLD = 0.57

def predict_keyword(file_path):
    """Loads audio, extracts MFCC, and runs TFLite inference"""
    try:
        audio, sr = librosa.load(file_path, sr=16000, mono=True)

        if len(audio) > SAMPLE_LENGTH:
            audio = audio[:SAMPLE_LENGTH]
        else:
            audio = np.pad(audio, (0, SAMPLE_LENGTH - len(audio)))

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T
        mfcc = np.expand_dims(mfcc, axis=0)
        mfcc = np.expand_dims(mfcc, axis=-1)

        if is_quantized:
            input_scale, input_zero_point = input_details[0]['quantization']
            mfcc_int8 = mfcc / input_scale + input_zero_point
            mfcc_int8 = mfcc_int8.astype(np.int8)
            interpreter.set_tensor(input_details[0]['index'], mfcc_int8)
        else:
            interpreter.set_tensor(input_details[0]['index'], mfcc.astype(np.float32))

        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])

        if is_quantized:
            output_scale, output_zero_point = output_details[0]['quantization']
            output_float = (output.astype(np.float32) - output_zero_point) * output_scale
        else:
            output_float = output

        confidence = float(output_float[0][1])
        return confidence
    except Exception as e:
        return f"ERROR: {str(e)}"

# Test ALL files in your folders
print("\n" + "="*70)
print("TESTING POSITIVE SAMPLES (posshi folder)")
print("="*70)

pos_folder = "posshi"
pos_scores = []

for filename in os.listdir(pos_folder):
    if filename.endswith('.wav'):
        file_path = os.path.join(pos_folder, filename)
        score = predict_keyword(file_path)
        if isinstance(score, float):
            pos_scores.append(score)
            status = "✅ DETECTED" if score > THRESHOLD else "❌ MISSED"
            print(f"{status} | {filename[:45]:45} | Score: {score:.4f}")
        else:
            print(f"❌ ERROR | {filename[:45]:45} | {score}")

print("\n" + "="*70)
print("TESTING NEGATIVE SAMPLES (Similar Sounding)")
print("="*70)

neg_similar_folder = "neg_similar_sounidng"  # matches your folder spelling
neg_similar_scores = []

for filename in os.listdir(neg_similar_folder):
    if filename.endswith('.wav'):
        file_path = os.path.join(neg_similar_folder, filename)
        score = predict_keyword(file_path)
        if isinstance(score, float):
            neg_similar_scores.append(score)
            status = "✅ REJECTED" if score < THRESHOLD else "❌ FALSE POSITIVE"
            print(f"{status} | {filename[:45]:45} | Score: {score:.4f}")
        else:
            print(f"❌ ERROR | {filename[:45]:45} | {score}")

print("\n" + "="*70)
print("TESTING NEGATIVE SAMPLES (Background Noise)")
print("="*70)

neg_noise_folder = "negshi_bg_noise"
neg_noise_scores = []

for filename in os.listdir(neg_noise_folder):
    if filename.endswith('.wav'):
        file_path = os.path.join(neg_noise_folder, filename)
        score = predict_keyword(file_path)
        if isinstance(score, float):
            neg_noise_scores.append(score)
            status = "✅ REJECTED" if score < THRESHOLD else "❌ FALSE POSITIVE"
            print(f"{status} | {filename[:45]:45} | Score: {score:.4f}")
        else:
            print(f"❌ ERROR | {filename[:45]:45} | {score}")

# Summary
print("\n" + "="*70)
print("SUMMARY (Threshold set to 0.60)")
print("="*70)

if pos_scores:
    avg_pos = np.mean(pos_scores)
    detected = sum(1 for s in pos_scores if s > THRESHOLD)
    print(f"\nPositive Samples:")
    print(f"  Total: {len(pos_scores)}")
    print(f"  Detected (>{THRESHOLD}): {detected} ({detected/len(pos_scores)*100:.1f}%)")
    print(f"  Avg Confidence: {avg_pos:.4f}")

if neg_similar_scores:
    avg_neg_sim = np.mean(neg_similar_scores)
    false_pos = sum(1 for s in neg_similar_scores if s > THRESHOLD)
    print(f"\nNegative Samples (Similar Sounding):")
    print(f"  Total: {len(neg_similar_scores)}")
    print(f"  False Positives: {false_pos} ({false_pos/len(neg_similar_scores)*100:.1f}%)")
    print(f"  Avg Confidence: {avg_neg_sim:.4f}")

if neg_noise_scores:
    avg_neg_noise = np.mean(neg_noise_scores)
    false_pos_noise = sum(1 for s in neg_noise_scores if s > THRESHOLD)
    print(f"\nNegative Samples (Background Noise):")
    print(f"  Total: {len(neg_noise_scores)}")
    print(f"  False Positives: {false_pos_noise} ({false_pos_noise/len(neg_noise_scores)*100:.1f}%)")
    print(f"  Avg Confidence: {avg_neg_noise:.4f}")

print("\n" + "="*70)