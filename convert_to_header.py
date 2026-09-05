import numpy as np
import librosa

# ---- 1. Convert the model itself ----
with open('model_int8.tflite', 'rb') as f:
    model_bytes = f.read()

with open('model_data.h', 'w') as f:
    f.write('#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n')
    f.write(f'const unsigned int model_data_len = {len(model_bytes)};\n')
    f.write('alignas(8) const unsigned char model_data[] = {\n')
    for i in range(0, len(model_bytes), 12):
        chunk = model_bytes[i:i+12]
        line = ', '.join(f'0x{b:02x}' for b in chunk)
        f.write(f'  {line},\n')
    f.write('};\n\n#endif\n')

print(f"model_data.h written — {len(model_bytes)} bytes embedded")

# ---- 2. Convert one real positive clip's MFCC into a test input array ----
SAMPLE_RATE = 16000
DURATION = 2.0
N_MFCC = 13

TEST_CLIP = 'posshi/Rec1.wav'  # pick a clip you know scores high (0.9+) in test_model.py

audio, sr = librosa.load(TEST_CLIP, sr=SAMPLE_RATE, mono=True)
target_length = int(SAMPLE_RATE * DURATION)
if len(audio) > target_length:
    audio = audio[:target_length]
else:
    audio = np.pad(audio, (0, target_length - len(audio)))

mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC).T  # (time_steps, n_mfcc)

# Quantize to int8 the same way TFLite does: you need your model's actual
# input scale/zero_point. Get these by running this in test_model.py:
#   print(interpreter.get_input_details()[0]['quantization'])
# and paste the two numbers below.
INPUT_SCALE = 4.528687000274658        # <-- REPLACE with real value from get_input_details()
INPUT_ZERO_POINT = 75    # <-- REPLACE with real value from get_input_details()

mfcc_int8 = np.round(mfcc / INPUT_SCALE + INPUT_ZERO_POINT).astype(np.int8)
flat = mfcc_int8.flatten()

with open('test_input.h', 'w') as f:
    f.write('#ifndef TEST_INPUT_H\n#define TEST_INPUT_H\n\n')
    f.write(f'const int test_input_len = {len(flat)};\n')
    f.write('const int8_t test_input[] = {\n')
    for i in range(0, len(flat), 12):
        chunk = flat[i:i+12]
        line = ', '.join(str(int(v)) for v in chunk)
        f.write(f'  {line},\n')
    f.write('};\n\n#endif\n')

print(f"test_input.h written — {len(flat)} int8 values, shape was {mfcc.shape}")
print("\nIMPORTANT: open test_model.py, add this line right after loading the interpreter:")
print("  print(interpreter.get_input_details()[0]['quantization'])")
print("Run it once, get the real (scale, zero_point) numbers, put them in this")
print("script's INPUT_SCALE / INPUT_ZERO_POINT, then rerun this script.")