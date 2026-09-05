import librosa
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Configuration
SAMPLE_RATE = 16000
DURATION = 2.0  # seconds
N_MFCC = 13      # Industry standard for speech

def extract_mfcc(file_path):
    """Load audio and extract MFCC features"""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # Ensure exactly 2 seconds
    target_length = int(SAMPLE_RATE * DURATION)
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, target_length - len(audio)))

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    return mfcc.T  # (time_steps, n_mfcc)

def load_dataset(positive_dir, negative_dirs):
    """Load and prepare dataset. negative_dirs can be a list of folders."""
    X = []
    y = []

    # Load positive samples (label = 1)
    for filename in os.listdir(positive_dir):
        if filename.endswith('.wav'):
            mfcc = extract_mfcc(os.path.join(positive_dir, filename))
            X.append(mfcc)
            y.append(1)
            print(f"Loaded positive: {filename}")

    # Load negative samples (label = 0) from one or more folders
    for neg_dir in negative_dirs:
        for filename in os.listdir(neg_dir):
            if filename.endswith('.wav'):
                mfcc = extract_mfcc(os.path.join(neg_dir, filename))
                X.append(mfcc)
                y.append(0)
                print(f"Loaded negative ({neg_dir}): {filename}")

    return np.array(X), np.array(y)

# --- Sanity check counts before doing anything expensive ---
print("Loading dataset...")
X, y = load_dataset('posshi', ['neg_similar_sounidng', 'negshi_bg_noise'])
print(f"\nTotal samples: {len(y)} | positives: {int(np.sum(y == 1))} | negatives: {int(np.sum(y == 0))}\n")

# Stratified train/val split so both sets have a real mix of classes
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Expand dims for CNN input: (batch, time, mfcc, channels)
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

# Build DS-CNN Model
model = models.Sequential([
    layers.Input(shape=X_train.shape[1:]),

    # Block 1
    layers.DepthwiseConv2D((3, 3), activation='relu', padding='same'),
    layers.Conv2D(8, (1, 1), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),  # pools time and frequency
    layers.Dropout(0.25),

    # Block 2 — only pool time, preserve frequency resolution
    layers.DepthwiseConv2D((3, 3), activation='relu', padding='same'),
    layers.Conv2D(16, (1, 1), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 1)),
    layers.Dropout(0.25),

    # Output
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train using the properly split validation data
print("\nTraining model...")
history = model.fit(X_train, y_train,
                     validation_data=(X_val, y_val),
                     epochs=30,
                     batch_size=8)

# Save as Keras model
model.save('kws_model.h5')
print("Model saved as kws_model.h5")

# Convert to TFLite (FP32)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model_fp32 = converter.convert()
with open('model_fp32.tflite', 'wb') as f:
    f.write(tflite_model_fp32)
print(f"FP32 model size: {len(tflite_model_fp32)/1024:.2f} KB")

# Convert to TFLite (INT8 Quantized)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset():
    for i in range(len(X_train)):
        yield [X_train[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_int8 = converter.convert()
with open('model_int8.tflite', 'wb') as f:
    f.write(tflite_model_int8)
print(f"INT8 model size: {len(tflite_model_int8)/1024:.2f} KB")

print("\n✅ Training complete! Check model_fp32.tflite and model_int8.tflite")