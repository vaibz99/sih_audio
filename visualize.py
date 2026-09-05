import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ---- CHANGE THIS to the clip you want to look at ----
FILE_PATH = 'posshi/pos5.wav'
SAMPLE_RATE = 16000

audio, sr = librosa.load(FILE_PATH, sr=SAMPLE_RATE, mono=True)

fig, axs = plt.subplots(4, 1, figsize=(10, 12))

# 1. Raw waveform
librosa.display.waveshow(audio, sr=sr, ax=axs[0])
axs[0].set_title('Waveform (raw audio)')

# 2. Spectrogram (STFT magnitude, log scale)
stft = librosa.stft(audio)
spec_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
img1 = librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='hz', ax=axs[1])
axs[1].set_title('Spectrogram (linear frequency)')
fig.colorbar(img1, ax=axs[1], format='%+2.0f dB')

# 3. Mel-spectrogram
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
img2 = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axs[2])
axs[2].set_title('Mel-spectrogram (perceptual frequency scale)')
fig.colorbar(img2, ax=axs[2], format='%+2.0f dB')

# 4. MFCC (what your model actually trains on)
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
img3 = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=axs[3])
axs[3].set_title('MFCC (13 coefficients — what the model sees)')
fig.colorbar(img3, ax=axs[3])

plt.tight_layout()
plt.savefig('clip_visualization.png', dpi=150)
print("Saved clip_visualization.png")
plt.show()


'''
import librosa
import os

FOLDER = 'negshi_bg_noise'  # change to check other folders

files = [f for f in os.listdir(FOLDER) if f.endswith('.wav')]
files.sort()

over_limit = []

for f in files:
    path = os.path.join(FOLDER, f)
    duration = librosa.get_duration(path=path)
    flag = " <-- OVER 1.0s" if duration > 1.0 else ""
    if duration > 1.0:
        over_limit.append(f)
    print(f"{f:40s} {duration:.3f}s{flag}")

print(f"\n{len(files)} clips checked. {len(over_limit)} are over 1.0s.")
'''