import librosa
import soundfile as sf
import os

SAMPLE_RATE = 16000
CHUNK_LENGTH = 2.0  # seconds

def chunk_long_file(file_path, out_dir, chunk_length=CHUNK_LENGTH, min_chunk_length=1.0):
    """
    Split one long audio file into consecutive chunks of `chunk_length` seconds.
    Drops a trailing chunk if it's shorter than `min_chunk_length` (too short to be useful).
    Saves chunks as <original_name>_chunk1.wav, _chunk2.wav, etc. in out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    chunk_samples = int(chunk_length * sr)
    min_samples = int(min_chunk_length * sr)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    n_chunks = 0
    for i, start in enumerate(range(0, len(audio), chunk_samples)):
        chunk = audio[start:start + chunk_samples]
        if len(chunk) < min_samples:
            continue  # skip leftover scrap at the end
        out_path = os.path.join(out_dir, f"{base_name}_chunk{i+1}.wav")
        sf.write(out_path, chunk, sr)
        n_chunks += 1

    print(f"{file_path}: split into {n_chunks} chunks -> {out_dir}")
    return n_chunks


def trim_file(file_path, out_path, max_length=CHUNK_LENGTH):
    """
    Trim a single file down to its first `max_length` seconds and save.
    Use this if you just want to shorten a clip instead of chunking it.
    """
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    max_samples = int(max_length * sr)
    trimmed = audio[:max_samples]
    sf.write(out_path, trimmed, sr)
    print(f"{file_path}: trimmed to {len(trimmed)/sr:.2f}s -> {out_path}")


if __name__ == "__main__":
    # ---- Example: chunk your 25s outlier file into usable 2s negative clips ----
    chunk_long_file(
        file_path='negshi_bg_noise/neg (20).wav',   # adjust path/name if needed
        out_dir='negshi_bg_noise',                   # drops chunks right back into the same folder
        chunk_length=2.0
    )

    # After running this, delete the original 25s file by hand so it doesn't
    # also get loaded whole (train_model.py will just truncate it to 2s otherwise,
    # wasting the rest of it).