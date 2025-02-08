import librosa
import numpy as np
import os

def extract_mfcc(audio_path, output_path):
    """Extracts MFCC features from an audio file and saves them as a NumPy file."""
    y, sr = librosa.load(audio_path, sr=44100)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    np.save(output_path, mfccs.T)  # Save as (time, features) numpy array

def process_audio_files(audio_folder, mfcc_folder):
    """Processes all audio files in a folder and extracts MFCCs."""
    os.makedirs(mfcc_folder, exist_ok=True)

    for audio in os.listdir(audio_folder):
        if audio.endswith(".wav"):  # Ensure we process only WAV files
            audio_path = os.path.join(audio_folder, audio)
            mfcc_path = os.path.join(mfcc_folder, os.path.splitext(audio)[0] + ".npy")

            print(f"Extracting MFCC from: {audio_path} → {mfcc_path}")
            extract_mfcc(audio_path, mfcc_path)

if __name__ == "__main__":
    # Process Stuttering Audio
    process_audio_files("data/audio/stuttering", "data/features/mfcc/stuttering")

    # Process Lisp Audio
    process_audio_files("data/audio/lisp", "data/features/mfcc/lisp")

    print("MFCC extraction complete for both stuttering and lisp audio files.")