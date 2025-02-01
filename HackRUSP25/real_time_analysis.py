import os
import cv2
import numpy as np
import librosa
import pyaudio
import wave
import torch
import multiprocessing
from transformers import pipeline
from scripts.train_model import model  # Import your trained LSTM model

# ✅ Fix multiprocessing issues on macOS
multiprocessing.set_start_method("fork", force=True)

# ✅ Prevent multiprocessing-related memory leaks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"

# ✅ Set up PyTorch to use CPU instead of GPU (Fix for macOS crashes)
device = "cpu"

# ✅ Load a smaller and more optimized Hugging Face model
llm = pipeline(
    "text-generation",
    model="tiiuae/falcon-7b-instruct",  # Faster, lightweight model
    torch_dtype=torch.float16,  # Reduce memory usage
    device=device  # Ensure it runs on CPU
)

# 🎙️ Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5
AUDIO_OUTPUT = "real_time_audio.wav"

# 🎥 Initialize OpenCV for lip tracking
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 🎙️ Initialize PyAudio for audio recording
audio = pyaudio.PyAudio()

def record_audio():
    """
    Record audio for a fixed duration.
    """
    print("🎙️ Recording Audio...")
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Audio Recording Complete.")
    stream.stop_stream()
    stream.close()

    # Save the recorded audio
    wf = wave.open(AUDIO_OUTPUT, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

def extract_mfcc(audio_path):
    """
    Extract MFCC features from the recorded audio.
    """
    audio, sample_rate = librosa.load(audio_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs = np.mean(mfccs, axis=1)  # Take mean across time
    return mfccs.reshape(1, 1, -1)  # Reshape for LSTM model

def extract_lip_motion():
    """
    Detects lip motion from video feed.
    """
    motion_features = []
    for _ in range(50):  # Capture 50 frames
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            mouth = frame[y + int(h * 0.6): y + h, x: x + w]  # Approximate mouth region
            mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
            motion_features.append(np.mean(mouth_gray))  # Use mean pixel value as feature

        cv2.imshow("Lip Motion", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    return np.array(motion_features).reshape(1, -1)

def predict_speech_dysfunction():
    """
    Predicts whether the person has a stutter or lisp based on real-time analysis.
    """
    record_audio()
    mfcc_features = extract_mfcc(AUDIO_OUTPUT)
    lip_motion_features = extract_lip_motion()

    # Ensure both features are aligned
    min_length = min(mfcc_features.shape[2], lip_motion_features.shape[1])
    mfcc_features = mfcc_features[:, :, :min_length]
    lip_motion_features = lip_motion_features[:, :min_length]

    # Stack features
    combined_features = np.hstack((lip_motion_features.T, mfcc_features.reshape(-1, 13)))
    combined_features = combined_features.reshape(1, 1, -1)  # Reshape for LSTM

    # Make prediction using LSTM model
    prediction = model.predict(combined_features)
    label = "Stutter" if prediction < 0.5 else "Lisp"
    print(f"Detected Condition: {label}")
    
    return label

def generate_lesson_plan(condition):
    """
    Generates a personalized lesson plan using Mistral.
    """
    prompt = f"Create a structured lesson plan for a person with {condition}. Include daily exercises, speech techniques, and improvement strategies."
    response = llm(prompt, max_length=500, do_sample=True, temperature=0.7)
    return response[0]["generated_text"]

def main():
    print("Starting Real-Time Speech Dysfunction Analysis...")
    detected_condition = predict_speech_dysfunction()
    lesson_plan = generate_lesson_plan(detected_condition)
    print(f"Lesson Plan for {detected_condition}:\n{lesson_plan}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    audio.terminate()

if __name__ == "__main__":
    main()