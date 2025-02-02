import os
import cv2
import numpy as np
import librosa
import pyaudio
import wave
import torch
import time
import mediapipe as mp
import multiprocessing
from transformers import pipeline
#from moviepy.editor import VideoFileClip
from scripts.train_model import model  # Import trained LSTM model

# Fix multiprocessing issues on macOS
multiprocessing.set_start_method("fork", force=True)

# Prevent multiprocessing-related memory leaks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"

# Set up PyTorch to use CPU instead of GPU (Fix for macOS crashes)
device = "cpu"

# Load a smaller and more optimized Hugging Face model
llm = pipeline(
    "text-generation",
    model="tiiuae/falcon-7b-instruct",
    torch_dtype=torch.float16,
    device=device
)

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512  # Reduced buffer size to prevent overflow
RECORD_SECONDS = 15  # Longer recording time for full paragraph
AUDIO_OUTPUT = "speech_audio.wav"
VIDEO_OUTPUT = "speech_video.mp4"

# Paragraph to be displayed for reading
PARAGRAPH = """ 
The slippery snake swiftly slithered through the thick grass.  
Brave explorers struggled to speak clearly in the brisk morning air.  
She sells sea shells by the shore, whispering words with soft sounds.  
The bright blue butterfly flapped gracefully above the glistening stream.  
Thrilling challenges help strengthen speech skills and boost confidence.  
A crisp autumn breeze brushed past, scattering golden leaves everywhere.
"""

# Initialize MediaPipe Face Mesh for Lip Tracking
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def display_paragraph():
    """Display the paragraph on the screen while recording."""
    frame = np.zeros((500, 800, 3), dtype=np.uint8)
    cv2.putText(frame, "Read the following aloud:", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    y_offset = 100
    for line in PARAGRAPH.split("\n"):
        cv2.putText(frame, line.strip(), (50, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 40

    return frame

def record_video_and_audio():
    """Record video & audio simultaneously and save them."""
    print("Recording video & audio...")

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, 20.0, (640, 480))

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, 
                        rate=RATE, input=True, 
                        frames_per_buffer=CHUNK, 
                        input_device_index=None)  # Auto-select the best mic

    frames = []

    start_time = time.time()
    
    while time.time() - start_time < RECORD_SECONDS:
        ret, frame = cap.read()
        if not ret:
            break

        overlay = display_paragraph()
        frame[:overlay.shape[0], :overlay.shape[1]] = overlay

        out.write(frame)
        cv2.imshow("Reading Task", frame)

        try:
            data = stream.read(CHUNK, exception_on_overflow=False)  # Handle buffer overflow
        except OSError:
            print("Warning: Audio buffer overflow. Skipping this chunk.")
            data = b'\x00' * CHUNK  # Insert silence to avoid crashing

        frames.append(data)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("Recording complete. Saving files...")

    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    wf = wave.open(AUDIO_OUTPUT, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def extract_mfcc():
    """Extract MFCC features from recorded audio."""
    y, sr = librosa.load(AUDIO_OUTPUT, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    if mfccs.shape[1] == 0:
        raise ValueError("Error: Extracted MFCC features are empty.")

    return np.mean(mfccs, axis=1).reshape(1, 1, -1)

def extract_lip_distance():
    """Extract lip distance changes using MediaPipe from recorded video."""
    cap = cv2.VideoCapture(VIDEO_OUTPUT)
    lip_distances = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                top_lip = face_landmarks.landmark[13]  # Upper lip
                bottom_lip = face_landmarks.landmark[14]  # Lower lip
                lip_distance = abs(top_lip.y - bottom_lip.y)
                lip_distances.append(lip_distance)

    cap.release()
    
    if len(lip_distances) == 0:
        print("No lip motion detected. Returning default zero array.")
        return np.zeros((1, 13))

    return np.array(lip_distances).reshape(1, -1)

def predict_speech_dysfunction():
    """Predict speech dysfunction based on MFCC & lip motion features."""
    mfcc_features = extract_mfcc()
    lip_motion_features = extract_lip_distance()

    min_length = min(mfcc_features.shape[2], lip_motion_features.shape[1])
    mfcc_features = mfcc_features[:, :, :min_length]
    lip_motion_features = lip_motion_features[:, :min_length]

    combined_features = np.hstack((lip_motion_features.T, mfcc_features.reshape(-1, 13)))
    combined_features = combined_features.reshape(1, 1, -1)

    prediction = model.predict(combined_features)
    return "Stutter" if prediction < 0.5 else "Lisp"

def main():
    record_video_and_audio()
    condition = predict_speech_dysfunction()
    lesson_plan = llm(f"Generate a speech improvement plan for {condition}.")
    
    print(f"Detected Condition: {condition}")
    print(f"Lesson Plan:\n{lesson_plan[0]['generated_text']}")

if __name__ == "__main__":
    main()