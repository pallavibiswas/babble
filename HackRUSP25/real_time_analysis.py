import cv2
import numpy as np
import librosa
import pyaudio  # type: ignore
import time
import mediapipe as mp
from scripts.train_model import model

# Initialize MediaPipe Face Mesh for Lip Tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512  

# Maximum camera runtime (10 minutes)
MAX_RUNTIME = 10 * 60  

# Paragraph to be displayed for reading
PARAGRAPH = """ 
The slippery snake swiftly slithered through the thick grass.  
Brave explorers struggled to speak clearly in the brisk morning air.  
She sells sea shells by the shore, whispering words with soft sounds.  
The bright blue butterfly flapped gracefully above the glistening stream.  
Thrilling challenges help strengthen speech skills and boost confidence.  
A crisp autumn breeze brushed past, scattering golden leaves everywhere.
"""

def display_paragraph(frame):
    """Display the paragraph on the screen while recording."""
    overlay = np.zeros((500, 800, 3), dtype=np.uint8)
    cv2.putText(overlay, "Read the following aloud:", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    y_offset = 100
    for line in PARAGRAPH.split("\n"):
        cv2.putText(overlay, line.strip(), (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 40

    frame[:overlay.shape[0], :overlay.shape[1]] = overlay
    return frame

def extract_mfcc(audio_data):
    """Extract MFCC features from audio data."""
    try:
        y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        mfccs = librosa.feature.mfcc(y=y, sr=RATE, n_mfcc=13)
        return mfccs.T
    except Exception as e:
        print(f"Error extracting MFCC features: {str(e)}")
        return np.zeros((1, 13))

def extract_lip_distance(frame):
    """Extract lip distance changes using MediaPipe from a video frame."""
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                top_lip = face_landmarks.landmark[13]
                bottom_lip = face_landmarks.landmark[14]
                return abs(top_lip.y - bottom_lip.y)
        return 0.0
    except Exception as e:
        print(f"Error extracting lip features: {str(e)}")
        return 0.0

def predict_speech_dysfunction(mfcc_features, lip_distance):
    """Predict speech dysfunction based on MFCC & lip motion features."""
    try:
        # Combine features for single frame
        combined_features = np.hstack((lip_distance, mfcc_features[0]))  # Shape: (14,)
        
        # Reshape to match model's expected input shape (None, 1, 14)
        model_input = combined_features.reshape(-1, 1, 14)
        
        # Make prediction
        predictions = model.predict(model_input)
        
        return "Stutter" if predictions[0] < 0.5 else "Lisp"
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return "Error: Could not complete analysis"

def generate_lesson_plan(condition):
    """Generate a lesson plan based on the speech dysfunction detected."""
    if condition == "Stutter":
        return """
        Lesson Plan for Stuttering:
        1. **Breathing Exercises:** Practice slow, controlled breathing before speaking.
        2. **Pausing Strategy:** Speak slowly and insert small pauses.
        3. **Word Stretching:** Hold onto difficult words longer than usual.
        4. **Soft Onsets:** Start words gently to reduce abrupt speech blocks.
        5. **Daily Practice:** Read aloud for 5 minutes daily with controlled pace.
        """
    elif condition == "Lisp":
        return """
        Lesson Plan for Lisp:
        1. **Tongue Placement:** Practice placing your tongue behind your upper teeth for "s" sounds.
        2. **Mirror Exercises:** Watch your mouth movements to correct positioning.
        3. **Slow Articulation:** Pronounce difficult words slowly and clearly.
        4. **Repeating Sentences:** Practice tongue twisters focusing on "s" and "z" sounds.
        5. **Daily Exercises:** Say words with "s" sound repeatedly with correct tongue placement.
        """
    else:
        return "No clear condition detected. Try again with a clearer audio sample."

def start_analysis():
    """Runs real-time AI speech analysis."""
    # Initialize audio stream
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, 
                        rate=RATE, input=True, 
                        frames_per_buffer=CHUNK)

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    start_time = time.time()  # Track start time

    try:
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > MAX_RUNTIME:
                print("Session ended: Maximum time reached.")
                break

            # Capture video frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Display paragraph on the frame
            frame = display_paragraph(frame)

            # Extract lip distance
            lip_distance = extract_lip_distance(frame)

            # Capture audio data
            audio_data = stream.read(CHUNK, exception_on_overflow=False)

            # Extract MFCC features
            mfcc_features = extract_mfcc(audio_data)

            # Predict speech dysfunction
            condition = predict_speech_dysfunction(mfcc_features, lip_distance)

            # Display prediction on the frame
            cv2.putText(frame, f"Condition: {condition}", (50, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display "Press Q to quit" message
            cv2.putText(frame, "Press 'Q' to quit", (50, 480),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Show the frame
            cv2.imshow("Reading Task", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Generate and print lesson plan at the end
        lesson_plan = generate_lesson_plan(condition)
        print(lesson_plan)

    except Exception as e:
        print(f"Program error: {str(e)}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        cap.release()
        cv2.destroyAllWindows()

# Prevent script from running automatically when imported
if __name__ == "__main__":
    start_analysis()
