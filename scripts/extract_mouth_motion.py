import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def get_lip_distance(landmarks):
    """Calculate vertical distance between upper and lower lip landmarks."""
    upper_lip = np.array([landmarks[13].x, landmarks[13].y])
    lower_lip = np.array([landmarks[14].x, landmarks[14].y])
    return np.linalg.norm(upper_lip - lower_lip)

def extract_mouth_motion(video_path, output_path, visualize=True):
    """Extracts lip motion data from a video and saves it as a .npy file. Optionally visualizes the landmarks."""
    cap = cv2.VideoCapture(video_path)
    mouth_motion_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lip_distance = get_lip_distance(face_landmarks.landmark)
                mouth_motion_data.append(lip_distance)

                if visualize:
                    for idx in [13, 14]:  # Upper and lower lip landmarks
                        x, y = int(face_landmarks.landmark[idx].x * frame.shape[1]), int(face_landmarks.landmark[idx].y * frame.shape[0])
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Draw green circles on lips
                    
                    cv2.putText(frame, f"Lip Distance: {lip_distance:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                
        if visualize:
            cv2.imshow("Mouth Motion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop visualization
                break

    cap.release()
    if visualize:
        cv2.destroyAllWindows()

    np.save(output_path, mouth_motion_data)  # Save as numpy file
    print(f"Saved mouth motion data: {output_path}")

def process_videos(videos_folder, output_folder, visualize=True):
    """Processes all videos in a folder and extracts mouth motion data with optional visualization."""
    os.makedirs(output_folder, exist_ok=True)

    for video in os.listdir(videos_folder):
        if video.endswith(".mp4"):  # Ensure only video files are processed
            video_path = os.path.join(videos_folder, video)
            output_path = os.path.join(output_folder, os.path.splitext(video)[0] + ".npy")

            print(f"Processing video: {video_path}")
            extract_mouth_motion(video_path, output_path, visualize=visualize)

if __name__ == "__main__":
    # Process Stuttering Videos with Visualization
    process_videos("data/videos/stuttering", "data/features/mouth_motion/stuttering", visualize=True)

    # Process Lisp Videos with Visualization
    process_videos("data/videos/lisp", "data/features/mouth_motion/lisp", visualize=True)

    print("Mouth motion extraction complete for both stuttering and lisp videos.")