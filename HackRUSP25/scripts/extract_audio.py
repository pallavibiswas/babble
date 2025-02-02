import ffmpeg
import os

def extract_audio(video_path, output_audio_path):
    """Extracts audio from a video file and saves it as WAV format."""
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_audio_path, format='wav')
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
        print(f"✅ Extracted: {video_path} → {output_audio_path}")
    except ffmpeg.Error as e:
        print(f"Error extracting audio from {video_path}: {e}")

def process_videos(videos_folder, audio_folder):
    """Processes all videos in a folder and extracts their audio."""
    if not os.path.exists(videos_folder):
        print(f"Warning: {videos_folder} does not exist. Skipping...")
        return
    
    os.makedirs(audio_folder, exist_ok=True)

    for video in os.listdir(videos_folder):
        if video.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):  # Ensure only video files are processed
            video_path = os.path.join(videos_folder, video)
            audio_path = os.path.join(audio_folder, os.path.splitext(video)[0] + ".wav")
            
            print(f"Extracting audio from: {video_path} → {audio_path}")
            extract_audio(video_path, audio_path)
        else:
            print(f"Skipping non-video file: {video}")

if __name__ == "__main__":
    # Process Stuttering Videos
    process_videos("data/videos/stuttering", "data/audio/stuttering")

    # Process Lisp Videos
    process_videos("data/videos/lisp", "data/audio/lisp")

    print("Audio extraction complete for both stuttering and lisp videos.")
