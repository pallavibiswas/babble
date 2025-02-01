import ffmpeg
import os

def extract_audio(video_path, output_audio_path):
    """Extracts audio from a video file and saves it as a WAV file."""
    ffmpeg.input(video_path).output(output_audio_path, format='wav').run(overwrite_output=True)
    return output_audio_path

def process_videos(videos_folder, audio_folder):
    """Processes all videos in a folder and extracts their audio."""
    os.makedirs(audio_folder, exist_ok=True)

    for video in os.listdir(videos_folder):
        video_path = os.path.join(videos_folder, video)
        audio_path = os.path.join(audio_folder, os.path.splitext(video)[0] + ".wav")
        
        print(f"Extracting audio from: {video_path} → {audio_path}")
        extract_audio(video_path, audio_path)

if __name__ == "__main__":
    # Process Stuttering Videos
    process_videos("data/videos/stuttering", "data/audio/stuttering")

    # Process Lisp Videos
    process_videos("data/videos/lisp", "data/audio/lisp")

    print("Audio extraction complete for both stuttering and lisp videos.")