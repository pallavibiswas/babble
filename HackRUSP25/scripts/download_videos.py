import yt_dlp
import os

def read_links_from_file(file_path):
    with open(file_path, "r") as file:
        return [line.strip() for line in file.readlines()]

def download_videos(video_links, output_folder):
    """Download multiple YouTube videos."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_paths = []
    for url in video_links:
        ydl_opts = {'format': 'bestvideo+bestaudio/best', 'outtmpl': f'{output_folder}/%(id)s.%(ext)s'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_paths.append(f"{output_folder}/{info['id']}.{info['ext']}")
    
    return video_paths

if __name__ == "__main__":
    stuttering_links = read_links_from_file("stuttering_videos.txt")
    lisp_links = read_links_from_file("lisp_videos.txt")
    
    stuttering_videos = download_videos(stuttering_links, "data/videos/stuttering")
    lisp_videos = download_videos(lisp_links, "data/videos/lisp")