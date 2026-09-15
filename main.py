from utils import read_video, save_video
from trackers import Tracker
import numpy as np

def main():
    # Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize tracker and extract object tracks.
    tracker = Tracker('yolov8n.pt')

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/tracks_stubss.pkl',
    )

    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    save_video(output_video_frames, 'output_videos/output_video.mp4')

if __name__ == "__main__":
    main()