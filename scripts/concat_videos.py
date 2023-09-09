import os

from moviepy.editor import *
from utils import get_video_list

if __name__ == "__main__":
    video_path_list = get_video_list(r"F:\videos_202305\good_infer")
    video_list = []
    for path in video_path_list:
        video_list.append(VideoFileClip(path))
    concat_video = concatenate_videoclips(video_list)

    concat_video.to_videofile(r"F:\show_video.mp4", fps=24, remove_temp=False)

