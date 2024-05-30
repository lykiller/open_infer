import os

from core.utils import get_video_list
from data.video.video_manager import VideoManager
from mono_task.model.model_manger import ModelManager

from mono_task.model.load_model import init_terrain_seg_model

if __name__ == "__main__":
    video_path_dir = r"\\10.128.3.137\算法\吉利摄像头图像评估\0301下午路测(阴天YUV亮度48)"
    video_list = get_video_list(video_path_dir)
    show_size = (1280, 720)
    down_sample = 10
    model_manager = ModelManager(
        terrain_seg_model=init_terrain_seg_model(),
    )
    if_save = False
    save_path = {
        "src_images": r"G:\info\武汉雪天自测\src_images",
        "terrain_seg_images": r"G:\info\武汉雪天自测\visual_masks_V0.5.2"
    }
    if if_save:
        for value in save_path.values():
            os.makedirs(value, exist_ok=True)

    for video_path in video_list:
        video_manager = VideoManager(
            video_path=video_path,
            show_size=show_size,
            down_sample=down_sample,
            save_path_dict=save_path,
            show_type="cv"

        )
        video_manager.infer_video(
            model_manager=model_manager,
            if_show_mot=False,
            if_show_det=False,
            if_show_src=True,
            if_show_terrain=True,
        )
