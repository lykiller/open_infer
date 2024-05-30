import random
from core.utils import get_video_list, mkdir
from data.video.video_manager import VideoManager
from mono_task.model.model_manger import ModelManager

from mono_task.model.load_model import init_det_model
from mono_task.model.load_model import init_tsr_model

if __name__ == "__main__":
    video_path_dir = r"\\10.128.3.137\算法\原始视频\高速场景"
    video_list = get_video_list(video_path_dir)
    random.shuffle(video_list)
    show_size = (1280, 720)
    down_sample = 1
    det_predictor = init_det_model()
    tsr_predictor = init_tsr_model()
    model_manager = ModelManager(
        det_model=det_predictor,
        tsr_model=tsr_predictor,

    )
    if_save = False
    save_path = {
        "det_bboxes": r"D:\dataset\高速\visual_det_videos"}
    if if_save:
        mkdir(save_path)

    for video_path in video_list:
        video_manager = VideoManager(
            video_path=video_path,
            show_size=show_size,
            down_sample=down_sample,
            save_path_dict=save_path,

        )
        video_manager.infer_video(
            model_manager=model_manager,
            if_show_mot=False,
            if_show_det=True,
            if_show_src=False,
            if_show_terrain=False,
        )
