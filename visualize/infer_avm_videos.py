from core.utils import get_video_list, mkdir
from data.video.video_manager import VideoManager
from mono_task.model.model_manger import ModelManager

from mono_task.model.load_model import init_apa_model

if __name__ == "__main__":
    video_path_dir = r"\\10.128.3.137\算法\原始视频\AVM\avm_videos"
    video_list = get_video_list(video_path_dir)
    show_size = (1024, 1024)
    down_sample = 1
    apa_predictor = init_apa_model()
    model_manager = ModelManager(
        apa_model=apa_predictor
    )
    if_save = False
    save_path = {
        "parking_slots": r"D:\dataset\高速"}
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
            if_show_det=False,
            if_show_src=False,
            if_show_terrain=False,
        )
