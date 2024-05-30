import random

from core.utils import get_video_list
from data.video.video_manager import VideoManager
from mono_task.model.model_manger import ModelManager

from mono_task.model.load_model import init_terrain_seg_model
from mono_task.model.terrain_predictor import terrain_class_names
from mono_task.model.state_machine import TerrainStateMachine

if __name__ == "__main__":
    video_path_dir = r"D:\文档\全地形分割\吉利\demo_video"
    video_list = get_video_list(video_path_dir)
    random.shuffle(video_list)
    show_size = (1280, 720)
    down_sample = 1
    terrain_seg_predictor = init_terrain_seg_model()

    terrain_state_machine = TerrainStateMachine(
        class_names=terrain_class_names[1:],
        driving_area_mask=None,
        mask_valid_threshold=0.5

    )
    model_manager = ModelManager(
        terrain_seg_model=terrain_seg_predictor,
        terrain_state_machine=terrain_state_machine
    )
    if_save = False
    save_path = {
        "terrain_seg_images": r"D:\文档\全地形分割\吉利\visual_video"}
    # if if_save:
    #     mkdir(save_path)

    for video_path in video_list:
        video_manager = VideoManager(
            video_path=video_path,
            show_size=show_size,
            down_sample=down_sample,
            save_path_dict=save_path,
            show_type="video"

        )
        video_manager.infer_video(
            model_manager=model_manager,
            if_show_mot=False,
            if_show_det=False,
            if_show_src=False,
            if_show_terrain=True,
        )
