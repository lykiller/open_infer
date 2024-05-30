import os
from data.image.image_manager import ImageManager
from mono_task.model import ModelManager

from mono_task.model.load_model import init_terrain_seg_model

if __name__ == "__main__":
    image_path_dir = r"\\10.128.3.137\算法\吉利摄像头图像评估\0301下午路测(阴天YUV亮度48)"

    show_size = (1280, 720)
    down_sample = 1
    terrain_seg_predictor = init_terrain_seg_model()
    model_manager = ModelManager(
        terrain_seg_model=terrain_seg_predictor
    )
    if_save = True
    save_path = {
        "terrain_seg_images": r"\\10.128.3.137\算法\吉利摄像头图像评估\全地形测试效果"
    }
    if if_save:
        for value in save_path.values():
            os.makedirs(value, exist_ok=True)
    image_manager = ImageManager(
        image_path=image_path_dir,
        show_size=show_size,
        show_type="image",
        save_path_dict=save_path
    )
    image_manager.infer_images(
        model_manager=model_manager,
        if_show_mot=False,
        if_show_det=False,
        if_show_src=False,
        if_show_terrain=True,
    )
