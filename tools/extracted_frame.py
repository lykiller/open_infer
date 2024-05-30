import os
import cv2

from core.utils import imwrite, mkdir

if __name__ == "__main__":
    src_avm_videos_dir = r"D:\文档\自动泊车\avm_videos"
    dst_image_dir = r"D:\文档\自动泊车\src_images"
    src_avm_videos_list = os.listdir(src_avm_videos_dir)
    src_avm_videos_list.sort()

    for avm_name in src_avm_videos_list:
        if "avm__" not in avm_name:
            temp_dst_image_dir = os.path.join(dst_image_dir, avm_name[:-4])
            mkdir(temp_dst_image_dir)

            cap = cv2.VideoCapture(os.path.join(src_avm_videos_dir, avm_name))
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
            fps = cap.get(cv2.CAP_PROP_FPS)

            print(f"{avm_name} begin")
            count = 0
            while True:
                ret_val, frame = cap.read()
                count += 1
                if ret_val:
                    if count % 5 == 0:
                        image_idx = 1000000 + count
                        image_name = avm_name[:-4] + "__frame__" + str(image_idx)[1:] + ".jpg"
                        imwrite(os.path.join(temp_dst_image_dir, image_name), frame)
                else:
                    cap.release()
                    break
