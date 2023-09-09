import os
import cv2
from utils import mkdir

down_sample_frequency = 50


def sample_images(video_path, save_dir, frequency=10):
    cap = cv2.VideoCapture(video_path)

    print(f"{video_path} begin")
    base_name = os.path.basename(video_path)
    count = 0
    while True:
        ret_val, frame = cap.read()
        if ret_val and count < 500:
            count += 1
            if count % frequency == down_sample_frequency // 2:
                cv2.imencode('.jpg', frame)[1].tofile(os.path.join(save_dir, base_name + "__" + str(count) + ".jpg"))
                # cv2.imwrite(os.path.join(save_dir, base_name + "__" + str(count) + ".jpg"), frame)
        else:
            cap.release()
            print(f"{video_path} end")
            break


if __name__ == "__main__":
    video_dir = r"F:\terrain_test"
    save_dir = r"F:\terrain_test_result\src_images"

    for video_path in os.listdir(video_dir):
        sample_images(os.path.join(video_dir, video_path), save_dir, frequency=down_sample_frequency)
