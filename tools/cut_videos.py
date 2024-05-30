import cv2

if __name__ == "__main__":
    video_path = r"D:\自测结果\terrain\visual_videos\2023_08_14_17_46_43_d.mp4"
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    image_np_list = []
    while True:
        ret, frame = video.read()
        if ret and 26*15 < frame_count < 38*15:
            image_np_list.append(frame)
            frame_count += 1
        else:
            break
    video.release()
    dst_path = r"D:\自测结果\terrain\visual_videos\典型铺装.mp4"
    video_writer = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (1280, 720), True)
    for image_np in image_np_list:
        video_writer.write(image_np)


