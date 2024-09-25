# Copyright (C) Peizhi Yan. 2024

import cv2

def video_to_images(video_path, original_fps = 60, subsample_fps = 30):
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    # Calculate the subsampling interval
    interval = original_fps / subsample_fps
    
    # Select every nth frame based on the interval
    subsampled_frames = [frames[int(i * interval)] for i in range(int(len(frames) / interval))]

    frame_count = len(subsampled_frames)
    
    cap.release()
    print(f"Conversion completed. {frame_count}")

    return subsampled_frames










