import numpy as np
import cv2 as cv

import os


def extract_frames(source_file, output_dir, num_frames, num_candidate_frames=100):
    frames = []
    
    # use saved image frames if they exist
    if os.path.isdir(output_dir) and os.listdir(output_dir) is not None:
        for filename in os.listdir(output_dir):
            frames.append(cv.imread(os.path.join(output_dir, filename)))
        return frames
    else:
        os.mkdir(output_dir)
    
    video_capture = cv.VideoCapture(source_file)
       
    #frame_count = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))
    # not working for .h264 video --> set frame_count manually
    frame_count = 4900
    frame_offset = int(frame_count / num_frames)

    current_frame = 0

    for i in range(num_frames):
        #video_capture.set(cv.CAP_PROP_POS_FRAMES, frame_offset)
        # not working for .h264 video (unexpected behaviour) --> seek frame manually by grabing consecutive frames 
        if i != 0:
            for _ in range(frame_offset-num_candidate_frames):
                video_capture.grab()
                current_frame += 1

        highest_focus = -1
        least_blurry_frame = None
        for _ in range(num_candidate_frames):
            success, candidate_frame = video_capture.read()
            current_frame += 1

            focus_score = cv.Laplacian(candidate_frame, cv.CV_64FC3).var()
            if focus_score > highest_focus:
                highest_focus = focus_score
                least_blurry_frame = candidate_frame

        frames.append(least_blurry_frame)
        cv.imwrite(os.path.join(output_dir, f"frame_{i}.jpg"), least_blurry_frame)

    return frames

def estimate_intrinsic_parameters():
    pass


def calibrate(frames):
    print(len(frames))
    estimate_intrinsic_parameters()


if __name__ == "__main__":
    DEBUG = True

    source_file = os.path.join(".", "src", "video.h264")
    output_dir = os.path.join(".", "src", "frames")

    frames = extract_frames(source_file, output_dir, num_frames=10, num_candidate_frames=100)
    calibrate(frames)
