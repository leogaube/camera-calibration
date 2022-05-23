import numpy as np
import cv2 as cv

import os


def extract_frames(filename, num_frames=10):
    video_capture = cv.VideoCapture(filename)
    frame_count = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))

    frames = []

    for frame_offset in range(0, frame_count, int(frame_count / num_frames)):
        video_capture.set(cv.CAP_PROP_POS_FRAMES, frame_offset)

        highest_focus = -1
        least_blurry_frame = None
        for i in range(25):
            ret, frame = video_capture.read()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            focus_score = cv.Laplacian(gray, cv.CV_64F).var()
            if focus_score > highest_focus:
                highest_focus = focus_score
                least_blurry_frame = frame

        frames.append(least_blurry_frame)

        # cv.imshow("frame", frame)
        # cv.waitKey()
        # cv.destroyAllWindows()
    return frames


def estimate_intrinsic_parameters():
    pass


def calibrate(frames):
    print(len(frames))
    estimate_intrinsic_parameters()


if __name__ == "__main__":
    DEBUG = True

    source_file = os.path.join(".", "src", "v.mp4")

    frames = extract_frames(source_file)
    calibrate(frames)
