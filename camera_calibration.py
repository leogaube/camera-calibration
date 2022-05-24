import numpy as np
import cv2 as cv

import os


def extract_frames(source_file, output_dir, num_frames, num_candidate_frames=50):
    frames = []

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # use saved image frames if they exist (need at least 3 - more is advantagious)
    if len(os.listdir(output_dir)) >= 3:
        for filename in os.listdir(output_dir):
            frames.append(cv.imread(os.path.join(output_dir, filename)))
        return frames

    video_capture = cv.VideoCapture(source_file)

    # frame_count = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))
    # not working for .h264 video --> set frame_count manually
    frame_count = 4900
    frame_offset = int(frame_count / num_frames)

    current_frame = 0

    for i in range(num_frames):
        # video_capture.set(cv.CAP_PROP_POS_FRAMES, frame_offset)
        # not working for .h264 video (unexpected behaviour) --> seek frame manually by grabing consecutive frames
        if i != 0:
            for _ in range(frame_offset - num_candidate_frames):
                video_capture.grab()
                current_frame += 1

        highest_focus = -1
        least_blurry_frame = None
        for _ in range(num_candidate_frames):
            success, candidate_frame = video_capture.read()
            current_frame += 1

            gray_frame = cv.cvtColor(candidate_frame, cv.COLOR_BGR2GRAY)
            focus_score = cv.Laplacian(gray_frame, cv.CV_64F).var()
            if focus_score > highest_focus:
                success, corners = cv.findChessboardCorners(
                    gray_frame, PATTERN_SIZE, CB_FLAGS
                )
                if success:
                    highest_focus = focus_score
                    least_blurry_frame = candidate_frame

        if least_blurry_frame is not None:
            frames.append(least_blurry_frame)
            cv.imwrite(os.path.join(output_dir, f"frame_{i}.jpg"), least_blurry_frame)

    return frames


def detectChessboardPattern(frames):
    # prepare chessboard corner points, like (0,0,0), (1,0,0), (2,0,0), ..., (4,4,0)
    corner_points = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    corner_points[:, :2] = np.mgrid[0 : PATTERN_SIZE[0], 0 : PATTERN_SIZE[1]].T.reshape(
        -1, 2
    )

    # print(corner_points)

    real_world_points_3d = []
    img_points_2d = []

    for frame in frames:
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # gray_frame = cv.resize(gray_frame, (1920//2, 1080//2))

        success, corners = cv.findChessboardCorners(gray_frame, PATTERN_SIZE, CB_FLAGS)
        if success:
            subpixel_corners = cv.cornerSubPix(
                gray_frame, corners, (11, 11), (-1, -1), SP_CRITERIA
            )

            real_world_points_3d.append(corner_points)
            img_points_2d.append(subpixel_corners)

            if DEBUG:
                cv.drawChessboardCorners(
                    gray_frame, PATTERN_SIZE, subpixel_corners, success
                )

                cv.imshow("cb", gray_frame)
                cv.waitKey()
                cv.destroyAllWindows()

    return real_world_points_3d, img_points_2d


def undistort(frames, result_dir):
    real_world_points_3d, img_points_2d = detectChessboardPattern(frames)

    # calibrate
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        real_world_points_3d, img_points_2d, (1920, 1080), None, None
    )

    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # undistort
        result = cv.undistort(frame, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        cropped_result = result[y : y + h, x : x + w]

        cv.imwrite(os.path.join(result_dir, f"frame{i}.jpg"), cropped_result)


if __name__ == "__main__":
    DEBUG = False
    PATTERN_SIZE = (5, 5)
    CB_FLAGS = (
        cv.CALIB_CB_NORMALIZE_IMAGE
        + cv.CALIB_CB_ADAPTIVE_THRESH
        + cv.CALIB_CB_FAST_CHECK
    )
    SP_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    source_file = os.path.join(".", "src", "video.h264")
    output_dir = os.path.join(".", "src", "frames")
    result_dir = os.path.join(".", "results")

    frames = extract_frames(
        source_file, output_dir, num_frames=10, num_candidate_frames=100
    )
    undistort(frames, result_dir)
