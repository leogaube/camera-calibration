import numpy as np
import cv2 as cv

import os

np.set_printoptions(suppress=True)


def extract_frames(source_file, output_dir, num_frames, num_candidate_frames=50):
    frames = []

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # use saved image frames if they exist (need at least 3 - more is advantagious)
    if len(os.listdir(output_dir)) >= 3:
        for filename in os.listdir(output_dir):
            frames.append(cv.imread(os.path.join(output_dir, filename)))
        return frames

    print("Extracting frames from calibration video")
    video_capture = cv.VideoCapture(source_file)

    # frame_count = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))
    # not working for .h264 video --> set frame_count manually (2375 for video_new.h264, 4900 for video_old.h264)
    frame_count = 2375 if "_new.h264" in source_file else 4900
    frame_offset = frame_count // (num_frames + 1)

    assert frame_offset >= num_candidate_frames

    current_frame = 0

    for _ in range(num_frames):
        # video_capture.set(cv.CAP_PROP_POS_FRAMES, frame_offset)
        # not working for .h264 video (unexpected behaviour) --> seek frame manually by grabing consecutive frames
        for _ in range(frame_offset - num_candidate_frames):
            video_capture.grab()
            current_frame += 1

        candidate_frames = []
        for _ in range(num_candidate_frames):
            success, frame = video_capture.read()
            current_frame += 1

            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            focus_score = cv.Laplacian(gray_frame, cv.CV_64F).var()

            candidate_frames.append((gray_frame, frame, focus_score))

        # sort candidate frames by focus_score
        candidate_frames.sort(key=lambda tup: tup[2], reverse=True)

        least_blurry_frame = None
        for gray_frame, color_frame, _ in candidate_frames:
            success, corners = cv.findChessboardCorners(
                gray_frame, PATTERN_SIZE, CB_FLAGS
            )
            if success:
                least_blurry_frame = color_frame
                break

        if least_blurry_frame is not None:
            cv.imwrite(
                os.path.join(output_dir, f"frame_{len(frames)}.png"), least_blurry_frame
            )
            frames.append(least_blurry_frame)
            print(
                f"calibration frame for [{current_frame-num_candidate_frames}:{current_frame}] was found"
            )
        else:
            print(
                f"unable to detect checkerboard in any in frames [{current_frame-num_candidate_frames}:{current_frame}]"
            )

    return frames


def detectChessboardPattern(frames):

    # prepare chessboard corner points, like (0,0,0), (1,0,0), (2,0,0), ..., (4,4,0)
    cb_corner_points = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    cb_corner_points[:, :2] = np.mgrid[
        0 : PATTERN_SIZE[0], 0 : PATTERN_SIZE[1]
    ].T.reshape(-1, 2)

    # print(cb_corner_points)

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

            real_world_points_3d.append(cb_corner_points)
            img_points_2d.append(subpixel_corners)

            if DEBUG:
                cv.drawChessboardCorners(frame, PATTERN_SIZE, subpixel_corners, success)

                for obj_point, pos in zip(cb_corner_points, corners):
                    cv.putText(
                        frame,
                        str(obj_point),
                        (int(pos[0][0]) + 10, int(pos[0][1]) - 10),
                        cv.FONT_HERSHEY_DUPLEX,
                        0.85,
                        (255, 0, 255),
                        1,
                    )

                cv.imshow("cb", frame)
                cv.waitKey()
                cv.destroyAllWindows()

                cv.imwrite("frame.png", frame)

    return real_world_points_3d, img_points_2d


def undistort(frames, result_dir):
    real_world_points_3d, img_points_2d = detectChessboardPattern(frames)

    ret, int_mtx, dist_coef, rvecs, tvecs = cv.calibrateCamera(
        real_world_points_3d, img_points_2d, (1920, 1080), None, None
    )

    print("Intrinsic Matrix:")
    print(int_mtx)

    print("\nDistortion Coefficients:")
    print(dist_coef)

    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    for i, frame in enumerate(frames):
        result = cv.undistort(frame, int_mtx, dist_coef, None, None)
        cv.imwrite(os.path.join(result_dir, f"frame{i}.png"), result)

    # calculate the mean reprojection error
    mean_error = 0
    for i in range(len(real_world_points_3d)):
        img_points_reprojected, _ = cv.projectPoints(
            real_world_points_3d[i], rvecs[i], tvecs[i], int_mtx, dist_coef
        )
        error = cv.norm(img_points_2d[i], img_points_reprojected, cv.NORM_L2) / len(
            img_points_reprojected
        )
        mean_error += error
    print(
        "\nMean Reprojection Error: {}".format(mean_error / len(real_world_points_3d))
    )


if __name__ == "__main__":
    DEBUG = False
    PATTERN_SIZE = (5, 5)
    NUM_CALIBRATION_FRAMES = 20

    CB_FLAGS = cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_ADAPTIVE_THRESH
    SP_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    source_file = os.path.join(".", "src", "video_old.h264")
    output_dir = os.path.join(".", "src", "frames")
    result_dir = os.path.join(".", "results")

    frames = extract_frames(
        source_file, output_dir, NUM_CALIBRATION_FRAMES, num_candidate_frames=50
    )
    undistort(frames, result_dir)
