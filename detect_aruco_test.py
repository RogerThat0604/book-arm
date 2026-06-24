import cv2
import numpy as np
import pyrealsense2 as rs

# ===== 설정 =====
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# 실제 출력한 ArUco 마커 한 변 길이
# 5cm면 0.05
MARKER_SIZE_M = 0.025

# ArUco Dictionary
ARUCO_DICT = cv2.aruco.DICT_4X4_50


def create_realsense():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)

    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)

    for _ in range(15):
        pipeline.wait_for_frames()

    print("[INFO] RealSense started")
    return pipeline, align


def get_camera_matrix(intrinsics):
    return np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float64)


def get_marker_center(corners):
    pts = corners.reshape((4, 2))
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))
    return cx, cy


def get_median_depth(depth_frame, cx, cy, radius=5):
    values = []

    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if x < 0 or y < 0 or x >= FRAME_WIDTH or y >= FRAME_HEIGHT:
                continue

            d = depth_frame.get_distance(x, y)

            if 0.05 <= d <= 2.0:
                values.append(d)

    if not values:
        return None

    return float(np.median(values))


def main():
    pipeline, align = create_realsense()

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            camera_matrix = get_camera_matrix(intrinsics)
            dist_coeffs = np.zeros((5, 1), dtype=np.float64)

            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            corners, ids, rejected = detector.detectMarkers(gray)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners,
                    MARKER_SIZE_M,
                    camera_matrix,
                    dist_coeffs
                )

                for i, marker_id in enumerate(ids.flatten()):
                    corner = corners[i]
                    rvec = rvecs[i][0]
                    tvec = tvecs[i][0]

                    cx, cy = get_marker_center(corner)

                    depth_distance_m = get_median_depth(depth_frame, cx, cy)
                    pose_distance_m = float(tvec[2])

                    cv2.drawFrameAxes(
                        color_image,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        MARKER_SIZE_M * 0.5
                    )

                    cv2.circle(color_image, (cx, cy), 5, (0, 255, 255), -1)

                    x_cm = tvec[0] * 100
                    y_cm = tvec[1] * 100
                    z_cm = tvec[2] * 100

                    depth_cm_text = "Depth: N/A"
                    if depth_distance_m is not None:
                        depth_cm_text = f"Depth: {depth_distance_m * 100:.1f} cm"

                    text_x = int(corner[0][0][0])
                    text_y = int(corner[0][0][1]) - 60

                    cv2.putText(
                        color_image,
                        f"ID: {marker_id}",
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        color_image,
                        f"Pose Z: {z_cm:.1f} cm",
                        (text_x, text_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        color_image,
                        depth_cm_text,
                        (text_x, text_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        color_image,
                        f"X:{x_cm:+.1f} Y:{y_cm:+.1f} Z:{z_cm:.1f} cm",
                        (20, 40 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2
                    )

            else:
                cv2.putText(
                    color_image,
                    "No ArUco marker detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("RealSense ArUco Distance Test", color_image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
