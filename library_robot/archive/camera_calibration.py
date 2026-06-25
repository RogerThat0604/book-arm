import cv2
import numpy as np
import pyrealsense2 as rs

# ===== RealSense 설정 =====
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# ===== 체커보드 설정 =====
# 주의: 칸 수가 아니라 내부 코너 개수입니다.
CHECKERBOARD = (8, 5)

# 한 칸 크기: 25mm
SQUARE_SIZE_MM = 25.0

SAVE_FILE = "realsense_camera_calibration.npz"


def create_realsense():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)

    profile = pipeline.start(config)

    device = profile.get_device()
    print("[REALSENSE] device:", device.get_info(rs.camera_info.name))
    print("[REALSENSE] serial:", device.get_info(rs.camera_info.serial_number))

    align = rs.align(rs.stream.color)

    for _ in range(15):
        pipeline.wait_for_frames()

    print("[REALSENSE] RGB + Depth started")
    return pipeline, align


def preprocess_checkerboard_image(color_image):
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # 대비 강화
    clahe = cv2.createCLAHE(
        clipLimit=4.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    # 샤프닝
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharp = cv2.addWeighted(enhanced, 1.8, blurred, -0.8, 0)

    # 추가 대비/밝기 보정
    corrected = cv2.convertScaleAbs(
        sharp,
        alpha=2.2,
        beta=0
    )

    return corrected


def find_checkerboard(processed_image):
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE

    found, corners = cv2.findChessboardCornersSB(
        processed_image,
        CHECKERBOARD,
        flags=flags
    )

    if found:
        print("[DETECT] checkerboard detected on processed image")
        return True, corners

    print("[DETECT] failed on processed image")
    return False, None


def main():
    pipeline, align = create_realsense()

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0:CHECKERBOARD[0],
        0:CHECKERBOARD[1]
    ].T.reshape(-1, 2)

    objp *= SQUARE_SIZE_MM

    objpoints = []
    imgpoints = []

    image_size = (FRAME_WIDTH, FRAME_HEIGHT)

    print("SPACE: 현재 프레임 저장")
    print("C: 캘리브레이션 실행")
    print("Q: 종료")
    print("현재 화면은 원본이 아니라 전처리된 이미지입니다.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # 전처리 이미지 생성
            processed = preprocess_checkerboard_image(color_image)

            # 체커보드 검출도 전처리 이미지 기준
            found, corners = find_checkerboard(processed)

            # 화면 표시도 전처리 이미지 기준
            display = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

            if found:
                refined_corners = corners

                cv2.drawChessboardCorners(
                    display,
                    CHECKERBOARD,
                    refined_corners,
                    found
                )

                status = "Checkerboard detected"
                color = (0, 255, 0)
            else:
                refined_corners = None
                status = "Checkerboard not detected"
                color = (0, 0, 255)

            cv2.putText(
                display,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.putText(
                display,
                f"Saved: {len(objpoints)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.putText(
                display,
                "View: processed image",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("RealSense Checkerboard Calibration", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if found:
                    objpoints.append(objp.copy())
                    imgpoints.append(refined_corners)
                    print(f"[SAVE] {len(objpoints)} frame saved")
                else:
                    print("[SKIP] 체커보드가 검출되지 않았습니다.")

            elif key == ord("c"):
                if len(objpoints) < 10:
                    print("[WARN] 최소 10장 이상 저장하세요. 권장: 15~30장")
                    continue

                rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints,
                    imgpoints,
                    image_size,
                    None,
                    None
                )

                print("\n===== Calibration Result =====")
                print("RMS error:", rms)
                print("Camera Matrix:")
                print(camera_matrix)
                print("Distortion Coefficients:")
                print(dist_coeffs)

                np.savez(
                    SAVE_FILE,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    rvecs=rvecs,
                    tvecs=tvecs,
                    image_width=FRAME_WIDTH,
                    image_height=FRAME_HEIGHT,
                    checkerboard=CHECKERBOARD,
                    square_size_mm=SQUARE_SIZE_MM
                )

                print(f"[SAVE] {SAVE_FILE} 저장 완료")

            elif key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
