# Robot/Raspberry Pi 쪽 실행 코드
# 목적: RealSense RGB(color) 프레임만 UDP로 PC에 전송
# 로봇팔 제어, depth 측정, QR 인식 모두 제외한 순수 영상 송신 테스트

import cv2
import time
import socket
import struct
import numpy as np
import pyrealsense2 as rs


PC_IP = "192.168.0.52"
UDP_VIDEO_PORT = 5000

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
JPEG_QUALITY = 70
CHUNK_SIZE = 1300


def start_realsense_color():
    pipeline = rs.pipeline()
    config = rs.config()

    # 일단 RGB(color)만 사용합니다.
    # depth stream은 켜지 않습니다.
    config.enable_stream(
        rs.stream.color,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        rs.format.bgr8,
        FPS
    )

    profile = pipeline.start(config)

    device = profile.get_device()

    print("[REALSENSE] device:", device.get_info(rs.camera_info.name))
    print("[REALSENSE] serial:", device.get_info(rs.camera_info.serial_number))
    print("[REALSENSE] color stream started")

    # 자동 노출 안정화용
    for _ in range(15):
        pipeline.wait_for_frames()

    return pipeline


def main():
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)

    pc_addr = (PC_IP, UDP_VIDEO_PORT)

    pipeline = start_realsense_color()

    frame_id = 0
    last_debug_time = 0.0

    print(f"[UDP] sending RealSense RGB to {PC_IP}:{UDP_VIDEO_PORT}")
    print("[INFO] Ctrl+C 로 종료")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                print("[WARN] no color frame")
                continue

            rs_frame_number = color_frame.get_frame_number()

            color_image = np.asanyarray(color_frame.get_data())

            # 실시간 여부 확인용 표시
            cv2.putText(
                color_image,
                f"RS frame: {rs_frame_number}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                color_image,
                time.strftime("%H:%M:%S"),
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            ok, buffer = cv2.imencode(
                ".jpg",
                color_image,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )

            if not ok:
                print("[WARN] jpg encode failed")
                continue

            data = buffer.tobytes()
            total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE

            for idx in range(total_chunks):
                chunk = data[idx * CHUNK_SIZE:(idx + 1) * CHUNK_SIZE]
                header = struct.pack("!IHH", frame_id, total_chunks, idx)
                udp_sock.sendto(header + chunk, pc_addr)

            now = time.time()
            if now - last_debug_time >= 2.0:
                print(
                    f"[UDP] sent frame_id={frame_id}, "
                    f"rs_frame={rs_frame_number}, "
                    f"chunks={total_chunks}, "
                    f"mean={color_image.mean():.1f}"
                )
                last_debug_time = now

            frame_id = (frame_id + 1) % 4294967295

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt")

    finally:
        pipeline.stop()
        udp_sock.close()
        print("[INFO] sender stopped")


if __name__ == "__main__":
    main()