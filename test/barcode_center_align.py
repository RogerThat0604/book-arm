import time
import cv2
import socket
import struct
from pymycobot import MyCobot280

# ==============================
# PC UDP 설정
# ==============================
PC_IP = "192.168.0.52"
PC_PORT = 5000

# ==============================
# 로봇 설정
# ==============================
PORT = "/dev/ttyJETCOBOT"
BAUD = 1000000

SPEED = 8
MODE_L = 1

# 0 = X축, 1 = Y축
# 책장과 평행하게 움직이는 축
SCAN_AXIS = 1

STEP_MM = 5
MAX_MOVE_MM = 120

CENTER_TOL_PX = 40

# 특정 QR만 추적하려면 문자열 입력
TARGET_CODE = None
# TARGET_CODE = "BOOK_001"

# ==============================
# 카메라 설정
# ==============================
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

JPEG_QUALITY = 60
MAX_PACKET_SIZE = 1400

qr_detector = cv2.QRCodeDetector()


def valid_coords(coords):
    return isinstance(coords, list) and len(coords) == 6


def find_target_qr(frame):
    data, points, _ = qr_detector.detectAndDecode(frame)

    if points is None or data == "":
        return None

    if TARGET_CODE is not None and data != TARGET_CODE:
        return None

    pts = points[0].astype(int)

    x_min = min(p[0] for p in pts)
    y_min = min(p[1] for p in pts)
    x_max = max(p[0] for p in pts)
    y_max = max(p[1] for p in pts)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    return {
        "data": data,
        "points": pts,
        "cx": cx,
        "cy": cy,
        "w": x_max - x_min,
        "h": y_max - y_min,
    }


def send_udp_frame(sock, addr, frame, frame_id):
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )

    if not ok:
        return

    data = encoded.tobytes()
    total_packets = (len(data) + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE

    for packet_id in range(total_packets):
        start = packet_id * MAX_PACKET_SIZE
        end = start + MAX_PACKET_SIZE
        chunk = data[start:end]

        header = struct.pack("!IHH", frame_id, packet_id, total_packets)
        sock.sendto(header + chunk, addr)


def draw_info(frame, qr, center_x, error_x=None, status=""):
    h, _ = frame.shape[:2]

    cv2.line(frame, (center_x, 0), (center_x, h), (255, 0, 0), 2)

    if qr is None:
        cv2.putText(
            frame,
            "NO QR",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )
        return

    pts = qr["points"]

    for i in range(4):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % 4])
        cv2.line(frame, p1, p2, (0, 255, 0), 2)

    cv2.circle(frame, (qr["cx"], qr["cy"]), 5, (0, 0, 255), -1)

    cv2.putText(
        frame,
        f"QR={qr['data']}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    if error_x is not None:
        cv2.putText(
            frame,
            f"error_x={error_x}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    if status:
        cv2.putText(
            frame,
            status,
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


def move_horizontal(mc, base_pose, current_pose, direction):
    target = current_pose.copy()

    # QR 위치에 맞춰 수평 이동
    target[SCAN_AXIS] += direction * STEP_MM

    # 수평축 외에는 전부 기준 자세 유지
    # 즉 거리, 높이, 카메라 방향 고정
    for i in range(6):
        if i != SCAN_AXIS:
            target[i] = base_pose[i]

    print("MoveL target:", target)

    mc.send_coords(target, SPEED, MODE_L)
    time.sleep(0.25)

    after = mc.get_coords()

    if not valid_coords(after):
        print("[ERROR] 이동 후 좌표 읽기 실패")
        mc.stop()
        return None

    return after


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_addr = (PC_IP, PC_PORT)

    mc = MyCobot280(PORT, BAUD)
    mc.thread_lock = True
    time.sleep(1)

    mc.power_on()
    time.sleep(1)

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] 카메라 열기 실패")
        sock.close()
        return

    base_pose = mc.get_coords()
    print("base_pose:", base_pose)

    if not valid_coords(base_pose):
        print("[ERROR] 로봇 좌표 읽기 실패")
        cap.release()
        sock.close()
        return

    current_pose = base_pose.copy()
    start_axis_value = base_pose[SCAN_AXIS]
    frame_id = 0

    print("[START] QR MoveL Align")
    print("SCAN_AXIS:", "X" if SCAN_AXIS == 0 else "Y")
    print("TARGET_CODE:", TARGET_CODE)

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[WARN] 프레임 읽기 실패")
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            h, w = frame.shape[:2]
            center_x = w // 2

            qr = find_target_qr(frame)

            if qr is None:
                draw_info(frame, None, center_x)
                print("NO QR")

            else:
                error_x = qr["cx"] - center_x

                print(
                    f"QR={qr['data']}, "
                    f"cx={qr['cx']}, "
                    f"error_x={error_x}"
                )

                if abs(error_x) <= CENTER_TOL_PX:
                    draw_info(frame, qr, center_x, error_x, "CENTERED - STOP")
                    send_udp_frame(sock, udp_addr, frame, frame_id)

                    print("[OK] QR 중앙 정렬 완료")
                    mc.stop()
                    break

                moved_dist = abs(current_pose[SCAN_AXIS] - start_axis_value)

                if moved_dist >= MAX_MOVE_MM:
                    draw_info(frame, qr, center_x, error_x, "LIMIT STOP")
                    send_udp_frame(sock, udp_addr, frame, frame_id)

                    print("[LIMIT] 최대 이동 범위 도달")
                    mc.stop()
                    break

                # QR이 오른쪽에 있으면 오른쪽으로 이동
                # 방향이 반대면 아래 한 줄의 부호를 반대로 바꾸세요.
                direction = 1 if error_x > 0 else -1

                status = "MOVE RIGHT" if direction > 0 else "MOVE LEFT"
                draw_info(frame, qr, center_x, error_x, status)

                next_pose = move_horizontal(
                    mc,
                    base_pose,
                    current_pose,
                    direction
                )

                if next_pose is None:
                    break

                current_pose = next_pose

            send_udp_frame(sock, udp_addr, frame, frame_id)
            frame_id += 1

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("[STOP] 사용자 중단")
        mc.stop()

    finally:
        cap.release()
        sock.close()
        mc.stop()

        print("final coords:", mc.get_coords())
        print("종료 - 서보 유지")


if __name__ == "__main__":
    main()