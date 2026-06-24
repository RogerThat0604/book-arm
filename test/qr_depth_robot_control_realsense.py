# qr_depth_movel_pc.py
# PC side
# 목표:
# - RealSense RGB UDP 영상 수신
# - QR 중심 검출
# - QR 중심 좌표(cx, cy)와 pixel error를 로봇에 전송
# - 로봇은 depth를 이용해 MoveL로 XYZ만 움직임
# - 카메라 각도는 로봇 쪽 locked orientation으로 유지
# - QR이 카메라 정중앙 + 10cm 거리일 때 정지

import cv2
import json
import time
import math
import socket
import struct
import threading
import queue
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image, ImageDraw, ImageFont


PC_IP = "192.168.0.52"
ROBOT_IP = "192.168.0.37"

UDP_VIDEO_PORT = 5000
TCP_CONTROL_PORT = 6000

DISPLAY_SCALE = 1.5

# 화면 표시만 좌우반전합니다. 제어 좌표에는 영향을 주지 않습니다.
MIRROR_DISPLAY = True

FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

latest_depth_m = None
latest_robot_zone = ""
latest_robot_speed = None
latest_robot_step = None
latest_point_cm = None
latest_depth_error_cm = None
latest_step_cm = None

TOL_X = 18
TOL_Y = 18

COMMAND_INTERVAL = 0.09

QR_WORKER_INTERVAL = 0.004

latest_frame = None
latest_qr_info = None
latest_udp_frame_id = -1
latest_robot_state = ""

frame_lock = threading.Lock()
qr_lock = threading.Lock()

command_queue = queue.Queue(maxsize=1)
stop_flag = False


def put_latest_command(command):
    try:
        while not command_queue.empty():
            command_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        command_queue.put_nowait(command)
    except queue.Full:
        pass


def connect_robot_tcp():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ROBOT_IP, TCP_CONTROL_PORT))
    sock.settimeout(1.0)
    print("[TCP] 로봇 연결 완료")
    return sock


def send_command(sock, command):
    sock.sendall((json.dumps(command) + "\n").encode("utf-8"))

    buffer = ""

    while "\n" not in buffer:
        data = sock.recv(4096)

        if not data:
            raise ConnectionError("TCP 연결 끊김")

        buffer += data.decode("utf-8", errors="ignore")

    line, _ = buffer.split("\n", 1)
    return json.loads(line)

def put_korean_text(frame, text, position, font_size=22, color=(255, 255, 255)):
    """
    OpenCV putText는 한글이 깨지므로 PIL + NanumGothic으로 표시합니다.
    color는 OpenCV BGR 기준입니다.
    """
    try:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(FONT_PATH, font_size)
        rgb_color = (color[2], color[1], color[0])
        draw.text(position, text, font=font, fill=rgb_color)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(
            frame,
            text.encode("ascii", errors="ignore").decode("ascii", errors="ignore"),
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        return frame


def calc_center_distance_angle(info):
    if info is None:
        return None, None

    ex = float(info["ex"])
    ey = float(info["ey"])

    distance_px = math.sqrt(ex * ex + ey * ey)

    # 화면 중심 기준 방향각입니다.
    # 0도: 오른쪽, 90도: 아래쪽, -90도: 위쪽
    angle_deg = math.degrees(math.atan2(ey, ex))

    return distance_px, angle_deg


def udp_receiver():
    global latest_frame, latest_udp_frame_id, stop_flag

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
    udp_sock.bind((PC_IP, UDP_VIDEO_PORT))
    udp_sock.settimeout(1.0)

    frames = {}
    frame_times = {}

    print(f"[UDP] 영상 수신 대기: {PC_IP}:{UDP_VIDEO_PORT}")

    while not stop_flag:
        try:
            packet, _ = udp_sock.recvfrom(65535)
        except socket.timeout:
            continue

        now = time.time()

        if len(packet) < 8:
            continue

        frame_id, total_chunks, chunk_idx = struct.unpack("!IHH", packet[:8])
        chunk = packet[8:]

        if latest_udp_frame_id >= 0 and frame_id < latest_udp_frame_id:
            continue

        if frame_id not in frames:
            frames[frame_id] = [None] * total_chunks
            frame_times[frame_id] = now

        if chunk_idx < total_chunks:
            frames[frame_id][chunk_idx] = chunk

        old_ids = [fid for fid, ts in frame_times.items() if now - ts > 0.5]
        for fid in old_ids:
            frames.pop(fid, None)
            frame_times.pop(fid, None)

        if not all(part is not None for part in frames.get(frame_id, [])):
            continue

        jpg_data = b"".join(frames[frame_id])
        frames.pop(frame_id, None)
        frame_times.pop(frame_id, None)

        np_data = np.frombuffer(jpg_data, dtype=np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        latest_udp_frame_id = frame_id

        with frame_lock:
            latest_frame = frame

    udp_sock.close()


def decode_qr_multi(gray):
    candidates = []

    # 너무 무겁게 돌리면 실시간성이 떨어져서 gray + clahe만 우선 사용
    candidates.append((gray, 1.0, "gray"))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    candidates.append((clahe.apply(gray), 1.0, "clahe"))

    for img, scale, method in candidates:
        codes = [code for code in decode(img) if code.type == "QRCODE"]
        if codes:
            return codes, scale, method

    return [], 1.0, "none"


def parse_qr_label(data):
    try:
        book = json.loads(data)
        title = book.get("title", "")
        category = book.get("category", "")
        book_id = book.get("id", "")

        if title and category:
            return f"{title} / {category}"
        if title:
            return title
        if book_id:
            return book_id
        return data
    except Exception:
        return data


def get_qr_info(frame):
    h, w = frame.shape[:2]
    frame_cx = w // 2
    frame_cy = h // 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    codes, scale, method = decode_qr_multi(gray)

    if not codes:
        return None

    code = max(codes, key=lambda c: c.rect.width * c.rect.height)
    data = code.data.decode("utf-8", errors="ignore")
    label = parse_qr_label(data)

    x, y, bw, bh = code.rect

    if scale != 1.0:
        x = int(x / scale)
        y = int(y / scale)
        bw = int(bw / scale)
        bh = int(bh / scale)

        if code.polygon and len(code.polygon) >= 4:
            pts = [(int(p.x / scale), int(p.y / scale)) for p in code.polygon]
        else:
            pts = None
    else:
        if code.polygon and len(code.polygon) >= 4:
            pts = [(p.x, p.y) for p in code.polygon]
        else:
            pts = None

    if pts:
        cx = int(sum(px for px, py in pts) / len(pts))
        cy = int(sum(py for px, py in pts) / len(pts))
    else:
        cx = x + bw // 2
        cy = y + bh // 2

    ex = cx - frame_cx
    ey = cy - frame_cy
    dist = math.sqrt(ex * ex + ey * ey)

    return {
        "data": data,
        "label": label,
        "rect": (x, y, bw, bh),
        "pts": pts,
        "center": (cx, cy),
        "frame_center": (frame_cx, frame_cy),
        "ex": ex,
        "ey": ey,
        "dist": dist,
        "method": method,
    }


def qr_worker():
    global latest_qr_info, stop_flag

    while not stop_flag:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.002)
            continue

        info = get_qr_info(frame)

        with qr_lock:
            latest_qr_info = info

        time.sleep(QR_WORKER_INTERVAL)


def tcp_sender_worker():
    global stop_flag, latest_robot_state, latest_depth_m, latest_robot_zone, latest_robot_speed, latest_robot_step, latest_point_cm, latest_depth_error_cm, latest_step_cm

    sock = connect_robot_tcp()
    last_send = 0.0

    try:
        while not stop_flag:
            try:
                command = command_queue.get(timeout=0.02)
            except queue.Empty:
                continue

            now = time.time()

            if command.get("type") == "visual_servo_step" and now - last_send < COMMAND_INTERVAL:
                continue

            try:
                response = send_command(sock, command)
                latest_robot_state = str(response.get("state", response.get("message", "")))

                if "distance_m" in response:
                    latest_depth_m = response.get("distance_m")

                latest_point_cm = response.get("point_cm", None)
                latest_depth_error_cm = response.get("depth_error_cm", None)
                latest_step_cm = response.get("step_cm", None)

                latest_robot_zone = str(response.get("zone", ""))
                latest_robot_speed = response.get("speed", None)
                latest_robot_step = response.get("step_mm", None)

                if command.get("type") != "visual_servo_step" or response.get("state") in ("aligned", "small_step_stop"):
                    print("[CMD]", command, "[RES]", response)

                last_send = now
            except Exception as e:
                print("[TCP ERROR]", e)
                time.sleep(0.2)

    finally:
        try:
            send_command(sock, {"type": "hold"})
        except Exception:
            pass
        sock.close()


def mirror_info_for_display(info, frame_width):
    if info is None:
        return None

    copied = info.copy()

    x, y, w, h = copied["rect"]
    cx, cy = copied["center"]

    copied["rect"] = (frame_width - x - w, y, w, h)
    copied["center"] = (frame_width - cx, cy)

    if copied["pts"]:
        copied["pts"] = [(frame_width - px, py) for px, py in copied["pts"]]

    copied["ex"] = -copied["ex"]

    return copied


def draw_center_guide(frame):
    h, w = frame.shape[:2]
    frame_cx = w // 2
    frame_cy = h // 2

    cv2.circle(frame, (frame_cx, frame_cy), 6, (255, 0, 0), -1)

    cv2.line(frame, (frame_cx - 30, frame_cy), (frame_cx + 30, frame_cy), (255, 0, 0), 2)
    cv2.line(frame, (frame_cx, frame_cy - 30), (frame_cx, frame_cy + 30), (255, 0, 0), 2)

    cv2.rectangle(
        frame,
        (frame_cx - TOL_X, frame_cy - TOL_Y),
        (frame_cx + TOL_X, frame_cy + TOL_Y),
        (255, 255, 255),
        2
    )

    return frame


def draw_info(frame, info):
    frame = draw_center_guide(frame)

    h, w = frame.shape[:2]
    frame_cx = w // 2
    frame_cy = h // 2

    frame = put_korean_text(
        frame,
        f"UDP 프레임: {latest_udp_frame_id} / 로봇 상태: {latest_robot_state} / 미러보기: {MIRROR_DISPLAY}",
        (20, h - 32),
        20,
        (0, 255, 255)
    )

    frame = put_korean_text(
        frame,
        "MoveL XYZ 제어 / 목표거리: RealSense Depth 기준 20cm",
        (20, 25),
        22,
        (255, 255, 0)
    )

    if latest_depth_m is not None:
        frame = put_korean_text(
            frame,
            f"Depth 거리(Z): {float(latest_depth_m) * 100.0:.1f} cm",
            (20, 55),
            22,
            (0, 255, 255)
        )

    if latest_point_cm is not None:
        try:
            x_cm, y_cm, z_cm = latest_point_cm
            frame = put_korean_text(
                frame,
                f"QR 실제 위치: X={x_cm:+.1f}cm / Y={y_cm:+.1f}cm / Z={z_cm:.1f}cm",
                (20, 85),
                20,
                (255, 255, 255)
            )
        except Exception:
            pass

    if latest_depth_error_cm is not None:
        frame = put_korean_text(
            frame,
            f"목표 20cm까지 전후 오차: {float(latest_depth_error_cm):+.1f} cm",
            (20, 113),
            20,
            (255, 255, 255)
        )

    if latest_step_cm is not None:
        try:
            sx, sy, sz = latest_step_cm
            frame = put_korean_text(
                frame,
                f"이번 이동량: X={sx:+.1f}cm / Y={sy:+.1f}cm / Z={sz:+.1f}cm",
                (20, 141),
                20,
                (255, 255, 255)
            )
        except Exception:
            pass

    if latest_robot_zone:
        frame = put_korean_text(
            frame,
            f"제어영역: {latest_robot_zone} / 속도: {latest_robot_speed}",
            (20, 169),
            18,
            (255, 255, 0)
        )

    if info is None:
        frame = put_korean_text(
            frame,
            "QR 인식 안 됨 / HOLD",
            (20, 205),
            26,
            (0, 0, 255)
        )
        return frame

    x, y, bw, bh = info["rect"]
    cx, cy = info["center"]

    distance_px, angle_deg = calc_center_distance_angle(info)

    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    if info["pts"]:
        pts = info["pts"]
        for i in range(len(pts)):
            cv2.line(frame, pts[i], pts[(i + 1) % len(pts)], (255, 0, 0), 2)

    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
    cv2.line(frame, (frame_cx, frame_cy), (cx, cy), (0, 255, 255), 2)

    frame = put_korean_text(
        frame,
        f"QR 정보: {info['label'][:40]}",
        (20, 205),
        22,
        (0, 255, 0)
    )

    frame = put_korean_text(
        frame,
        f"QR 중심: ({cx}, {cy}) / 프레임 중심: ({frame_cx}, {frame_cy})",
        (20, 235),
        20,
        (255, 255, 255)
    )

    frame = put_korean_text(
        frame,
        f"화면 방향각: {angle_deg:.1f}도 / 픽셀값은 표시용이며 이동거리는 cm 기준",
        (20, 263),
        20,
        (255, 255, 255)
    )

    frame = put_korean_text(
        frame,
        f"검출 방식: {info['method']} / 실제 이동 기준: RealSense 3D 좌표(cm)",
        (20, 291),
        18,
        (255, 255, 0)
    )

    if abs(info["ex"]) <= TOL_X and abs(info["ey"]) <= TOL_Y:
        frame = put_korean_text(
            frame,
            "화면 중심 정렬 범위 안에 있음",
            (20, 323),
            24,
            (0, 255, 0)
        )

    return frame


def main():
    global stop_flag, MIRROR_DISPLAY

    threading.Thread(target=udp_receiver, daemon=True).start()
    threading.Thread(target=qr_worker, daemon=True).start()
    threading.Thread(target=tcp_sender_worker, daemon=True).start()

    hold_sent = False

    print("[INFO] MoveL 기반 QR Depth 정렬")
    print("[INFO] 목표: 카메라 자세 유지 + QR 중앙 + depth 10cm")
    print("[INFO] q: quit / s: hold / v: display mirror toggle")

    while not stop_flag:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.002)
            continue

        with qr_lock:
            info = None if latest_qr_info is None else latest_qr_info.copy()

        if info is not None:
            hold_sent = False

            cx, cy = info["center"]

            put_latest_command(
                {
                    "type": "visual_servo_step",
                    "cx": int(cx),
                    "cy": int(cy),
                    "ex": float(info["ex"]),
                    "ey": float(info["ey"]),
                }
            )
        else:
            if not hold_sent:
                put_latest_command({"type": "hold"})
                hold_sent = True

        display_frame = frame.copy()

        if MIRROR_DISPLAY:
            display_frame = cv2.flip(display_frame, 1)
            display_info = mirror_info_for_display(info, frame.shape[1])
        else:
            display_info = info

        display_frame = draw_info(display_frame, display_info)

        if DISPLAY_SCALE != 1.0:
            display_frame = cv2.resize(
                display_frame,
                None,
                fx=DISPLAY_SCALE,
                fy=DISPLAY_SCALE,
                interpolation=cv2.INTER_LINEAR
            )

        cv2.imshow("QR Depth MoveL Visual Servo PC", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            put_latest_command({"type": "hold"})
            hold_sent = True

        if key == ord("v"):
            MIRROR_DISPLAY = not MIRROR_DISPLAY
            print("[INFO] MIRROR_DISPLAY =", MIRROR_DISPLAY)

        if key == ord("q"):
            break

    stop_flag = True
    put_latest_command({"type": "hold"})
    cv2.destroyAllWindows()
    print("[PC] 종료")


if __name__ == "__main__":
    main()