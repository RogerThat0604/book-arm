import cv2
import time
import json
import socket
import struct
import threading
from pymycobot import MyCobot280

# =========================
# Network
# =========================
ROBOT_IP = "192.168.0.37"
PC_IP = "192.168.0.52"

UDP_VIDEO_PORT = 5000
TCP_CONTROL_PORT = 6000

# =========================
# Camera
# =========================
CAMERA_PATH = "/dev/jetcocam0"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 60
FPS_LIMIT = 15

CHUNK_SIZE = 1300

# =========================
# Robot
# =========================
PORT = "/dev/ttyJETCOBOT"
BAUD = 1000000

SPEED = 12
MODE_L = 1

X_MIN, X_MAX = 30, 230
Y_MIN, Y_MAX = -180, 100
Z_MIN, Z_MAX = 120, 410

stop_flag = False
robot_lock = threading.Lock()

FIXED_RX = None
FIXED_RY = None
FIXED_RZ = None


def clamp(v, min_v, max_v):
    return max(min_v, min(v, max_v))


def valid_coords(coords):
    return isinstance(coords, list) and len(coords) == 6


def send_udp_video():
    global stop_flag

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pc_addr = (PC_IP, UDP_VIDEO_PORT)

    cap = cv2.VideoCapture(CAMERA_PATH)

    if not cap.isOpened():
        print("[VIDEO] 카메라를 열 수 없습니다.")
        stop_flag = True
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    frame_id = 0
    interval = 1.0 / FPS_LIMIT

    print("[VIDEO] UDP 영상 송신 시작")

    while not stop_flag:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("[VIDEO] 프레임 읽기 실패")
            continue

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        ok, buffer = cv2.imencode(".jpg", frame, encode_param)

        if not ok:
            continue

        data = buffer.tobytes()
        total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE

        for chunk_idx in range(total_chunks):
            chunk = data[chunk_idx * CHUNK_SIZE:(chunk_idx + 1) * CHUNK_SIZE]
            header = struct.pack("!IHH", frame_id, total_chunks, chunk_idx)
            udp_sock.sendto(header + chunk, pc_addr)

        frame_id = (frame_id + 1) % 4294967295

        elapsed = time.time() - start_time
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    udp_sock.close()
    print("[VIDEO] UDP 영상 송신 종료")


def handle_command(mc, command):
    global stop_flag

    cmd_type = command.get("type")

    if cmd_type == "stop":
        print("[CONTROL] STOP")
        mc.stop()
        stop_flag = True
        return {"ok": True, "message": "stopped"}

    if cmd_type == "status":
        coords = mc.get_coords()
        return {"ok": True, "coords": coords}

    if cmd_type == "move_axis":
        axis = command.get("axis")
        delta = float(command.get("delta", 0))
        reason = command.get("reason", "")

        with robot_lock:
            current = mc.get_coords()

            if not valid_coords(current):
                return {"ok": False, "message": "get_coords failed"}

            target = current.copy()

            if axis == "X":
                target[0] = clamp(target[0] + delta, X_MIN, X_MAX)
            elif axis == "Y":
                target[1] = clamp(target[1] + delta, Y_MIN, Y_MAX)
            elif axis == "Z":
                target[2] = clamp(target[2] + delta, Z_MIN, Z_MAX)
            else:
                return {"ok": False, "message": "unknown axis"}

            # 핵심: 카메라가 QR 책면과 평행하도록 자세 고정
            target[3] = FIXED_RX
            target[4] = FIXED_RY
            target[5] = FIXED_RZ

            print(f"[MOVE] {reason} axis={axis}, delta={delta}, target={target}")

            mc.send_coords(target, SPEED, MODE_L)

        return {"ok": True, "target": target}

    return {"ok": False, "message": "unknown command type"}


def tcp_control_server(mc):
    global stop_flag

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((ROBOT_IP, TCP_CONTROL_PORT))
    server.listen(1)

    print(f"[CONTROL] TCP 서버 대기: {ROBOT_IP}:{TCP_CONTROL_PORT}")

    while not stop_flag:
        conn, addr = server.accept()
        print("[CONTROL] PC 연결:", addr)

        with conn:
            buffer = ""

            while not stop_flag:
                data = conn.recv(4096)

                if not data:
                    print("[CONTROL] PC 연결 종료")
                    break

                buffer += data.decode("utf-8", errors="ignore")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)

                    if not line.strip():
                        continue

                    try:
                        command = json.loads(line)
                        response = handle_command(mc, command)
                    except Exception as e:
                        response = {"ok": False, "message": str(e)}

                    conn.sendall((json.dumps(response) + "\n").encode("utf-8"))

    server.close()
    print("[CONTROL] TCP 서버 종료")


def main():
    global stop_flag, FIXED_RX, FIXED_RY, FIXED_RZ

    print("[ROBOT] MyCobot 연결 중...")

    mc = MyCobot280(PORT, BAUD)
    mc.thread_lock = True

    time.sleep(1)
    mc.power_on()
    time.sleep(1)

    mc.set_fresh_mode(0)
    time.sleep(0.2)

    home = mc.get_coords()
    print("[ROBOT] home:", home)

    if not valid_coords(home):
        print("[ROBOT] 좌표 읽기 실패")
        return

    # 시작 자세를 카메라 정면 기준 자세로 고정
    FIXED_RX = home[3]
    FIXED_RY = home[4]
    FIXED_RZ = home[5]

    print("[ROBOT] fixed orientation:", [FIXED_RX, FIXED_RY, FIXED_RZ])

    video_thread = threading.Thread(target=send_udp_video, daemon=True)
    video_thread.start()

    try:
        tcp_control_server(mc)

    except KeyboardInterrupt:
        print("\n[ROBOT] KeyboardInterrupt")

    finally:
        stop_flag = True
        mc.stop()
        print("[ROBOT] 최종 좌표:", mc.get_coords())
        print("[ROBOT] 종료")


if __name__ == "__main__":
    main()