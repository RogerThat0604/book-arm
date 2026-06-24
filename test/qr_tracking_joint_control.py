import cv2
import time
import json
import socket
import struct
import threading
from pymycobot import MyCobot280

ROBOT_IP = "192.168.0.37"
PC_IP = "192.168.0.52"

UDP_VIDEO_PORT = 5000
TCP_CONTROL_PORT = 6000

CAMERA_PATH = "/dev/jetcocam0"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 60
FPS_LIMIT = 20
CHUNK_SIZE = 1300

PORT = "/dev/ttyJETCOBOT"
BAUD = 1000000

# 로봇 내부 motion loop가 부드럽게 따라가는 구조입니다.
SPEED = 60

J1_MIN, J1_MAX = -90.0, 90.0
J4_MIN, J4_MAX = -120.0, 120.0

MOTION_HZ = 18.0
MAX_STEP_PER_TICK_J1 = 1.8
MAX_STEP_PER_TICK_J4 = 1.8
ANGLE_EPS = 0.08

stop_flag = False
hold_flag = False

robot_lock = threading.Lock()

current_target_angles = None
desired_angles = None


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def valid_angles(angles):
    return isinstance(angles, list) and len(angles) >= 6


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
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            continue

        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )

        if not ok:
            continue

        data = buffer.tobytes()
        total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE

        for idx in range(total_chunks):
            chunk = data[idx * CHUNK_SIZE:(idx + 1) * CHUNK_SIZE]
            header = struct.pack("!IHH", frame_id, total_chunks, idx)
            udp_sock.sendto(header + chunk, pc_addr)

        frame_id = (frame_id + 1) % 4294967295

        sleep_time = interval - (time.time() - start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    udp_sock.close()
    print("[VIDEO] UDP 영상 송신 종료")


def motion_loop(mc):
    """
    TCP 명령이 들어올 때마다 바로 send_angles() 하지 않습니다.
    PC는 desired_angles만 갱신하고, 이 스레드가 일정 주기로 조금씩 따라갑니다.
    """
    global stop_flag, hold_flag, current_target_angles, desired_angles

    interval = 1.0 / MOTION_HZ

    print("[MOTION] 내부 부드러운 추종 루프 시작")

    while not stop_flag:
        start = time.time()

        if hold_flag:
            time.sleep(interval)
            continue

        with robot_lock:
            if not valid_angles(current_target_angles) or not valid_angles(desired_angles):
                pass
            else:
                next_angles = current_target_angles.copy()

                diff_j1 = desired_angles[0] - current_target_angles[0]
                diff_j4 = desired_angles[3] - current_target_angles[3]

                step_j1 = clamp(diff_j1, -MAX_STEP_PER_TICK_J1, MAX_STEP_PER_TICK_J1)
                step_j4 = clamp(diff_j4, -MAX_STEP_PER_TICK_J4, MAX_STEP_PER_TICK_J4)

                if abs(step_j1) >= ANGLE_EPS or abs(step_j4) >= ANGLE_EPS:
                    next_angles[0] = clamp(current_target_angles[0] + step_j1, J1_MIN, J1_MAX)
                    next_angles[3] = clamp(current_target_angles[3] + step_j4, J4_MIN, J4_MAX)

                    current_target_angles = next_angles.copy()

                    print(
                        f"[MOTION] J1={next_angles[0]:.2f}, "
                        f"J4={next_angles[3]:.2f}, "
                        f"step=({step_j1:.3f},{step_j4:.3f})"
                    )

                    mc.send_angles(next_angles, SPEED)

        elapsed = time.time() - start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("[MOTION] 종료")


def handle_command(mc, command):
    global stop_flag, hold_flag, current_target_angles, desired_angles

    cmd_type = command.get("type")

    if cmd_type == "stop":
        print("[CONTROL] STOP")
        mc.stop()
        stop_flag = True
        return {"ok": True, "message": "stopped"}

    if cmd_type == "hold":
        print("[CONTROL] HOLD")
        mc.stop()
        hold_flag = True

        with robot_lock:
            real = mc.get_angles()
            if valid_angles(real):
                current_target_angles = real.copy()
                desired_angles = real.copy()

        return {"ok": True, "message": "hold"}

    if cmd_type == "resume":
        print("[CONTROL] RESUME")
        hold_flag = False
        return {"ok": True, "message": "resume"}

    if cmd_type == "status":
        return {
            "ok": True,
            "angles": mc.get_angles(),
            "coords": mc.get_coords(),
            "current_target_angles": current_target_angles,
            "desired_angles": desired_angles,
        }

    if cmd_type == "refresh_angles":
        real = mc.get_angles()

        if not valid_angles(real):
            return {"ok": False, "message": "get_angles failed"}

        with robot_lock:
            current_target_angles = real.copy()
            desired_angles = real.copy()

        print("[CONTROL] angles refreshed:", real)

        return {
            "ok": True,
            "current_target_angles": current_target_angles,
            "desired_angles": desired_angles
        }

    if cmd_type == "joint_delta":
        dj1 = float(command.get("dj1", 0.0))
        dj4 = float(command.get("dj4", 0.0))
        reason = command.get("reason", "")

        hold_flag = False

        with robot_lock:
            if not valid_angles(desired_angles):
                real = mc.get_angles()
                if not valid_angles(real):
                    return {"ok": False, "message": "get_angles failed"}
                current_target_angles = real.copy()
                desired_angles = real.copy()

            desired_angles[0] = clamp(desired_angles[0] + dj1, J1_MIN, J1_MAX)
            desired_angles[3] = clamp(desired_angles[3] + dj4, J4_MIN, J4_MAX)

            print(
                f"[CONTROL] {reason} "
                f"add=({dj1:.3f},{dj4:.3f}) "
                f"desired J1={desired_angles[0]:.2f}, J4={desired_angles[3]:.2f}"
            )

            return {
                "ok": True,
                "desired_j1": desired_angles[0],
                "desired_j4": desired_angles[3]
            }

    return {"ok": False, "message": "unknown command"}


def tcp_control_server(mc):
    global stop_flag

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((ROBOT_IP, TCP_CONTROL_PORT))
    server.listen(1)

    print(f"[CONTROL] TCP 서버 대기: {ROBOT_IP}:{TCP_CONTROL_PORT}")

    while not stop_flag:
        try:
            conn, addr = server.accept()
        except OSError:
            break

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
    global stop_flag, current_target_angles, desired_angles

    print("[ROBOT] MyCobot 연결 중...")

    mc = MyCobot280(PORT, BAUD)
    mc.thread_lock = True

    time.sleep(1)
    mc.power_on()
    time.sleep(1)

    mc.set_fresh_mode(1)
    time.sleep(0.2)

    angles = mc.get_angles()

    if not valid_angles(angles):
        print("[ROBOT] 각도 읽기 실패")
        return

    current_target_angles = angles.copy()
    desired_angles = angles.copy()

    print("[ROBOT] initial angles:", angles)
    print("[ROBOT] initial coords:", mc.get_coords())

    threading.Thread(target=send_udp_video, daemon=True).start()
    threading.Thread(target=motion_loop, args=(mc,), daemon=True).start()

    try:
        tcp_control_server(mc)

    except KeyboardInterrupt:
        print("\n[ROBOT] KeyboardInterrupt")

    finally:
        stop_flag = True
        mc.stop()
        print("[ROBOT] final angles:", mc.get_angles())
        print("[ROBOT] 종료")


if __name__ == "__main__":
    main()