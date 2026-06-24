# aruco_height_z_robot.py
# Robot side
# ------------------------------------------------------------
# 목적:
# - 높이 보정: z 좌표만 사용 (send_coords, mode=1 / MoveL)
# - x, y, rx, ry, rz는 항상 고정값 유지 -> 카메라가 항상 같은 자세(정면)를 봄
# - 관절(J2/J4)을 직접 건드리지 않으므로 화면이 흔들리는 피드백 루프가 생기지 않음
#
# 사용 전 필수:
#   1. robothome 등으로 로봇을 원하는 시작 자세로 이동
#   2. mc.get_coords() 로 그 자세의 좌표를 읽어서 FIXED_X, FIXED_Y, FIXED_RX, FIXED_RY, FIXED_RZ,
#      START_Z 값을 실측값으로 채워 넣기
#   3. Z_SIGN 부호를 실측으로 확정 (z를 +30 보내봤을 때 카메라가 위로 가면 +1, 아래로 가면 -1)
#
# 실행:
#   python3 aruco_height_z_robot.py
# ------------------------------------------------------------

import cv2
import time
import json
import socket
import struct
import threading
import numpy as np
import pyrealsense2 as rs
from pymycobot import MyCobot280


ROBOT_IP = "192.168.0.37"
PC_IP = "192.168.0.52"

UDP_VIDEO_PORT = 5000
TCP_CONTROL_PORT = 6000

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
JPEG_QUALITY = 70
CHUNK_SIZE = 1300

PORT = "/dev/ttyJETCOBOT"
BAUD = 1000000

# =========================
# 고정 자세 (실측값으로 반드시 교체)
# robothome 직후 mc.get_coords() 결과를 그대로 사용
# 예시 로그 기준: [57.8, -66.1, 407.0, -94.13, 0.65, -91.63]
# =========================
FIXED_X = 57.8
FIXED_Y = -66.1
FIXED_RX = -94.13
FIXED_RY = 0.65
FIXED_RZ = -91.63

START_Z = 407.0          # 시작 높이
MIN_Z = 250.0             # 안전 하한 (반드시 작업환경에 맞게 재설정)
MAX_Z = 450.0             # 안전 상한 (반드시 작업환경에 맞게 재설정)

# ey(화면상 마커 위치, 아래로 갈수록 +)가 양수일 때 z를 어느 방향으로 바꿀지.
# 실측 후 반대로 움직이면 이 값을 뒤집을 것.
# 가설: 마커가 화면 아래(ey>0) -> 카메라(=z)가 너무 높다 -> z를 낮춘다 -> 우선 -1로 시작
Z_SIGN = -1.0

MM_PER_PX = 0.35          # ey 1px 당 보정할 z(mm). 처음엔 작게 시작해서 튜닝
MIN_Z_STEP_MM = 1.0
MAX_Z_STEP_MM = 12.0

CENTER_Y_TOL_PX = 25
CONTROL_INTERVAL_SEC = 0.22
HOLD_INTERVAL_SEC = 0.45
MAX_CORRECTION_COUNT = 800

MOVE_SPEED = 15
MOVE_MODE = 1   # 1 = MoveL (직선) - z만 바뀌고 자세 고정 보장

stop_flag = False
robot_lock = threading.Lock()

last_control_time = 0.0
last_hold_time = 0.0
correction_count = 0
current_z = START_Z


def clamp(value, lo, hi):
    return max(lo, min(value, hi))


def safe_get_coords(mc):
    try:
        with robot_lock:
            return mc.get_coords()
    except Exception as e:
        print("[ROBOT WARN] get_coords failed:", e)
        return None


def safe_send_coords(mc, coords, speed, mode):
    try:
        with robot_lock:
            return mc.send_coords(coords, int(speed), int(mode))
    except Exception as e:
        print("[ROBOT WARN] send_coords failed:", e)
        return {"exception": str(e)}


def safe_stop(mc):
    try:
        with robot_lock:
            return mc.stop()
    except Exception as e:
        print("[ROBOT WARN] stop failed:", e)
        return None


def create_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)

    profile = pipeline.start(config)
    device = profile.get_device()

    print("[REALSENSE] device:", device.get_info(rs.camera_info.name))
    print("[REALSENSE] serial:", device.get_info(rs.camera_info.serial_number))

    for _ in range(15):
        pipeline.wait_for_frames()

    print("[REALSENSE] RGB started")
    return pipeline


def send_udp_video():
    global stop_flag

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
    pc_addr = (PC_IP, UDP_VIDEO_PORT)

    try:
        pipeline = create_realsense()
    except Exception as e:
        print("[REALSENSE] start failed:", e)
        stop_flag = True
        return

    frame_id = 0
    last_log_time = 0.0

    try:
        while not stop_flag:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            cv2.putText(
                frame,
                "Height Z-only (pose fixed)",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            ok, buffer = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
            )

            if not ok:
                continue

            data = buffer.tobytes()
            total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE

            for idx in range(total_chunks):
                chunk = data[idx * CHUNK_SIZE:(idx + 1) * CHUNK_SIZE]
                header = struct.pack("!IHH", frame_id, total_chunks, idx)
                udp_sock.sendto(header + chunk, pc_addr)

            now = time.time()
            if now - last_log_time >= 2.0:
                print(f"[VIDEO] sent frame={frame_id}, chunks={total_chunks}")
                last_log_time = now

            frame_id = (frame_id + 1) % 4294967295

    finally:
        pipeline.stop()
        udp_sock.close()
        print("[VIDEO] UDP 송신 종료")


def calc_z_step(ey):
    abs_ey = abs(float(ey))
    raw = abs_ey * MM_PER_PX
    step = clamp(raw, MIN_Z_STEP_MM, MAX_Z_STEP_MM)

    direction = 1.0 if ey > 0 else -1.0
    z_step = Z_SIGN * step * direction

    return z_step


def handle_height_step(mc, command):
    global last_control_time, last_hold_time, correction_count, current_z

    now = time.time()
    ey = float(command.get("ey", 0.0))

    inside_deadzone = abs(ey) <= CENTER_Y_TOL_PX

    if inside_deadzone:
        if now - last_hold_time >= HOLD_INTERVAL_SEC:
            safe_stop(mc)
            last_hold_time = now
        return {
            "ok": True,
            "state": "deadzone_stop",
            "ey_px": ey,
            "current_z": current_z,
            "correction_count": correction_count,
        }

    if now - last_control_time < CONTROL_INTERVAL_SEC:
        return {"ok": True, "state": "skip_interval", "ey_px": ey}

    if correction_count >= MAX_CORRECTION_COUNT:
        safe_stop(mc)
        return {"ok": True, "state": "max_correction_stop", "ey_px": ey}

    z_step = calc_z_step(ey)
    target_z = clamp(current_z + z_step, MIN_Z, MAX_Z)

    if abs(target_z - current_z) < 0.3:
        if now - last_hold_time >= HOLD_INTERVAL_SEC:
            safe_stop(mc)
            last_hold_time = now
        return {
            "ok": True,
            "state": "small_delta_hold",
            "ey_px": ey,
            "current_z": current_z,
        }

    # x, y, rx, ry, rz는 절대 변경하지 않는다 - z만 갱신
    target_coords = [FIXED_X, FIXED_Y, target_z, FIXED_RX, FIXED_RY, FIXED_RZ]

    ret = safe_send_coords(mc, target_coords, MOVE_SPEED, MOVE_MODE)

    current_z = target_z
    last_control_time = now
    correction_count += 1

    print(
        f"[Z_HEIGHT] count={correction_count} ey={ey:+.1f}px "
        f"z_step={z_step:+.2f} target_z={target_z:.2f}"
    )

    return {
        "ok": True,
        "state": "height_sent",
        "ey_px": ey,
        "z_step": z_step,
        "target_z": target_z,
        "target_coords": target_coords,
        "send_return": ret,
        "correction_count": correction_count,
    }


def handle_reset(mc):
    global correction_count, last_control_time, current_z

    correction_count = 0
    last_control_time = 0.0
    current_z = START_Z

    target_coords = [FIXED_X, FIXED_Y, START_Z, FIXED_RX, FIXED_RY, FIXED_RZ]
    ret = safe_send_coords(mc, target_coords, MOVE_SPEED, MOVE_MODE)

    return {"ok": True, "state": "reset", "send_return": ret}


def tcp_control_server(mc):
    global stop_flag

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((ROBOT_IP, TCP_CONTROL_PORT))
    server.listen(1)

    print(f"[TCP] control server listening: {ROBOT_IP}:{TCP_CONTROL_PORT}")

    while not stop_flag:
        conn, addr = server.accept()
        print("[TCP] connected:", addr)

        buffer = ""

        with conn:
            while not stop_flag:
                try:
                    data = conn.recv(4096)
                    if not data:
                        break

                    buffer += data.decode("utf-8", errors="ignore")

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)

                        if not line.strip():
                            continue

                        try:
                            command = json.loads(line)
                        except Exception as e:
                            response = {"ok": False, "state": "bad_json", "message": str(e)}
                            conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
                            continue

                        cmd_type = command.get("type")

                        if cmd_type == "aruco_height_step":
                            response = handle_height_step(mc, command)

                        elif cmd_type == "hold":
                            safe_stop(mc)
                            response = {"ok": True, "state": "hold"}

                        elif cmd_type == "reset":
                            response = handle_reset(mc)

                        else:
                            response = {"ok": False, "state": "unknown_command", "message": cmd_type}

                        conn.sendall((json.dumps(response) + "\n").encode("utf-8"))

                except Exception as e:
                    print("[TCP ERROR]", e)
                    break

    server.close()


def main():
    global stop_flag, current_z

    print("[INFO] ArUco Height Z-only Robot Side")
    print("[INFO] z coord controls height. x,y,rx,ry,rz are fixed -> camera pose never changes.")
    print(f"[INFO] FIXED pose: x={FIXED_X} y={FIXED_Y} rx={FIXED_RX} ry={FIXED_RY} rz={FIXED_RZ}")
    print(f"[INFO] Z range: [{MIN_Z}, {MAX_Z}], start={START_Z}, Z_SIGN={Z_SIGN}")

    mc = MyCobot280(PORT, BAUD)
    time.sleep(1.0)

    current_z = START_Z

    threading.Thread(target=send_udp_video, daemon=True).start()

    try:
        tcp_control_server(mc)
    finally:
        stop_flag = True
        safe_stop(mc)
        print("[ROBOT] 종료")


if __name__ == "__main__":
    main()