# qr_depth_robot_control_realsense.py
# Robot side
# - RealSense RGB color frame UDP 송신
# - RealSense aligned depth frame 보관
# - TCP로 joint_delta / hold / resume / get_depth_at / move_forward_mm 처리
# - 기존 MyCobot joint motion loop 구조 유지

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

SPEED = 70

J1_MIN, J1_MAX = -90.0, 90.0
J4_MIN, J4_MAX = -120.0, 120.0

MOTION_HZ = 25.0
MAX_STEP_PER_TICK_J1 = 2.0
MAX_STEP_PER_TICK_J4 = 2.0
ANGLE_EPS = 0.05

MOVE_SPEED = 45

# QR 정렬 후 전진 축.
# 반대로 움직이면 FORWARD_SIGN = -1.0 으로 변경하세요.
FORWARD_AXIS = 0
FORWARD_SIGN = 1.0

stop_flag = False
hold_flag = False
linear_move_flag = False

robot_lock = threading.Lock()
depth_lock = threading.Lock()

current_target_angles = None
desired_angles = None

latest_depth_frame = None
latest_depth_time = 0.0


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def valid_angles(angles):
    return isinstance(angles, list) and len(angles) >= 6


def valid_coords(coords):
    return isinstance(coords, list) and len(coords) >= 6


def create_realsense():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)

    profile = pipeline.start(config)

    device = profile.get_device()
    print("[REALSENSE] device:", device.get_info(rs.camera_info.name))
    print("[REALSENSE] serial:", device.get_info(rs.camera_info.serial_number))

    depth_sensor = device.first_depth_sensor()

    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)

    if depth_sensor.supports(rs.option.laser_power):
        laser_range = depth_sensor.get_option_range(rs.option.laser_power)
        depth_sensor.set_option(rs.option.laser_power, laser_range.max)

    align = rs.align(rs.stream.color)

    # 자동 노출 안정화
    for _ in range(15):
        pipeline.wait_for_frames()

    print("[REALSENSE] RGB + Depth stream started")
    return pipeline, align


def get_depth_median(depth_frame, cx, cy, radius=5):
    if depth_frame is None:
        return None

    values = []

    for y in range(int(cy) - radius, int(cy) + radius + 1):
        for x in range(int(cx) - radius, int(cx) + radius + 1):
            if x < 0 or y < 0 or x >= FRAME_WIDTH or y >= FRAME_HEIGHT:
                continue

            d = depth_frame.get_distance(x, y)

            # 0은 측정 실패로 보는 편이 안전합니다.
            if 0.05 <= d <= 2.0:
                values.append(d)

    if not values:
        return None

    return float(np.median(values))


def send_udp_video():
    global stop_flag, latest_depth_frame, latest_depth_time

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
    pc_addr = (PC_IP, UDP_VIDEO_PORT)

    try:
        pipeline, align = create_realsense()
    except Exception as e:
        print("[REALSENSE] start failed:", e)
        stop_flag = True
        return

    frame_id = 0
    last_log_time = 0.0

    print("[VIDEO] RealSense RGB UDP 송신 시작")

    try:
        while not stop_flag:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            with depth_lock:
                latest_depth_frame = depth_frame
                latest_depth_time = time.time()

            color_image = np.asanyarray(color_frame.get_data())

            rs_frame_no = color_frame.get_frame_number()

            cv2.putText(
                color_image,
                f"RS frame: {rs_frame_no}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
            )

            ok, buffer = cv2.imencode(
                ".jpg",
                color_image,
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

            now = time.time()
            if now - last_log_time >= 2.0:
                print(
                    f"[VIDEO] sent frame={frame_id}, rs={rs_frame_no}, "
                    f"chunks={total_chunks}, mean={color_image.mean():.1f}"
                )
                last_log_time = now

            frame_id = (frame_id + 1) % 4294967295

    except Exception as e:
        print("[VIDEO ERROR]", e)

    finally:
        pipeline.stop()
        udp_sock.close()
        print("[VIDEO] RealSense RGB UDP 송신 종료")


def motion_loop(mc):
    global stop_flag, hold_flag, linear_move_flag
    global current_target_angles, desired_angles

    interval = 1.0 / MOTION_HZ

    print("[MOTION] 부드러운 joint 추종 루프 시작")

    while not stop_flag:
        start = time.time()

        if hold_flag or linear_move_flag:
            time.sleep(interval)
            continue

        with robot_lock:
            if valid_angles(current_target_angles) and valid_angles(desired_angles):
                next_angles = current_target_angles.copy()

                diff_j1 = desired_angles[0] - current_target_angles[0]
                diff_j4 = desired_angles[3] - current_target_angles[3]

                step_j1 = clamp(diff_j1, -MAX_STEP_PER_TICK_J1, MAX_STEP_PER_TICK_J1)
                step_j4 = clamp(diff_j4, -MAX_STEP_PER_TICK_J4, MAX_STEP_PER_TICK_J4)

                if abs(step_j1) >= ANGLE_EPS or abs(step_j4) >= ANGLE_EPS:
                    next_angles[0] = clamp(current_target_angles[0] + step_j1, J1_MIN, J1_MAX)
                    next_angles[3] = clamp(current_target_angles[3] + step_j4, J4_MIN, J4_MAX)

                    current_target_angles = next_angles.copy()
                    mc.send_angles(next_angles, SPEED)

        sleep_time = interval - (time.time() - start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("[MOTION] 종료")


def refresh_current_angles(mc):
    global current_target_angles, desired_angles

    real = mc.get_angles()

    if valid_angles(real):
        current_target_angles = real.copy()
        desired_angles = real.copy()
        return True, real

    return False, None


def handle_command(mc, command):
    global stop_flag, hold_flag, linear_move_flag
    global current_target_angles, desired_angles
    global latest_depth_frame, latest_depth_time

    cmd_type = command.get("type")

    if cmd_type == "stop":
        print("[CONTROL] STOP")
        mc.stop()
        stop_flag = True
        return {"ok": True, "message": "stopped"}

    if cmd_type == "hold":
        mc.stop()
        hold_flag = True

        with robot_lock:
            ok, real = refresh_current_angles(mc)

        if not ok:
            return {"ok": False, "message": "hold: get_angles failed"}

        return {"ok": True, "message": "hold", "angles": real}

    if cmd_type == "resume":
        hold_flag = False
        return {"ok": True, "message": "resume"}

    if cmd_type == "refresh_angles":
        with robot_lock:
            ok, real = refresh_current_angles(mc)

        if not ok:
            return {"ok": False, "message": "get_angles failed"}

        print("[CONTROL] angles refreshed:", real)
        return {"ok": True, "angles": real}

    if cmd_type == "status":
        return {
            "ok": True,
            "angles": mc.get_angles(),
            "coords": mc.get_coords(),
            "hold_flag": hold_flag,
            "linear_move_flag": linear_move_flag,
            "current_target_angles": current_target_angles,
            "desired_angles": desired_angles,
            "has_depth": latest_depth_frame is not None,
            "depth_age": time.time() - latest_depth_time if latest_depth_time else None,
        }

    if cmd_type == "get_depth_at":
        cx = int(command.get("cx", FRAME_WIDTH // 2))
        cy = int(command.get("cy", FRAME_HEIGHT // 2))
        radius = int(command.get("radius", 5))

        with depth_lock:
            depth_frame = latest_depth_frame
            depth_age = time.time() - latest_depth_time if latest_depth_time else None
            distance = get_depth_median(depth_frame, cx, cy, radius)

        if distance is None:
            return {
                "ok": False,
                "message": "depth unavailable",
                "cx": cx,
                "cy": cy,
                "depth_age": depth_age,
            }

        print(f"[DEPTH] ({cx}, {cy}) = {distance:.3f} m")

        return {
            "ok": True,
            "cx": cx,
            "cy": cy,
            "distance_m": distance,
            "depth_age": depth_age,
        }

    if cmd_type == "joint_delta":
        dj1 = float(command.get("dj1", 0.0))
        dj4 = float(command.get("dj4", 0.0))
        reason = command.get("reason", "")

        if linear_move_flag:
            return {"ok": False, "message": "linear move running"}

        hold_flag = False

        with robot_lock:
            if not valid_angles(desired_angles):
                ok, _ = refresh_current_angles(mc)
                if not ok:
                    return {"ok": False, "message": "get_angles failed"}

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
                "desired_j4": desired_angles[3],
            }

    if cmd_type == "move_forward_mm":
        move_mm = float(command.get("move_mm", 0.0))
        speed = int(command.get("speed", MOVE_SPEED))
        max_move_mm = float(command.get("max_move_mm", 250.0))

        move_mm = clamp(move_mm, -max_move_mm, max_move_mm)

        print(f"[CONTROL] MOVE_FORWARD_MM: {move_mm:.1f} mm")

        linear_move_flag = True
        hold_flag = True
        mc.stop()
        time.sleep(0.15)

        coords = mc.get_coords()

        if not valid_coords(coords):
            linear_move_flag = False
            return {"ok": False, "message": "get_coords failed"}

        target = coords.copy()
        target[FORWARD_AXIS] += FORWARD_SIGN * move_mm

        print("[CONTROL] current coords:", coords)
        print("[CONTROL] target coords :", target)

        mc.send_coords(target, speed, 1)

        wait_time = clamp(abs(move_mm) / 70.0, 1.0, 4.0)
        time.sleep(wait_time)

        mc.stop()
        time.sleep(0.2)

        with robot_lock:
            refresh_current_angles(mc)

        linear_move_flag = False

        return {
            "ok": True,
            "message": "linear move done",
            "move_mm": move_mm,
            "target_coords": target,
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


if __name__ == "__main__":ᅁᅠ
    main()