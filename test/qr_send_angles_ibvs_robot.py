# aruco_joint_servo_robot.py
# Robot side
# 목표:
# - RealSense RGB 영상을 PC로 UDP 송신
# - RealSense aligned depth/intrinsics는 로봇 쪽에서 유지
# - PC가 ArUco 중심/코너/rvec/tvec 정보를 보내면
#   move_j / move_l 없이 관절제어(send_angles 또는 jog_angle)만으로 정렬
# - 화면 중심 정렬 + ArUco 정면 자세 정렬 + 10cm 거리 정렬
#
# 권장 실행:
#   python3 aruco_joint_servo_robot.py
#
# 중요:
# - 처음에는 반드시 그리퍼/팔 주변을 비워두고 낮은 속도로 테스트하세요.
# - SIGN 값은 설치 방향에 따라 반대일 수 있습니다.
# - 기본값은 send_angles 기반입니다. jog_angle은 한 번에 여러 축 동시 제어가 어려운 환경이 많아 옵션으로 둡니다.

import cv2
import time
import json
import socket
import struct
import threading
import math
import numpy as np
import pyrealsense2 as rs
from pymycobot import MyCobot280


# ---------------------------------------------------------------------
# 네트워크 / 카메라 설정
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# ArUco / 목표값
# ---------------------------------------------------------------------
ARUCO_MARKER_SIZE_M = 0.025   # 사용 중인 ArUco 한 변 길이. 25mm이면 0.025
FINAL_STANDOFF_M = 0.10       # 카메라가 마커 앞 10cm에서 정렬
CENTER_TOL_PX = 12            # 화면 중심 허용 오차
DIST_TOL_M = 0.010            # 거리 허용 오차 1cm
ANGLE_TOL_DEG = 5.0           # 마커 정면 자세 허용 오차

# 목표 거리에서의 예상 마커 픽셀 크기 = marker_size * fx / z
SIZE_TOL_PX = 18.0


# ---------------------------------------------------------------------
# 제어 방식
# ---------------------------------------------------------------------
# "send_angles": 현재 관절각을 조금씩 갱신. 여러 관절 동시 보정 가능. 기본 추천.
# "jog_angle": 가장 큰 오차 축 하나를 속도 명령처럼 jog. 환경에 따라 stop 응답이 느릴 수 있음.
CONTROL_BACKEND = "send_angles"

# 프레임/명령 주기
# PC가 30fps로 보내도 로봇 쪽에서는 이 간격보다 빠르게 명령하지 않습니다.
# send_angles 기준 0.06~0.12초부터 테스트 권장.
CONTROL_INTERVAL_SEC = 0.08

# QR 미검출 시 바로 stop을 연속 호출하면 시리얼이 막히는 경우가 있어 제한합니다.
HOLD_INTERVAL_SEC = 0.35

# 관절 속도
SERVO_SPEED_MIN = 8
SERVO_SPEED_MAX = 35

# 중심에서 멀수록 속도/스텝 증가: 동심원 반경(px)
RING_1_PX = 25
RING_2_PX = 70
RING_3_PX = 140
RING_4_PX = 230

# send_angles 1회당 최대 보정량(deg)
STEP_RING_0_DEG = 0.10
STEP_RING_1_DEG = 0.25
STEP_RING_2_DEG = 0.55
STEP_RING_3_DEG = 0.90
STEP_RING_4_DEG = 1.25

# 거리/각도 보정 최대 스텝
MAX_DIST_STEP_DEG = 0.70
MAX_POSE_STEP_DEG = 0.70

# 너무 작은 스텝이면 떨림 방지
MIN_EFFECTIVE_STEP_DEG = 0.04

# 안전 관절 제한. 실제 설치자세에 맞게 좁혀도 됩니다.
SOFT_ANGLE_RANGES_DEG = [
    (-170.0, 170.0),
    (-135.0, 135.0),
    (-150.0, 150.0),
    (-170.0, 170.0),
    (-170.0, 170.0),
    (-180.0, 180.0),
]


# ---------------------------------------------------------------------
# 관절 매핑 / 부호
# ---------------------------------------------------------------------
# 이미지 중심 오차:
#   ex > 0 : 마커가 화면 오른쪽
#   ey > 0 : 마커가 화면 아래쪽
#
# 기본 가정:
#   ex -> J1 회전
#   ey -> J3 또는 J2/J3 조합
#   거리 -> J2/J3 조합
#   마커 정면 각도 -> J4/J5/J6 미세 보정
#
# 반대로 움직이면 해당 SIGN만 바꾸세요.
J1_FROM_EX_SIGN = -1.0
J3_FROM_EY_SIGN = +1.0

# 거리:
#   dist_error_m = 현재거리 - 목표거리
#   dist_error_m > 0 이면 멀다 -> 앞으로 접근해야 함
J2_FROM_DIST_SIGN = 0.0 #+1.0
J3_FROM_DIST_SIGN = 0.0 #-1.0

# 자세:
# PC에서 보낸 marker_euler_deg는 solvePnP rvec 기반 디버그용 값입니다.
# 보통 yaw/pitch/roll 부호는 설치 방향마다 달라서 반드시 테스트 후 조정하세요.
J5_FROM_PITCH_SIGN = -1.0
J6_FROM_YAW_SIGN = -1.0
J4_FROM_ROLL_SIGN = -0.5

# 픽셀/각도 게인
J1_DEG_PER_PX = 0.006
J3_DEG_PER_PX = 0.006
DIST_DEG_PER_M = 7.0         # 거리오차 0.1m당 약 0.7도
POSE_DEG_PER_DEG = 0.035     # 마커 각도 10도 오차당 0.35도 보정


# ---------------------------------------------------------------------
# 전역 상태
# ---------------------------------------------------------------------
stop_flag = False
robot_lock = threading.Lock()
depth_lock = threading.Lock()

latest_depth_frame = None
latest_depth_intrinsics = None
latest_depth_time = 0.0

approach_done = False
last_control_time = 0.0
last_hold_time = 0.0


def clamp(value, lo, hi):
    return max(lo, min(value, hi))


def valid_angles(angles):
    return isinstance(angles, list) and len(angles) >= 6


def limit_angles(angles):
    limited = list(angles[:6])
    for i in range(6):
        lo, hi = SOFT_ANGLE_RANGES_DEG[i]
        limited[i] = clamp(float(limited[i]), lo, hi)
    return limited


def safe_get_angles(mc):
    try:
        with robot_lock:
            return mc.get_angles()
    except Exception as e:
        print("[ROBOT WARN] get_angles failed:", e)
        return None


def safe_send_angles(mc, angles, speed):
    try:
        with robot_lock:
            return mc.send_angles(angles, int(speed))
    except Exception as e:
        print("[ROBOT WARN] send_angles failed:", e)
        return {"exception": str(e)}


def safe_send_angle(mc, joint_id, angle, speed):
    try:
        with robot_lock:
            return mc.send_angle(int(joint_id), float(angle), int(speed))
    except Exception as e:
        print("[ROBOT WARN] send_angle failed:", e)
        return {"exception": str(e)}


def safe_jog_angle(mc, joint_id, direction, speed):
    try:
        with robot_lock:
            return mc.jog_angle(int(joint_id), int(direction), int(speed))
    except Exception as e:
        print("[ROBOT WARN] jog_angle failed:", e)
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

    for _ in range(15):
        pipeline.wait_for_frames()

    print("[REALSENSE] RGB + Depth started")
    return pipeline, align


def send_udp_video():
    global stop_flag, latest_depth_frame, latest_depth_intrinsics, latest_depth_time

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

    try:
        while not stop_flag:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            with depth_lock:
                latest_depth_frame = depth_frame
                latest_depth_intrinsics = intrinsics
                latest_depth_time = time.time()

            color_image = np.asanyarray(color_frame.get_data())

            cv2.putText(
                color_image,
                f"RS frame: {color_frame.get_frame_number()}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
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
                print(f"[VIDEO] sent frame={frame_id}, chunks={total_chunks}")
                last_log_time = now

            frame_id = (frame_id + 1) % 4294967295

    finally:
        pipeline.stop()
        udp_sock.close()
        print("[VIDEO] UDP 송신 종료")


def median_depth(depth_frame, cx, cy, radius=4):
    values = []
    for y in range(int(cy) - radius, int(cy) + radius + 1):
        for x in range(int(cx) - radius, int(cx) + radius + 1):
            if x < 0 or y < 0 or x >= FRAME_WIDTH or y >= FRAME_HEIGHT:
                continue
            d = depth_frame.get_distance(x, y)
            if 0.04 <= d <= 2.0:
                values.append(d)

    if not values:
        return None

    return float(np.median(values))


def desired_marker_edge_px(intrinsics):
    try:
        return float(ARUCO_MARKER_SIZE_M * intrinsics.fx / FINAL_STANDOFF_M)
    except Exception:
        return None


def calc_ring_step_speed(pixel_dist):
    if pixel_dist < RING_1_PX:
        return STEP_RING_0_DEG, SERVO_SPEED_MIN, "R0"
    if pixel_dist < RING_2_PX:
        return STEP_RING_1_DEG, 12, "R1"
    if pixel_dist < RING_3_PX:
        return STEP_RING_2_DEG, 18, "R2"
    if pixel_dist < RING_4_PX:
        return STEP_RING_3_DEG, 25, "R3"
    return STEP_RING_4_DEG, SERVO_SPEED_MAX, "R4"


def build_joint_delta(command, depth_z_m, intrinsics):
    ex = float(command.get("ex", 0.0))
    ey = float(command.get("ey", 0.0))
    pixel_dist = math.sqrt(ex * ex + ey * ey)

    marker_edge_px = command.get("marker_edge_px")
    marker_euler = command.get("marker_euler_deg") or [0.0, 0.0, 0.0]

    try:
        roll_deg = float(marker_euler[0])
        pitch_deg = float(marker_euler[1])
        yaw_deg = float(marker_euler[2])
    except Exception:
        roll_deg, pitch_deg, yaw_deg = 0.0, 0.0, 0.0

    ring_step, speed, ring = calc_ring_step_speed(pixel_dist)

    # 중심 보정
    j1 = 0.0
    j3_center = 0.0

    if abs(ex) > CENTER_TOL_PX:
        raw = abs(ex) * J1_DEG_PER_PX
        j1 = J1_FROM_EX_SIGN * clamp(raw, MIN_EFFECTIVE_STEP_DEG, ring_step) * (1.0 if ex > 0 else -1.0)

    if abs(ey) > CENTER_TOL_PX:
        raw = abs(ey) * J3_DEG_PER_PX
        j3_center = J3_FROM_EY_SIGN * clamp(raw, MIN_EFFECTIVE_STEP_DEG, ring_step) * (1.0 if ey > 0 else -1.0)

    # 거리 보정: 1순위 depth, 보조로 marker edge 사용
    dist_error_m = None
    size_error_px = None
    j2_dist = 0.0
    j3_dist = 0.0

    if depth_z_m is not None:
        dist_error_m = float(depth_z_m) - FINAL_STANDOFF_M
    else:
        desired_edge = desired_marker_edge_px(intrinsics)
        if desired_edge and marker_edge_px:
            size_error_px = float(desired_edge) - float(marker_edge_px)
            # edge가 작다 = 멀다. 대략 거리오차처럼 사용
            dist_error_m = size_error_px / max(desired_edge, 1.0) * FINAL_STANDOFF_M

    # 중심이 어느 정도 맞은 뒤에만 전후 접근
    if dist_error_m is not None and pixel_dist < RING_3_PX and abs(dist_error_m) > DIST_TOL_M:
        dist_step = clamp(abs(dist_error_m) * DIST_DEG_PER_M, MIN_EFFECTIVE_STEP_DEG, MAX_DIST_STEP_DEG)
        direction = 1.0 if dist_error_m > 0 else -1.0
        j2_dist = J2_FROM_DIST_SIGN * dist_step * direction
        j3_dist = J3_FROM_DIST_SIGN * dist_step * direction

    # 정면 자세 보정: 중심이 심하게 틀어져 있으면 자세 보정은 약하게
    pose_gain_scale = 1.0 if pixel_dist < RING_2_PX else 0.35
    j4_pose = 0.0
    j5_pose = 0.0
    j6_pose = 0.0

    if abs(roll_deg) > ANGLE_TOL_DEG:
        j4_pose = J4_FROM_ROLL_SIGN * clamp(abs(roll_deg) * POSE_DEG_PER_DEG, MIN_EFFECTIVE_STEP_DEG, MAX_POSE_STEP_DEG) * (1.0 if roll_deg > 0 else -1.0) * pose_gain_scale

    if abs(pitch_deg) > ANGLE_TOL_DEG:
        j5_pose = J5_FROM_PITCH_SIGN * clamp(abs(pitch_deg) * POSE_DEG_PER_DEG, MIN_EFFECTIVE_STEP_DEG, MAX_POSE_STEP_DEG) * (1.0 if pitch_deg > 0 else -1.0) * pose_gain_scale

    if abs(yaw_deg) > ANGLE_TOL_DEG:
        j6_pose = J6_FROM_YAW_SIGN * clamp(abs(yaw_deg) * POSE_DEG_PER_DEG, MIN_EFFECTIVE_STEP_DEG, MAX_POSE_STEP_DEG) * (1.0 if yaw_deg > 0 else -1.0) * pose_gain_scale

    delta = [0.0] * 6
    delta[0] = j1
    delta[1] = j2_dist
    delta[2] = j3_center + j3_dist
    delta[3] = j4_pose
    delta[4] = j5_pose
    delta[5] = j6_pose

    debug = {
        "ex": ex,
        "ey": ey,
        "pixel_dist": pixel_dist,
        "ring": ring,
        "ring_step_deg": ring_step,
        "speed": speed,
        "depth_z_m": depth_z_m,
        "dist_error_m": dist_error_m,
        "size_error_px": size_error_px,
        "marker_edge_px": marker_edge_px,
        "marker_euler_deg": [roll_deg, pitch_deg, yaw_deg],
        "delta_deg": delta,
    }

    return delta, speed, debug


def is_aligned(command, depth_z_m, intrinsics):
    ex = float(command.get("ex", 0.0))
    ey = float(command.get("ey", 0.0))
    marker_euler = command.get("marker_euler_deg") or [999.0, 999.0, 999.0]
    marker_edge_px = command.get("marker_edge_px")

    center_ok = abs(ex) <= CENTER_TOL_PX and abs(ey) <= CENTER_TOL_PX

    dist_ok = False
    if depth_z_m is not None:
        dist_ok = abs(float(depth_z_m) - FINAL_STANDOFF_M) <= DIST_TOL_M
    else:
        desired_edge = desired_marker_edge_px(intrinsics)
        if desired_edge and marker_edge_px:
            dist_ok = abs(float(desired_edge) - float(marker_edge_px)) <= SIZE_TOL_PX

    try:
        roll, pitch, yaw = [abs(float(v)) for v in marker_euler[:3]]
        # roll까지 너무 강하게 보면 마커 회전 때문에 완료가 안 날 수 있어 pitch/yaw 중심으로 봅니다.
        angle_ok = pitch <= ANGLE_TOL_DEG and yaw <= ANGLE_TOL_DEG
    except Exception:
        angle_ok = False

    return center_ok and dist_ok and angle_ok, {
        "center_ok": center_ok,
        "dist_ok": dist_ok,
        "angle_ok": angle_ok,
    }


def handle_servo_align_once(mc, command):
    global approach_done, last_control_time, last_hold_time

    now = time.time()

    if approach_done:
        return {
            "ok": True,
            "state": "aligned_done",
            "message": "already aligned; press r on PC to reset",
            "control_method": CONTROL_BACKEND,
        }

    if now - last_control_time < CONTROL_INTERVAL_SEC:
        return {
            "ok": True,
            "state": "skip_interval",
            "message": "control interval guard",
            "control_method": CONTROL_BACKEND,
        }

    cx = int(command.get("cx", FRAME_WIDTH // 2))
    cy = int(command.get("cy", FRAME_HEIGHT // 2))

    with depth_lock:
        depth_frame = latest_depth_frame
        intrinsics = latest_depth_intrinsics

    if intrinsics is None:
        return {
            "ok": False,
            "state": "no_intrinsics",
            "message": "RealSense intrinsics not ready",
            "control_method": CONTROL_BACKEND,
        }

    depth_z_m = None
    if depth_frame is not None:
        depth_z_m = median_depth(depth_frame, cx, cy, radius=4)

    aligned, aligned_detail = is_aligned(command, depth_z_m, intrinsics)

    if aligned:
        safe_stop(mc)
        approach_done = True
        return {
            "ok": True,
            "state": "aligned_10cm_front",
            "message": "center + distance + front angle aligned",
            "distance_cm": None if depth_z_m is None else depth_z_m * 100.0,
            "aligned_detail": aligned_detail,
            "control_method": CONTROL_BACKEND,
        }

    current_angles = safe_get_angles(mc)
    if not valid_angles(current_angles):
        return {
            "ok": False,
            "state": "angle_read_fail",
            "message": "get_angles failed",
            "control_method": CONTROL_BACKEND,
        }

    delta, speed, debug = build_joint_delta(command, depth_z_m, intrinsics)
    max_abs_delta = max(abs(v) for v in delta)

    if max_abs_delta < MIN_EFFECTIVE_STEP_DEG:
        if now - last_hold_time >= HOLD_INTERVAL_SEC:
            safe_stop(mc)
            last_hold_time = now
        return {
            "ok": True,
            "state": "hold_small_delta",
            "message": "delta too small",
            "distance_cm": None if depth_z_m is None else depth_z_m * 100.0,
            "aligned_detail": aligned_detail,
            "debug": debug,
            "control_method": CONTROL_BACKEND,
        }

    if CONTROL_BACKEND == "jog_angle":
        # jog_angle은 보통 한 축씩 jog하는 API라, 가장 큰 delta 축만 실행합니다.
        # 여러 축 동시 정렬이 필요하면 send_angles가 더 안정적입니다.
        joint_index = int(np.argmax([abs(v) for v in delta]))
        direction = 1 if delta[joint_index] > 0 else 0
        ret = safe_jog_angle(mc, joint_index + 1, direction, speed)
        target_angles = current_angles
    else:
        target_angles = [float(current_angles[i]) + float(delta[i]) for i in range(6)]
        target_angles = limit_angles(target_angles)
        ret = safe_send_angles(mc, target_angles, speed)

    last_control_time = now

    return {
        "ok": True,
        "state": "servo_aligning",
        "message": "joint servo correction sent",
        "distance_cm": None if depth_z_m is None else depth_z_m * 100.0,
        "target_angles": target_angles,
        "current_angles": current_angles,
        "aligned_detail": aligned_detail,
        "debug": debug,
        "send_return": ret,
        "control_method": CONTROL_BACKEND,
    }


def handle_hold(mc):
    global last_hold_time
    now = time.time()
    if now - last_hold_time >= HOLD_INTERVAL_SEC:
        safe_stop(mc)
        last_hold_time = now
    return {"ok": True, "state": "hold", "message": "robot stopped"}


def handle_reset():
    global approach_done, last_control_time
    approach_done = False
    last_control_time = 0.0
    return {"ok": True, "state": "reset", "message": "alignment reset"}


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

                        if cmd_type == "aruco_servo_align":
                            response = handle_servo_align_once(mc, command)
                        elif cmd_type == "hold":
                            response = handle_hold(mc)
                        elif cmd_type == "reset_approach":
                            response = handle_reset()
                        else:
                            response = {"ok": False, "state": "unknown_command", "message": cmd_type}

                        conn.sendall((json.dumps(response) + "\n").encode("utf-8"))

                except Exception as e:
                    print("[TCP ERROR]", e)
                    break

    server.close()


def main():
    global stop_flag

    print("[INFO] ArUco Joint Servo Robot Side")
    print("[INFO] move_j/move_l 미사용, CONTROL_BACKEND =", CONTROL_BACKEND)

    mc = MyCobot280(PORT, BAUD)
    time.sleep(1.0)

    threading.Thread(target=send_udp_video, daemon=True).start()

    try:
        tcp_control_server(mc)
    finally:
        stop_flag = True
        safe_stop(mc)
        print("[ROBOT] 종료")


if __name__ == "__main__":
    main()