# qr_depth_movel_robot_refine_deadzone_debug.py
# Robot side
# 목표:
# - RealSense RGB 영상을 PC로 UDP 송신
# - RealSense aligned depth를 로봇 쪽에서 유지
# - PC가 QR 중심좌표(cx, cy)를 보내면 RealSense depth + intrinsics로 3D 오차 계산
# - J1/J4 카메라 각도 추적이 아니라 send_coords(MoveL)로 XYZ만 보정
# - 카메라 자세(Rx, Ry, Rz)는 현재 자세 또는 고정 자세를 유지
# - QR depth가 정상 검출되면 MoveL 이동 후 다시 QR 좌표를 확인하고, 데드존 밖이면 추가 보정

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

# MoveL 속도
# 안정화용으로 낮게 설정했습니다. 정렬이 안정되면 조금씩 올리세요.
MOVE_SPEED_FAST = 35
MOVE_SPEED_MID = 25
MOVE_SPEED_SLOW = 15

# 반복 보정 방식 설정
# QR depth가 정상 검출되면 현재 QR 3D 위치를 기준으로 목표 위치까지 MoveL 이동합니다.
# 이동 후 PC가 다시 QR 중심 좌표를 보내면, 데드존 안에 들어올 때까지 추가 보정합니다.
# 최종적으로 QR 앞에 남길 거리입니다.
FINAL_STANDOFF_M = 0.10

# 데드존 조건
# - 화면 중심 오차가 CENTER_TOL_PX 이내
# - 카메라 기준 QR Z거리가 FINAL_STANDOFF_M 근처이거나 그보다 가까우면 정렬 완료로 봅니다.
CENTER_TOL_PX = 18
RANGE_TOL_M = 0.015
DEPTH_TOL_M = RANGE_TOL_M

# 1회 보정 최대 이동량. 좌표계/측정 오류 방지를 위해 제한합니다.
MAX_APPROACH_MOVE_MM = 45.0
APPROACH_SPEED = 15
MAX_CORRECTION_COUNT = 8

# 한 번에 너무 크게 이동하지 않도록 제한
# QR 정렬에서는 큰 이동보다 작은 이동을 반복하는 편이 안정적입니다.
MAX_STEP_FAST_MM = 20.0
MAX_STEP_MID_MM = 10.0
MAX_STEP_SLOW_MM = 4.0

MIN_STEP_MM = 0.2

# 좌표축 매핑
# RealSense camera coordinate:
#   X: 화면 오른쪽 +
#   Y: 화면 아래쪽 +
#   Z: 카메라 앞쪽 +
#
# MyCobot coords:
#   coords[0] = X mm
#   coords[1] = Y mm
#   coords[2] = Z mm
#
# 기본 가정:
#   카메라 앞쪽(Z_cam) -> 로봇 X축
#   카메라 오른쪽(X_cam) -> 로봇 Y축
#   카메라 아래쪽(Y_cam) -> 로봇 Z축의 반대 방향
#
# 실제 방향이 반대면 SIGN만 바꾸세요.
CAM_X_TO_ROBOT_AXIS = 1
CAM_X_TO_ROBOT_SIGN = -1.0

CAM_Y_TO_ROBOT_AXIS = 2
CAM_Y_TO_ROBOT_SIGN = -1.0

CAM_Z_TO_ROBOT_AXIS = 0
CAM_Z_TO_ROBOT_SIGN = 1.0

# 자세 유지 방식
# True: 시작 시 현재 Rx,Ry,Rz를 계속 유지
# False: 매 이동 시 현재 자세를 그대로 사용
LOCK_START_ORIENTATION = True

stop_flag = False
linear_move_flag = False

depth_lock = threading.Lock()
robot_lock = threading.Lock()

# 로봇 명령 안정화
MOVE_COOLDOWN_SEC = 0.55
MOVE_SETTLE_SEC = 2.0
SERIAL_TIMEOUT_SEC = 0.25

# MyCobot/JetCobot에서 MoveL(mode=1)은 현재 자세/직선 경로가 조금만 불가능해도
# 명령이 들어가도 실제 이동이 안 되는 경우가 있습니다.
# 그래서 기본 이동은 mode=0(Joint interpolation)로 실행하고, 좌표 변화가 없으면 한 번 더 재시도합니다.
COORD_MOVE_MODE = 0
MOVE_RETRY_IF_UNCHANGED = True
MOVE_DETECT_THRESHOLD_MM = 1.0
TARGET_REACHED_THRESHOLD_MM = 4.0
PARTIAL_MOVE_THRESHOLD_MM = 8.0

# 디버깅용 소프트 작업공간 제한입니다.
# 이 범위를 벗어나면 JetCobot IK/작업공간 문제 가능성이 높다고 표시합니다.
# 실제 기구/초기 자세에 따라 조정하세요.
SOFT_X_RANGE_MM = (-120.0, 260.0)
SOFT_Y_RANGE_MM = (-220.0, 220.0)
SOFT_Z_RANGE_MM = (120.0, 430.0)

last_move_time = 0.0

latest_depth_frame = None
latest_depth_intrinsics = None
latest_depth_time = 0.0

locked_orientation = None
approach_done = False
correction_count = 0


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


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

    for _ in range(15):
        pipeline.wait_for_frames()

    print("[REALSENSE] RGB + Depth started, aligned to color")
    return pipeline, align


def median_depth(depth_frame, cx, cy, radius=5):
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


def deproject_qr_point(depth_frame, intrinsics, cx, cy, radius=5):
    distance_m = median_depth(depth_frame, cx, cy, radius)

    if distance_m is None:
        return None, None

    point_m = rs.rs2_deproject_pixel_to_point(
        intrinsics,
        [float(cx), float(cy)],
        distance_m
    )

    # point_m = [x, y, z] meter
    return point_m, distance_m


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

    print("[VIDEO] RealSense RGB UDP 송신 시작")

    try:
        while not stop_flag:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            with depth_lock:
                latest_depth_frame = depth_frame
                latest_depth_intrinsics = depth_intrinsics
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
                print(
                    f"[VIDEO] sent frame={frame_id}, rs={rs_frame_no}, "
                    f"chunks={total_chunks}, mean={color_image.mean():.1f}"
                )
                last_log_time = now

            frame_id = (frame_id + 1) % 4294967295

    finally:
        pipeline.stop()
        udp_sock.close()
        print("[VIDEO] UDP 송신 종료")



def camera_delta_to_robot_target(coords, delta_cam_m):
    """
    delta_cam_m: 카메라 좌표계 기준 이동 벡터[m]
      X: 화면 오른쪽 +
      Y: 화면 아래쪽 +
      Z: 카메라 전방 +

    QR을 향해 1회 접근할 때 사용합니다.
    """
    target = coords.copy()

    dx_mm = delta_cam_m[0] * 1000.0
    dy_mm = delta_cam_m[1] * 1000.0
    dz_mm = delta_cam_m[2] * 1000.0

    # 안전 제한: 한 번에 너무 많이 이동하지 않도록 제한
    dx_mm = clamp(dx_mm, -MAX_APPROACH_MOVE_MM, MAX_APPROACH_MOVE_MM)
    dy_mm = clamp(dy_mm, -MAX_APPROACH_MOVE_MM, MAX_APPROACH_MOVE_MM)
    dz_mm = clamp(dz_mm, -MAX_APPROACH_MOVE_MM, MAX_APPROACH_MOVE_MM)

    if abs(dx_mm) < MIN_STEP_MM:
        dx_mm = 0.0
    if abs(dy_mm) < MIN_STEP_MM:
        dy_mm = 0.0
    if abs(dz_mm) < MIN_STEP_MM:
        dz_mm = 0.0

    target[CAM_X_TO_ROBOT_AXIS] += CAM_X_TO_ROBOT_SIGN * dx_mm
    target[CAM_Y_TO_ROBOT_AXIS] += CAM_Y_TO_ROBOT_SIGN * dy_mm
    target[CAM_Z_TO_ROBOT_AXIS] += CAM_Z_TO_ROBOT_SIGN * dz_mm

    if LOCK_START_ORIENTATION and locked_orientation is not None:
        target[3] = locked_orientation[0]
        target[4] = locked_orientation[1]
        target[5] = locked_orientation[2]

    return target, (dx_mm, dy_mm, dz_mm)



def workspace_check(coords):
    """목표 좌표가 대략적인 작업 가능 범위 안인지 디버깅합니다."""
    if not valid_coords(coords):
        return {
            "ok": False,
            "reason": "invalid_coords",
            "x_ok": False,
            "y_ok": False,
            "z_ok": False,
        }

    x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
    x_ok = SOFT_X_RANGE_MM[0] <= x <= SOFT_X_RANGE_MM[1]
    y_ok = SOFT_Y_RANGE_MM[0] <= y <= SOFT_Y_RANGE_MM[1]
    z_ok = SOFT_Z_RANGE_MM[0] <= z <= SOFT_Z_RANGE_MM[1]

    reasons = []
    if not x_ok:
        reasons.append(f"X out {x:.1f}mm not in {SOFT_X_RANGE_MM}")
    if not y_ok:
        reasons.append(f"Y out {y:.1f}mm not in {SOFT_Y_RANGE_MM}")
    if not z_ok:
        reasons.append(f"Z out {z:.1f}mm not in {SOFT_Z_RANGE_MM}")

    return {
        "ok": x_ok and y_ok and z_ok,
        "reason": "ok" if not reasons else "; ".join(reasons),
        "x_ok": x_ok,
        "y_ok": y_ok,
        "z_ok": z_ok,
        "range": {
            "x": list(SOFT_X_RANGE_MM),
            "y": list(SOFT_Y_RANGE_MM),
            "z": list(SOFT_Z_RANGE_MM),
        },
    }


def distance_to_target_mm(coords, target):
    if not valid_coords(coords) or not valid_coords(target):
        return None
    return math.sqrt(
        (float(coords[0]) - float(target[0])) ** 2
        + (float(coords[1]) - float(target[1])) ** 2
        + (float(coords[2]) - float(target[2])) ** 2
    )


def classify_move(before, after, target, moved_mm, send_return, workspace):
    """명령 후 상태를 사람이 읽기 쉬운 상태 코드로 분류합니다."""
    before_error = distance_to_target_mm(before, target)
    after_error = distance_to_target_mm(after, target)

    if not valid_coords(before):
        return "coord_before_fail", "이동 전 좌표를 읽지 못했습니다.", before_error, after_error
    if not valid_coords(after):
        return "coord_after_fail", "이동 후 좌표를 읽지 못했습니다.", before_error, after_error
    if not workspace.get("ok", False):
        # 실제로 움직였더라도 목표가 소프트 범위 밖이면 별도 표시합니다.
        if moved_mm is not None and moved_mm >= MOVE_DETECT_THRESHOLD_MM:
            return "moved_but_target_suspicious", "목표 좌표가 소프트 작업범위 밖이지만 일부 이동했습니다.", before_error, after_error
        return "target_out_of_soft_workspace", "목표 좌표가 소프트 작업범위 밖이라 IK/작업공간 문제 가능성이 큽니다.", before_error, after_error
    if send_return is None:
        # pymycobot 계열은 성공 시 None을 반환하는 경우도 있어, 이것만으로 실패 판단하지 않습니다.
        pass
    if moved_mm is None:
        return "move_unknown", "좌표 변화량을 계산하지 못했습니다.", before_error, after_error
    if moved_mm < MOVE_DETECT_THRESHOLD_MM:
        return "command_sent_but_not_moved", "명령은 보냈지만 실제 좌표 변화가 거의 없습니다. IK 실패/서보 상태/명령 무시 가능성이 있습니다.", before_error, after_error
    if after_error is not None and after_error <= TARGET_REACHED_THRESHOLD_MM:
        return "target_reached", "목표 좌표에 거의 도달했습니다.", before_error, after_error
    if after_error is not None and before_error is not None and after_error < before_error:
        return "partial_move", "움직였지만 목표 좌표까지는 아직 남았습니다. 다음 QR 재검출 후 추가 보정합니다.", before_error, after_error
    return "moved_wrong_or_overshoot", "움직였지만 목표와 가까워지지 않았습니다. 좌표축 SIGN/축 매핑이 틀렸을 수 있습니다.", before_error, after_error

def safe_stop(mc):
    """mc.stop()이 시리얼 read에서 오래 막히는 것을 줄이기 위한 안전 정지."""
    try:
        with robot_lock:
            return mc.stop()
    except Exception as e:
        print("[ROBOT WARN] stop failed:", e)
        return None


def safe_get_coords(mc):
    try:
        with robot_lock:
            return mc.get_coords()
    except Exception as e:
        print("[ROBOT WARN] get_coords failed:", e)
        return None


def coord_delta_mm(a, b):
    if not valid_coords(a) or not valid_coords(b):
        return None
    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )


def safe_send_coords(mc, coords, speed, mode=COORD_MOVE_MODE):
    try:
        with robot_lock:
            ret = mc.send_coords(coords, speed, mode)
            print(f"[SEND_COORDS RETURN] mode={mode} speed={speed} ret={ret}")
            return ret
    except Exception as e:
        print("[ROBOT WARN] send_coords failed:", e)
        return {"exception": str(e)}


def execute_coord_move(mc, target, speed):
    """
    좌표 이동 실행 + 실제 좌표 변화 확인 + 디버깅 상태 분류.

    알 수 있는 것:
    - 목표 좌표가 소프트 작업공간 안인지
    - send_coords가 예외 없이 호출됐는지
    - 실제 좌표가 바뀌었는지
    - 목표 좌표 쪽으로 가까워졌는지

    알 수 없는 것:
    - pymycobot API가 내부 IK 해를 실제로 찾았는지 여부는 직접 반환하지 않는 경우가 많습니다.
      그래서 "명령은 보냈지만 좌표가 안 바뀜"이면 IK 실패/작업공간 문제/서보 상태 문제로 추정합니다.
    """
    before = safe_get_coords(mc)
    workspace = workspace_check(target)

    if not valid_coords(before):
        debug = {
            "phase": "coord_before_fail",
            "reason": "이동 전 현재 좌표를 읽지 못했습니다.",
            "workspace": workspace,
            "send_return": None,
            "retry_send_return": None,
            "before_error_mm": None,
            "after_error_mm": None,
        }
        return before, None, "coord_before_fail", None, debug

    print("=" * 70)
    print(f"[MOVE PLAN] target={target}")
    print(f"[MOVE PLAN] before={before}")
    print(f"[MOVE PLAN] target_delta_xyz_mm="
          f"[{target[0]-before[0]:+.1f}, {target[1]-before[1]:+.1f}, {target[2]-before[2]:+.1f}]")
    print(f"[WORKSPACE CHECK] ok={workspace['ok']} reason={workspace['reason']}")

    print(f"[MOVE CMD] mode={COORD_MOVE_MODE} speed={speed} target={target}")
    send_ret = safe_send_coords(mc, target, speed, COORD_MOVE_MODE)
    time.sleep(MOVE_SETTLE_SEC)

    after = safe_get_coords(mc)
    moved_mm = coord_delta_mm(before, after)
    phase, reason, before_error, after_error = classify_move(before, after, target, moved_mm, send_ret, workspace)

    retry_ret = None
    move_exec_state = "normal"

    if (
        MOVE_RETRY_IF_UNCHANGED
        and moved_mm is not None
        and moved_mm < MOVE_DETECT_THRESHOLD_MM
    ):
        retry_mode = 1 if COORD_MOVE_MODE == 0 else 0
        retry_speed = min(int(speed) + 10, 40)
        print(
            f"[MOVE RETRY] first move almost unchanged ({moved_mm:.2f}mm). "
            f"retry mode={retry_mode} speed={retry_speed}"
        )
        retry_ret = safe_send_coords(mc, target, retry_speed, retry_mode)
        time.sleep(MOVE_SETTLE_SEC)
        after_retry = safe_get_coords(mc)
        moved_retry_mm = coord_delta_mm(before, after_retry)
        phase, reason, before_error, after_error = classify_move(
            before, after_retry, target, moved_retry_mm, retry_ret, workspace
        )
        after = after_retry
        moved_mm = moved_retry_mm
        move_exec_state = "retry"

    print(f"[MOVE RESULT] phase={phase}")
    print(f"[MOVE RESULT] reason={reason}")
    print(f"[MOVE RESULT] after={after}")
    print(f"[MOVE RESULT] moved_delta_mm={moved_mm}")
    print(f"[MOVE RESULT] before_error_mm={before_error} after_error_mm={after_error}")
    print("=" * 70)

    debug = {
        "phase": phase,
        "reason": reason,
        "workspace": workspace,
        "send_return": send_ret,
        "retry_send_return": retry_ret,
        "before_error_mm": before_error,
        "after_error_mm": after_error,
    }

    return before, after, move_exec_state, moved_mm, debug


def handle_visual_approach_once(mc, command):
    """
    반복 보정 방식입니다.

    방식:
    1. PC가 보낸 QR 중심 픽셀(cx, cy)의 depth를 로봇 쪽 RealSense에서 읽음
    2. 픽셀+depth를 카메라 좌표계 3D point로 변환
    3. 목표 위치 [0, 0, FINAL_STANDOFF_M]와 현재 QR 위치의 차이를 계산
       delta_cam = [point_x, point_y, point_z - FINAL_STANDOFF_M]
    4. MoveL로 1회 이동
    5. 이동 완료 후 PC가 다시 QR 좌표를 보내면, 데드존 밖일 때만 추가 보정
    6. 데드존 안이면 stop 후 approach_done=True
    """
    global linear_move_flag, last_move_time, approach_done, correction_count

    now = time.time()
    if approach_done:
        print("[STATE] aligned_done: 이미 데드존 완료 상태입니다. r/reset 전까지 추가 이동하지 않습니다.")
        return {
            "ok": True,
            "state": "aligned_done",
            "message": "already inside deadzone",
            "debug_phase": "already_aligned_done",
            "debug_reason": "이미 데드존 완료 상태라 추가 이동하지 않습니다. PC에서 r을 누르면 reset 됩니다.",
        }

    if linear_move_flag or (now - last_move_time) < MOVE_COOLDOWN_SEC:
        remain = max(0.0, MOVE_COOLDOWN_SEC - (now - last_move_time))
        print(f"[STATE] busy: 이전 이동 안정화 대기 중 remain={remain:.2f}s")
        return {
            "ok": True,
            "state": "busy",
            "message": "previous move settling",
            "debug_phase": "busy_settling",
            "debug_reason": f"이전 이동 후 안정화 대기 중입니다. 남은 시간 약 {remain:.2f}s",
            "correction_count": correction_count,
            "debug_phase": "deadzone_ok",
            "debug_reason": "QR 중심 오차와 거리 오차가 모두 데드존 안이라 정지했습니다.",
            "deadzone": {
                "center_px": CENTER_TOL_PX,
                "range_tol_cm": RANGE_TOL_M * 100.0,
                "center_in_deadzone": center_in_deadzone,
                "depth_in_deadzone": depth_in_deadzone,
            },
        }

    cx = int(command.get("cx", FRAME_WIDTH // 2))
    cy = int(command.get("cy", FRAME_HEIGHT // 2))
    pixel_ex = float(command.get("ex", 0.0))
    pixel_ey = float(command.get("ey", 0.0))
    pixel_dist = math.sqrt(pixel_ex * pixel_ex + pixel_ey * pixel_ey)

    with depth_lock:
        depth_frame = latest_depth_frame
        intrinsics = latest_depth_intrinsics

        if depth_frame is None or intrinsics is None:
            print("[REFINE] no depth frame - RealSense depth not ready")
            return {"ok": False, "state": "no_depth", "message": "no depth frame", "debug_phase": "no_depth", "debug_reason": "로봇 쪽 RealSense depth frame이 아직 준비되지 않았습니다."}

        point_m, depth_z_m = deproject_qr_point(depth_frame, intrinsics, cx, cy, radius=5)

    if point_m is None:
        print(f"[REFINE] depth unavailable at QR center cx={cx}, cy={cy}")
        return {
            "ok": False,
            "state": "depth_unavailable",
            "message": "depth unavailable",
            "debug_phase": "depth_unavailable",
            "debug_reason": "QR 중심 픽셀 주변에서 유효한 depth 값을 얻지 못했습니다.",
            "cx": cx,
            "cy": cy,
            "correction_count": correction_count,
        }

    point_x_m = float(point_m[0])
    point_y_m = float(point_m[1])
    point_z_m = float(point_m[2])
    range_error_m = point_z_m - FINAL_STANDOFF_M

    center_in_deadzone = abs(pixel_ex) <= CENTER_TOL_PX and abs(pixel_ey) <= CENTER_TOL_PX
    depth_in_deadzone = abs(range_error_m) <= RANGE_TOL_M or point_z_m <= FINAL_STANDOFF_M + RANGE_TOL_M

    if center_in_deadzone and depth_in_deadzone:
        print(
            f"[STATE] deadzone_ok: center={center_in_deadzone} depth={depth_in_deadzone} "
            f"px=({pixel_ex:+.1f},{pixel_ey:+.1f}) depthZ={point_z_m*100:.1f}cm"
        )
        safe_stop(mc)
        approach_done = True
        print(
            f"[ALIGNED] px=({pixel_ex:+.1f},{pixel_ey:+.1f}) "
            f"depthZ={point_z_m * 100:.1f}cm final={FINAL_STANDOFF_M * 100:.1f}cm "
            f"count={correction_count}"
        )
        return {
            "ok": True,
            "state": "aligned_deadzone",
            "message": "QR is inside deadzone",
            "distance_m": point_z_m,
            "distance_cm": point_z_m * 100.0,
            "range_cm": point_z_m * 100.0,
            "point_m": point_m,
            "point_cm": [point_m[0] * 100.0, point_m[1] * 100.0, point_m[2] * 100.0],
            "pixel_dist": pixel_dist,
            "pixel_error": [pixel_ex, pixel_ey],
            "depth_error_cm": range_error_m * 100.0,
            "correction_count": correction_count,
        }

    if correction_count >= MAX_CORRECTION_COUNT:
        safe_stop(mc)
        approach_done = True
        print(
            f"[MAX CORRECTION STOP] px=({pixel_ex:+.1f},{pixel_ey:+.1f}) "
            f"depthZ={point_z_m * 100:.1f}cm count={correction_count}"
        )
        return {
            "ok": True,
            "state": "max_correction_stop",
            "message": "max correction count reached",
            "distance_m": point_z_m,
            "distance_cm": point_z_m * 100.0,
            "range_cm": point_z_m * 100.0,
            "point_m": point_m,
            "point_cm": [point_m[0] * 100.0, point_m[1] * 100.0, point_m[2] * 100.0],
            "pixel_dist": pixel_dist,
            "pixel_error": [pixel_ex, pixel_ey],
            "depth_error_cm": range_error_m * 100.0,
            "correction_count": correction_count,
            "debug_phase": "max_correction_stop",
            "debug_reason": "최대 보정 횟수에 도달해서 안전상 정지했습니다.",
        }

    # 목표: QR이 카메라 중심선 위에 있고, Z=FINAL_STANDOFF_M 위치에 있도록 카메라를 이동합니다.
    # 너무 가까운 상태에서는 전진 보정은 막고, 좌우/상하만 보정합니다.
    dz_m = range_error_m
    if point_z_m <= FINAL_STANDOFF_M + 0.003:
        dz_m = 0.0

    delta_cam_m = [point_x_m, point_y_m, dz_m]

    coords = safe_get_coords(mc)
    if not valid_coords(coords):
        return {"ok": False, "state": "coord_fail", "message": "get_coords failed"}

    target, step_mm = camera_delta_to_robot_target(coords, delta_cam_m)

    print(
        f"[STATE] need_correction: center_in_deadzone={center_in_deadzone} "
        f"depth_in_deadzone={depth_in_deadzone} "
        f"px=({pixel_ex:+.1f},{pixel_ey:+.1f}) "
        f"point_cm=({point_x_m*100:+.1f},{point_y_m*100:+.1f},{point_z_m*100:.1f}) "
        f"delta_cam_cm=({delta_cam_m[0]*100:+.1f},{delta_cam_m[1]*100:+.1f},{delta_cam_m[2]*100:+.1f})"
    )

    if step_mm == (0.0, 0.0, 0.0):
        safe_stop(mc)
        approach_done = True
        return {
            "ok": True,
            "state": "small_step_stop",
            "message": "step below minimum; treated as done",
            "distance_m": point_z_m,
            "distance_cm": point_z_m * 100.0,
            "point_cm": [point_m[0] * 100.0, point_m[1] * 100.0, point_m[2] * 100.0],
            "pixel_dist": pixel_dist,
            "depth_error_cm": range_error_m * 100.0,
            "correction_count": correction_count,
            "debug_phase": "small_step_stop",
            "debug_reason": "계산된 이동량이 최소 이동량보다 작아서 완료로 처리했습니다.",
        }

    correction_count += 1

    print(
        f"[REFINE MOVE] count={correction_count}/{MAX_CORRECTION_COUNT} "
        f"px=({pixel_ex:+.1f},{pixel_ey:+.1f}) distPx={pixel_dist:.1f} "
        f"depthZ={point_z_m * 100:.1f}cm final={FINAL_STANDOFF_M * 100:.1f}cm "
        f"errZ={range_error_m * 100:+.1f}cm step_mm={step_mm} speed={APPROACH_SPEED}"
    )

    linear_move_flag = True
    before_coords, after_coords, move_exec_state, moved_delta_mm, move_debug = execute_coord_move(mc, target, APPROACH_SPEED)
    last_move_time = time.time()
    linear_move_flag = False

    if moved_delta_mm is not None and moved_delta_mm < MOVE_DETECT_THRESHOLD_MM:
        print(
            f"[MOVE NOT STARTED] before={before_coords} after={after_coords} "
            f"target={target} moved={moved_delta_mm:.2f}mm"
        )

    return {
        "ok": True,
        "state": "correcting",
        "message": "moved once; send next QR position for deadzone check",
        "distance_m": point_z_m,
        "distance_cm": point_z_m * 100.0,
        "range_cm": point_z_m * 100.0,
        "point_m": point_m,
        "point_cm": [point_m[0] * 100.0, point_m[1] * 100.0, point_m[2] * 100.0],
        "pixel_dist": pixel_dist,
        "pixel_error": [pixel_ex, pixel_ey],
        "depth_error_cm": range_error_m * 100.0,
        "approach_cm": max(range_error_m, 0.0) * 100.0,
        "step_mm": step_mm,
        "step_cm": [step_mm[0] / 10.0, step_mm[1] / 10.0, step_mm[2] / 10.0],
        "speed": APPROACH_SPEED,
        "target_coords": target,
        "before_coords": before_coords,
        "after_coords": after_coords,
        "move_exec_state": move_exec_state,
        "moved_delta_mm": moved_delta_mm,
        "coord_move_mode": COORD_MOVE_MODE,
        "move_debug": move_debug,
        "debug_phase": move_debug.get("phase") if isinstance(move_debug, dict) else "unknown",
        "debug_reason": move_debug.get("reason") if isinstance(move_debug, dict) else "move debug unavailable",
        "correction_count": correction_count,
        "deadzone": {
            "center_px": CENTER_TOL_PX,
            "range_tol_cm": RANGE_TOL_M * 100.0,
            "center_in_deadzone": center_in_deadzone,
            "depth_in_deadzone": depth_in_deadzone,
        },
    }

def handle_command(mc, command):
    global stop_flag, linear_move_flag, approach_done, correction_count

    cmd_type = command.get("type")

    if cmd_type == "stop":
        print("[CONTROL] STOP")
        safe_stop(mc)
        stop_flag = True
        return {"ok": True, "message": "stopped"}

    if cmd_type == "hold":
        # 일반 hold는 응답성을 위해 stop을 호출하지 않습니다.
        # 실제 정지는 stop 명령 또는 1회 이동 후 내부 safe_stop에서 처리합니다.
        return {"ok": True, "message": "hold_ack_only"}

    if cmd_type == "reset_approach":
        approach_done = False
        correction_count = 0
        safe_stop(mc)
        print("[CONTROL] correction state reset")
        return {"ok": True, "state": "reset", "message": "correction reset"}

    if cmd_type == "status":
        return {
            "ok": True,
            "coords": safe_get_coords(mc),
            "has_depth": latest_depth_frame is not None,
            "depth_age": time.time() - latest_depth_time if latest_depth_time else None,
            "locked_orientation": locked_orientation,
            "approach_done": approach_done,
            "correction_count": correction_count,
        }

    if cmd_type == "visual_approach_once":
        return handle_visual_approach_once(mc, command)

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
    global stop_flag, locked_orientation

    print("[ROBOT] MyCobot 연결 중... refine_deadzone_debug")

    mc = MyCobot280(PORT, BAUD)
    mc.thread_lock = True
    try:
        mc._serial_port.timeout = SERIAL_TIMEOUT_SEC
    except Exception:
        pass

    time.sleep(1)
    mc.power_on()
    time.sleep(1)

    mc.set_fresh_mode(0)
    time.sleep(0.2)

    coords = safe_get_coords(mc)

    if not valid_coords(coords):
        print("[ROBOT] 좌표 읽기 실패")
        return

    if LOCK_START_ORIENTATION:
        locked_orientation = [coords[3], coords[4], coords[5]]

    print("[ROBOT] initial coords:", coords)
    print("[ROBOT] locked orientation:", locked_orientation)

    threading.Thread(target=send_udp_video, daemon=True).start()

    try:
        tcp_control_server(mc)

    except KeyboardInterrupt:
        print("\n[ROBOT] KeyboardInterrupt")

    finally:
        stop_flag = True
        safe_stop(mc)
        print("[ROBOT] final coords:", safe_get_coords(mc))
        print("[ROBOT] 종료")


if __name__ == "__main__":
    main()