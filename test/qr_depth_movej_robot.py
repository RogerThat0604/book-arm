# qr_depth_ibvs_movej_robot_qr5cm_debug.py
# Robot side
# 목표:
# - RealSense RGB 영상을 PC로 UDP 송신
# - RealSense aligned depth를 로봇 쪽에서 유지
# - PC가 QR 중심좌표와 4개 코너를 보내면 QR 실제 크기 5cm + solvePnP로 pose 계산
# - IBVS + send_angles(MoveJ) 기반으로 이미지 오차를 관절각으로 직접 보정
# - 카메라 자세(Rx, Ry, Rz)는 현재 자세 또는 고정 자세를 유지
# - QR 5cm 픽셀 크기와 RealSense depth를 함께 디버그하고 이미지 오차 기반 IBVS 반복 보정

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

# QR 실제 크기 기반 Pose Estimation 설정
# 현재 사용하는 QR 코드 한 변이 5cm이므로 solvePnP의 object point 기준 크기를 0.05m로 둡니다.
QR_REAL_SIZE_M = 0.05
USE_QR_SIZE_POSE = True
PNP_DEPTH_WARN_DIFF_M = 0.08

# 데드존 조건
# - 화면 중심 오차가 CENTER_TOL_PX 이내
# - 카메라 기준 QR Z거리가 FINAL_STANDOFF_M 근처이거나 그보다 가까우면 정렬 완료로 봅니다.
CENTER_TOL_PX = 18
RANGE_TOL_M = 0.015
DEPTH_TOL_M = RANGE_TOL_M

# 1회 보정 최대 이동량. 좌표계/측정 오류 방지를 위해 제한합니다.
MAX_APPROACH_MOVE_MM = 45.0
APPROACH_SPEED = 15
MAX_CORRECTION_COUNT = 60

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
MOVE_COOLDOWN_SEC = 0.05
MOVE_SETTLE_SEC = 0.70
SERIAL_TIMEOUT_SEC = 0.25
BUSY_FORCE_RELEASE_SEC = 3.0  # 이 시간 이상 busy가 지속되면 안전하게 busy 플래그를 해제합니다.


# IBVS + MoveJ 제어 파라미터
# QR 중심 오차(ex/ey)와 QR 5cm 기준 픽셀 크기 오차를 관절각 보정량으로 변환합니다.
# 부호가 반대로 움직이면 SIGN 값만 바꾸면 됩니다.
IBVS_USE_SIZE_FOR_DISTANCE = True
MOVEJ_SPEED = 20
IBVS_SIZE_TOL_PX = 18.0
MOVE_DETECT_THRESHOLD_DEG = 0.25  # send_angles 후 실제 관절 변화 감지 최소값(deg)
DISTANCE_CONTROL_CENTER_GATE_PX = 90.0

J1_FROM_EX_SIGN = -1.0
J3_FROM_EY_SIGN = -1.0
J2_FROM_DIST_SIGN = 1.0
J3_FROM_DIST_SIGN = -1.0

J1_DEG_PER_PX = 0.010
J3_DEG_PER_PX = 0.010
MIN_ANGLE_STEP_DEG = 0.6
MAX_J1_STEP_DEG = 3.0
MAX_J3_STEP_DEG = 2.5
MAX_DIST_STEP_DEG = 2.0

# Joint soft limit. 실제 JetCobot 설치 자세에 따라 필요하면 좁히세요.
SOFT_ANGLE_RANGES_DEG = [
    (-170.0, 170.0),
    (-135.0, 135.0),
    (-150.0, 150.0),
    (-170.0, 170.0),
    (-170.0, 170.0),
    (-180.0, 180.0),
]

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
move_start_time = 0.0

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



def order_points_tl_tr_br_bl(points):
    """QR polygon 점들을 TL, TR, BR, BL 순서로 정렬합니다."""
    if points is None or len(points) < 4:
        return None

    pts = np.array(points[:4], dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    ordered = np.array([tl, tr, br, bl], dtype=np.float32)

    # 중복점이 생기면 정렬 실패로 봅니다.
    if len({(int(p[0]), int(p[1])) for p in ordered}) < 4:
        return None

    return ordered


def rotation_matrix_to_euler_deg(rotation_matrix):
    """회전행렬을 roll, pitch, yaw(deg)로 변환합니다. 디버그 표시용입니다."""
    sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = 0.0

    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def estimate_qr_pose_from_corners(intrinsics, pts, qr_size_m=QR_REAL_SIZE_M):
    """
    QR 실제 크기 5cm와 4개 코너 픽셀 좌표로 QR pose를 계산합니다.
    반환되는 tvec은 카메라 좌표계 기준 QR 중심 위치[m]입니다.
    """
    ordered = order_points_tl_tr_br_bl(pts)
    if ordered is None:
        return None

    half = qr_size_m / 2.0

    # TL, TR, BR, BL 순서. y축은 이미지와 비슷하게 아래쪽을 +로 둡니다.
    object_points = np.array([
        [-half, -half, 0.0],
        [ half, -half, 0.0],
        [ half,  half, 0.0],
        [-half,  half, 0.0],
    ], dtype=np.float32)

    camera_matrix = np.array([
        [intrinsics.fx, 0.0, intrinsics.ppx],
        [0.0, intrinsics.fy, intrinsics.ppy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # RealSense color stream이 이미 보정된 intrinsics를 제공한다고 보고 왜곡계수는 0으로 둡니다.
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            ordered.astype(np.float32),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except Exception as e:
        return {"ok": False, "reason": f"solvePnP exception: {e}"}

    if not ok:
        return {"ok": False, "reason": "solvePnP returned false"}

    rot, _ = cv2.Rodrigues(rvec)
    euler_deg = rotation_matrix_to_euler_deg(rot)

    edge_top = float(np.linalg.norm(ordered[1] - ordered[0]))
    edge_right = float(np.linalg.norm(ordered[2] - ordered[1]))
    edge_bottom = float(np.linalg.norm(ordered[2] - ordered[3]))
    edge_left = float(np.linalg.norm(ordered[3] - ordered[0]))
    mean_edge_px = (edge_top + edge_right + edge_bottom + edge_left) / 4.0
    approx_z_from_size_m = (qr_size_m * float(intrinsics.fx)) / mean_edge_px if mean_edge_px > 1.0 else None

    return {
        "ok": True,
        "ordered_pts": ordered.tolist(),
        "rvec": rvec.reshape(3).astype(float).tolist(),
        "tvec": tvec.reshape(3).astype(float).tolist(),
        "euler_deg": euler_deg,
        "edge_px": {
            "top": edge_top,
            "right": edge_right,
            "bottom": edge_bottom,
            "left": edge_left,
            "mean": mean_edge_px,
        },
        "approx_z_from_size_m": approx_z_from_size_m,
    }


def estimate_qr_point_with_size_and_depth(depth_frame, intrinsics, cx, cy, pts):
    """
    1순위: QR 5cm 실제 크기 + solvePnP로 QR 중심 3D 위치 계산
    보조: RealSense depth로 중심 Z 검증
    fallback: pose 실패 시 기존 depth 중심점 방식
    """
    depth_point_m, depth_z_m = deproject_qr_point(depth_frame, intrinsics, cx, cy, radius=5)
    pose = estimate_qr_pose_from_corners(intrinsics, pts, QR_REAL_SIZE_M) if USE_QR_SIZE_POSE and pts else None

    if pose and pose.get("ok"):
        pnp_point_m = [float(v) for v in pose["tvec"]]
        pnp_z_m = float(pnp_point_m[2])
        depth_diff_m = None
        depth_status = "no_depth_for_compare"

        if depth_z_m is not None:
            depth_diff_m = abs(float(depth_z_m) - pnp_z_m)
            if depth_diff_m <= PNP_DEPTH_WARN_DIFF_M:
                depth_status = "pnp_depth_consistent"
            else:
                depth_status = "pnp_depth_mismatch_warning"

        return pnp_point_m, pnp_z_m, {
            "source": "qr_size_5cm_solvepnp",
            "pose": pose,
            "depth_center_m": depth_z_m,
            "depth_point_m": depth_point_m,
            "pnp_z_m": pnp_z_m,
            "depth_diff_m": depth_diff_m,
            "depth_status": depth_status,
            "qr_real_size_cm": QR_REAL_SIZE_M * 100.0,
        }

    if depth_point_m is not None:
        return depth_point_m, depth_z_m, {
            "source": "depth_center_fallback",
            "pose": pose,
            "depth_center_m": depth_z_m,
            "depth_point_m": depth_point_m,
            "depth_status": "pnp_failed_depth_used",
            "qr_real_size_cm": QR_REAL_SIZE_M * 100.0,
        }

    return None, None, {
        "source": "no_pose_no_depth",
        "pose": pose,
        "depth_center_m": depth_z_m,
        "depth_status": "both_pose_and_depth_failed",
        "qr_real_size_cm": QR_REAL_SIZE_M * 100.0,
    }


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



# -----------------------------------------------------------------------------
# IBVS + MoveJ 기반 반복 보정부
# -----------------------------------------------------------------------------
# IBVS(Image Based Visual Servoing) 방식으로, 3D 목표 좌표를 직접 만들지 않습니다.
# 대신 이미지 특징량을 바로 관절 보정량으로 바꿉니다.
#
# 사용하는 이미지 특징량:
# - ex: QR 중심 x 오차(px)  -> J1(base yaw) 보정
# - ey: QR 중심 y 오차(px)  -> J3(elbow/tilt) 보정
# - QR edge mean(px)        -> 거리 보정. 5cm QR이 목표 거리 10cm에서 보일 픽셀 크기와 비교
#
# RealSense depth와 solvePnP pose는 계속 계산하지만, 이동 목표 생성에는 직접 쓰지 않고
# 신뢰도/디버깅/상태 표시용으로 유지합니다.

def angle_soft_limit_check(angles):
    if not valid_angles(angles):
        return {
            "ok": False,
            "reason": "invalid_angles",
            "joint_ok": [False] * 6,
            "ranges": SOFT_ANGLE_RANGES_DEG,
        }

    joint_ok = []
    reasons = []
    for i, angle in enumerate(angles[:6]):
        lo, hi = SOFT_ANGLE_RANGES_DEG[i]
        ok = lo <= float(angle) <= hi
        joint_ok.append(ok)
        if not ok:
            reasons.append(f"J{i+1} {float(angle):.2f}deg out of {lo}~{hi}")

    return {
        "ok": all(joint_ok),
        "reason": "ok" if not reasons else "; ".join(reasons),
        "joint_ok": joint_ok,
        "ranges": SOFT_ANGLE_RANGES_DEG,
    }


def valid_angles(angles):
    return isinstance(angles, list) and len(angles) >= 6


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
            ret = mc.send_angles(angles, speed)
            print(f"[SEND_ANGLES RETURN] speed={speed} ret={ret}")
            return ret
    except Exception as e:
        print("[ROBOT WARN] send_angles failed:", e)
        return {"exception": str(e)}


def angle_delta_norm_deg(a, b):
    if not valid_angles(a) or not valid_angles(b):
        return None
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(6)))


def distance_to_target_deg(angles, target):
    return angle_delta_norm_deg(angles, target)


def classify_joint_move(before, after, target, moved_deg, send_return, soft_limit):
    before_error = distance_to_target_deg(before, target)
    after_error = distance_to_target_deg(after, target)

    if not valid_angles(before):
        return "angle_before_fail", "이동 전 관절각을 읽지 못했습니다.", before_error, after_error
    if not valid_angles(after):
        return "angle_after_fail", "이동 후 관절각을 읽지 못했습니다.", before_error, after_error
    if not soft_limit.get("ok", False):
        if moved_deg is not None and moved_deg >= MOVE_DETECT_THRESHOLD_DEG:
            return "moved_but_target_angle_suspicious", "목표 관절각이 소프트 리미트 밖이지만 일부 이동했습니다.", before_error, after_error
        return "target_angle_out_of_soft_limit", "목표 관절각이 소프트 리미트 밖이라 명령 거부/위험 가능성이 큽니다.", before_error, after_error
    if moved_deg is None:
        return "move_unknown", "관절각 변화량을 계산하지 못했습니다.", before_error, after_error
    if moved_deg < MOVE_DETECT_THRESHOLD_DEG:
        return "command_sent_but_not_moved", "send_angles 명령은 보냈지만 실제 관절각 변화가 거의 없습니다. 서보 상태/전원/명령 무시 가능성이 있습니다.", before_error, after_error
    if after_error is not None and after_error <= TARGET_REACHED_THRESHOLD_DEG:
        return "target_reached", "목표 관절각에 거의 도달했습니다.", before_error, after_error
    if after_error is not None and before_error is not None and after_error < before_error:
        return "partial_move", "움직였지만 목표 관절각까지는 남았습니다. 다음 QR 재검출 후 추가 보정합니다.", before_error, after_error
    return "moved_wrong_or_overshoot", "움직였지만 목표 관절각과 가까워지지 않았습니다. SIGN/gain/관절 매핑이 틀렸을 수 있습니다.", before_error, after_error


def execute_joint_move(mc, target_angles, speed):
    before_angles = safe_get_angles(mc)
    before_coords = safe_get_coords(mc)
    soft_limit = angle_soft_limit_check(target_angles)

    if not valid_angles(before_angles):
        debug = {
            "phase": "angle_before_fail",
            "reason": "이동 전 현재 관절각을 읽지 못했습니다.",
            "soft_limit": soft_limit,
            "send_return": None,
            "before_error_deg": None,
            "after_error_deg": None,
        }
        return before_angles, None, before_coords, None, "angle_before_fail", None, debug

    print("=" * 70)
    print(f"[MOVEJ PLAN] target_angles={target_angles}")
    print(f"[MOVEJ PLAN] before_angles={before_angles}")
    print(f"[MOVEJ PLAN] before_coords={before_coords}")
    print("[MOVEJ PLAN] target_delta_deg=" + str([round(target_angles[i] - before_angles[i], 3) for i in range(6)]))
    print(f"[ANGLE LIMIT CHECK] ok={soft_limit['ok']} reason={soft_limit['reason']}")

    send_ret = safe_send_angles(mc, target_angles, speed)
    time.sleep(MOVE_SETTLE_SEC)

    after_angles = safe_get_angles(mc)
    after_coords = safe_get_coords(mc)
    moved_deg = angle_delta_norm_deg(before_angles, after_angles)
    phase, reason, before_error, after_error = classify_joint_move(
        before_angles, after_angles, target_angles, moved_deg, send_ret, soft_limit
    )

    print(f"[MOVEJ RESULT] phase={phase}")
    print(f"[MOVEJ RESULT] reason={reason}")
    print(f"[MOVEJ RESULT] after_angles={after_angles}")
    print(f"[MOVEJ RESULT] after_coords={after_coords}")
    print(f"[MOVEJ RESULT] moved_delta_deg={moved_deg}")
    print(f"[MOVEJ RESULT] before_error_deg={before_error} after_error_deg={after_error}")
    print("=" * 70)

    debug = {
        "phase": phase,
        "reason": reason,
        "soft_limit": soft_limit,
        "send_return": send_ret,
        "before_error_deg": before_error,
        "after_error_deg": after_error,
    }

    return before_angles, after_angles, before_coords, after_coords, "movej", moved_deg, debug


def get_qr_edge_mean_px(edge_px):
    if isinstance(edge_px, dict):
        value = edge_px.get("mean", None)
        if value is not None:
            return float(value)
    return None


def get_desired_qr_edge_px(intrinsics):
    """
    5cm QR이 목표 거리 FINAL_STANDOFF_M에서 보여야 하는 평균 한 변 픽셀 크기입니다.
    pinhole model: pixel_size = real_size * fx / z
    """
    try:
        fx = float(intrinsics.fx)
        return (QR_REAL_SIZE_M * fx) / FINAL_STANDOFF_M
    except Exception:
        return None


def build_ibvs_movej_target(current_angles, pixel_ex, pixel_ey, qr_edge_mean_px, desired_edge_px):
    """
    IBVS + MoveJ 관절 목표 생성.

    좌표계 목표 X/Y/Z를 만들지 않고 이미지 오차를 직접 관절 보정량으로 변환합니다.

    - ex: 화면 중심보다 QR이 오른쪽이면 J1을 조금 회전
    - ey: 화면 중심보다 QR이 아래쪽이면 J3를 조금 회전
    - size_error_px: 목표 QR 픽셀 크기보다 작으면 멀다고 보고 J2/J3로 접근
    """
    target = current_angles.copy()
    center_abs = math.sqrt(pixel_ex * pixel_ex + pixel_ey * pixel_ey)

    j1_step = 0.0
    j3_center_step = 0.0
    j2_size_step = 0.0
    j3_size_step = 0.0
    size_error_px = None
    size_control_enabled = False
    desired_edge_valid = desired_edge_px is not None and desired_edge_px > 1.0
    qr_edge_valid = qr_edge_mean_px is not None and qr_edge_mean_px > 1.0

    if abs(pixel_ex) > CENTER_TOL_PX:
        j1_step = J1_FROM_EX_SIGN * clamp(abs(pixel_ex) * J1_DEG_PER_PX, MIN_ANGLE_STEP_DEG, MAX_J1_STEP_DEG) * (1.0 if pixel_ex > 0 else -1.0)

    if abs(pixel_ey) > CENTER_TOL_PX:
        j3_center_step = J3_FROM_EY_SIGN * clamp(abs(pixel_ey) * J3_DEG_PER_PX, MIN_ANGLE_STEP_DEG, MAX_J3_STEP_DEG) * (1.0 if pixel_ey > 0 else -1.0)

    # 중심이 너무 틀어진 상태에서는 거리/크기 보정을 하지 않습니다.
    # 먼저 화면 중앙으로 가져온 뒤 QR 크기로 접근/후퇴를 보정합니다.
    if IBVS_USE_SIZE_FOR_DISTANCE and center_abs <= DISTANCE_CONTROL_CENTER_GATE_PX and desired_edge_valid and qr_edge_valid:
        size_error_px = desired_edge_px - qr_edge_mean_px
        if abs(size_error_px) > IBVS_SIZE_TOL_PX:
            # size_error_px > 0: QR이 목표보다 작음 = 멀다 = 접근
            # size_error_px < 0: QR이 목표보다 큼 = 가깝다 = 후퇴
            # 기존 DIST_DEG_PER_CM 대신 픽셀 크기 오차를 각도 스텝으로 변환합니다.
            base_step = clamp(abs(size_error_px) * 0.018, MIN_ANGLE_STEP_DEG, MAX_DIST_STEP_DEG)
            direction = 1.0 if size_error_px > 0 else -1.0
            j2_size_step = J2_FROM_DIST_SIGN * base_step * direction
            j3_size_step = J3_FROM_DIST_SIGN * base_step * direction
            size_control_enabled = True

    target[0] += j1_step
    target[1] += j2_size_step
    target[2] += j3_center_step + j3_size_step

    for i in range(6):
        lo, hi = SOFT_ANGLE_RANGES_DEG[i]
        target[i] = clamp(float(target[i]), lo, hi)

    debug = {
        "mode": "ibvs_movej_center_and_qr_size",
        "ibvs_features": {
            "ex_px": pixel_ex,
            "ey_px": pixel_ey,
            "center_abs_px": center_abs,
            "qr_edge_mean_px": qr_edge_mean_px,
            "desired_edge_px": desired_edge_px,
            "size_error_px": size_error_px,
        },
        "j1_step_deg": j1_step,
        "j2_step_deg": j2_size_step,
        "j3_center_step_deg": j3_center_step,
        "j3_size_step_deg": j3_size_step,
        "j3_total_step_deg": j3_center_step + j3_size_step,
        "size_control_enabled": size_control_enabled,
        "size_control_gate_px": DISTANCE_CONTROL_CENTER_GATE_PX,
        "size_tol_px": IBVS_SIZE_TOL_PX,
        "signs": {
            "J1_FROM_EX_SIGN": J1_FROM_EX_SIGN,
            "J3_FROM_EY_SIGN": J3_FROM_EY_SIGN,
            "J2_FROM_DIST_SIGN": J2_FROM_DIST_SIGN,
            "J3_FROM_DIST_SIGN": J3_FROM_DIST_SIGN,
        },
        "gains": {
            "J1_DEG_PER_PX": J1_DEG_PER_PX,
            "J3_DEG_PER_PX": J3_DEG_PER_PX,
            "SIZE_ERR_PX_TO_DEG": 0.018,
        },
    }

    return target, debug


def handle_visual_approach_once(mc, command):
    """
    IBVS + MoveJ 반복 보정 방식입니다.

    방식:
    1. PC가 보낸 QR 중심 픽셀(cx, cy)과 4개 코너 pts를 받음
    2. QR 실제 크기 5cm + RealSense intrinsics로 pose/depth/QR 크기 디버그 계산
    3. 제어는 3D 목표 좌표가 아니라 이미지 특징량(ex, ey, QR edge size)으로 수행
    4. ex/ey는 J1/J3 관절 보정, QR 픽셀 크기는 J2/J3 접근/후퇴 보정
    5. send_angles(MoveJ)로 1회 이동
    6. 이동 완료 후 PC가 다시 QR 좌표를 보내면, 데드존 밖일 때만 추가 보정
    7. 데드존 안이면 stop 후 approach_done=True
    """
    global linear_move_flag, last_move_time, move_start_time, approach_done, correction_count

    now = time.time()
    if approach_done:
        print("[STATE] aligned_done: 이미 데드존 완료 상태입니다. r/reset 전까지 추가 이동하지 않습니다.")
        return {
            "ok": True,
            "state": "aligned_done",
            "message": "already inside deadzone",
            "debug_phase": "already_aligned_done",
            "debug_reason": "이미 데드존 완료 상태라 추가 이동하지 않습니다. PC에서 r을 누르면 reset 됩니다.",
            "qr_real_size_cm": QR_REAL_SIZE_M * 100.0,
            "control_method": "IBVS + MoveJ/send_angles",
        }

    if linear_move_flag:
        busy_elapsed = now - move_start_time if move_start_time else 0.0
        if busy_elapsed > BUSY_FORCE_RELEASE_SEC:
            print(f"[STATE WARN] busy 플래그가 {busy_elapsed:.2f}s 이상 지속되어 강제 해제합니다.")
            linear_move_flag = False
        else:
            print(f"[STATE] busy: 이전 MoveJ 명령 실행 중 elapsed={busy_elapsed:.2f}s")
            return {
                "ok": True,
                "state": "busy",
                "message": "previous move executing",
                "debug_phase": "busy_executing",
                "debug_reason": f"이전 MoveJ 명령이 아직 실행 함수 안에 있습니다. 경과 시간 {busy_elapsed:.2f}s",
                "correction_count": correction_count,
                "settle_elapsed_sec": busy_elapsed,
                "settle_remain_sec": None,
                "qr_real_size_cm": QR_REAL_SIZE_M * 100.0,
                "control_method": "IBVS + MoveJ/send_angles",
            }

    cooldown_remain = MOVE_COOLDOWN_SEC - (now - last_move_time)
    if cooldown_remain > 0:
        print(f"[STATE] cooldown: 이전 MoveJ 후 짧은 안정화 대기 remain={cooldown_remain:.2f}s")
        return {
            "ok": True,
            "state": "busy",
            "message": "previous move cooldown",
            "debug_phase": "busy_cooldown",
            "debug_reason": f"이전 MoveJ 이동 후 짧은 안정화 대기 중입니다. 남은 시간 약 {cooldown_remain:.2f}s",
            "correction_count": correction_count,
            "settle_elapsed_sec": now - last_move_time,
            "settle_remain_sec": cooldown_remain,
            "qr_real_size_cm": QR_REAL_SIZE_M * 100.0,
            "control_method": "IBVS + MoveJ/send_angles",
        }

    cx = int(command.get("cx", FRAME_WIDTH // 2))
    cy = int(command.get("cy", FRAME_HEIGHT // 2))
    pixel_ex = float(command.get("ex", 0.0))
    pixel_ey = float(command.get("ey", 0.0))
    pts = command.get("pts", None)
    pixel_dist = math.sqrt(pixel_ex * pixel_ex + pixel_ey * pixel_ey)

    with depth_lock:
        depth_frame = latest_depth_frame
        intrinsics = latest_depth_intrinsics

        if depth_frame is None or intrinsics is None:
            print("[POSE] no depth frame - RealSense depth not ready")
            return {
                "ok": False,
                "state": "no_depth",
                "message": "no depth frame",
                "debug_phase": "no_depth",
                "debug_reason": "로봇 쪽 RealSense depth frame/intrinsics가 아직 준비되지 않았습니다.",
                "qr_real_size_cm": QR_REAL_SIZE_M * 100.0,
                "control_method": "IBVS + MoveJ/send_angles",
            }

        point_m, pose_z_m, pose_debug = estimate_qr_point_with_size_and_depth(depth_frame, intrinsics, cx, cy, pts)

    if point_m is None:
        print(f"[POSE FAIL] cx={cx}, cy={cy}, pts={pts}, debug={pose_debug}")
        return {
            "ok": False,
            "state": "pose_depth_unavailable",
            "message": "QR 5cm pose and depth unavailable",
            "debug_phase": "pose_depth_unavailable",
            "debug_reason": "QR 4코너 solvePnP와 중심 depth 둘 다 유효한 3D 위치를 만들지 못했습니다.",
            "pose_debug": pose_debug,
            "cx": cx,
            "cy": cy,
            "correction_count": correction_count,
            "qr_real_size_cm": QR_REAL_SIZE_M * 100.0,
            "control_method": "IBVS + MoveJ/send_angles",
        }

    point_x_m = float(point_m[0])
    point_y_m = float(point_m[1])
    point_z_m = float(point_m[2])
    range_error_m = point_z_m - FINAL_STANDOFF_M

    center_in_deadzone = abs(pixel_ex) <= CENTER_TOL_PX and abs(pixel_ey) <= CENTER_TOL_PX
    # IBVS 거리 데드존은 QR의 실제 크기 5cm에서 계산한 목표 edge pixel과 현재 edge pixel 차이로 판단합니다.
    # depth_in_deadzone은 기존 디버그 호환을 위해 size/depth 둘 중 하나가 만족하면 True로 둡니다.
    depth_in_deadzone = abs(range_error_m) <= RANGE_TOL_M or point_z_m <= FINAL_STANDOFF_M + RANGE_TOL_M

    pose_source = pose_debug.get("source", "unknown") if isinstance(pose_debug, dict) else "unknown"
    depth_status = pose_debug.get("depth_status", "unknown") if isinstance(pose_debug, dict) else "unknown"
    euler_deg = None
    edge_px = None
    approx_z_from_size_m = None
    if isinstance(pose_debug, dict) and isinstance(pose_debug.get("pose"), dict) and pose_debug["pose"].get("ok"):
        euler_deg = pose_debug["pose"].get("euler_deg")
        edge_px = pose_debug["pose"].get("edge_px")
        approx_z_from_size_m = pose_debug["pose"].get("approx_z_from_size_m")

    qr_edge_mean_px = get_qr_edge_mean_px(edge_px)
    desired_edge_px = get_desired_qr_edge_px(intrinsics)
    size_error_px = None
    if desired_edge_px is not None and qr_edge_mean_px is not None:
        size_error_px = desired_edge_px - qr_edge_mean_px
    size_in_deadzone = size_error_px is not None and abs(size_error_px) <= IBVS_SIZE_TOL_PX
    # 최종 정렬 완료 판정은 중심 데드존 + QR 픽셀 크기 데드존입니다.
    # depth_in_deadzone은 화면/로그 참고용으로 계속 같이 보냅니다.
    ibvs_in_deadzone = center_in_deadzone and size_in_deadzone

    print(
        f"[QR 5CM DEBUG + IBVS MOVEJ] source={pose_source} depth_status={depth_status} "
        f"point_cm=({point_x_m*100:+.1f},{point_y_m*100:+.1f},{point_z_m*100:.1f}) "
        f"depth_center_cm={(pose_debug.get('depth_center_m') or 0)*100:.1f} "
        f"pnp_z_cm={(pose_debug.get('pnp_z_m') or 0)*100:.1f} "
        f"size_z_cm={(approx_z_from_size_m or 0)*100:.1f} "
        f"euler={euler_deg} edge_px={edge_px} "
        f"desired_edge_px={(desired_edge_px or 0):.1f} size_error_px={(size_error_px or 0):+.1f}"
    )

    if ibvs_in_deadzone:
        print(
            f"[STATE] ibvs_deadzone_ok: center={center_in_deadzone} size={size_in_deadzone} depth_ref={depth_in_deadzone} "
            f"px=({pixel_ex:+.1f},{pixel_ey:+.1f}) edge={qr_edge_mean_px} desired={desired_edge_px} poseZ={point_z_m*100:.1f}cm"
        )
        safe_stop(mc)
        approach_done = True
        return {
            "ok": True,
            "state": "aligned_deadzone",
            "message": "QR is inside deadzone",
            "distance_m": point_z_m,
            "distance_cm": point_z_m * 100.0,
            "range_cm": point_z_m * 100.0,
            "point_m": point_m,
            "point_cm": [point_x_m * 100.0, point_y_m * 100.0, point_z_m * 100.0],
            "pixel_dist": pixel_dist,
            "pixel_error": [pixel_ex, pixel_ey],
            "depth_error_cm": range_error_m * 100.0,
            "pose_source": pose_source,
            "pose_debug": pose_debug,
            "qr_euler_deg": euler_deg,
            "qr_edge_px": edge_px,
            "qr_size_distance_cm": approx_z_from_size_m * 100.0 if approx_z_from_size_m else None,
            "qr_real_size_cm": QR_REAL_SIZE_M * 100.0,
            "correction_count": correction_count,
            "debug_phase": "ibvs_deadzone_ok",
            "debug_reason": "QR 중심 오차와 5cm QR 픽셀 크기 오차가 모두 데드존 안이라 정지했습니다.",
            "control_method": "IBVS + MoveJ/send_angles",
            "ibvs": {
                "qr_edge_mean_px": qr_edge_mean_px,
                "desired_edge_px": desired_edge_px,
                "size_error_px": size_error_px,
                "size_tol_px": IBVS_SIZE_TOL_PX,
            },
            "deadzone": {
                "center_px": CENTER_TOL_PX,
                "range_tol_cm": RANGE_TOL_M * 100.0,
                "size_tol_px": IBVS_SIZE_TOL_PX,
                "center_in_deadzone": center_in_deadzone,
                "size_in_deadzone": size_in_deadzone,
                "depth_in_deadzone": depth_in_deadzone,
            },
        }

    if correction_count >= MAX_CORRECTION_COUNT:
        safe_stop(mc)
        approach_done = True
        print(
            f"[MAX CORRECTION STOP] px=({pixel_ex:+.1f},{pixel_ey:+.1f}) "
            f"poseZ={point_z_m * 100:.1f}cm count={correction_count}/{MAX_CORRECTION_COUNT} "
            f"center_deadzone={center_in_deadzone} depth_deadzone={depth_in_deadzone}"
        )
        return {
            "ok": True,
            "state": "max_correction_stop",
            "message": "max correction count reached; correction paused for safety",
            "distance_m": point_z_m,
            "distance_cm": point_z_m * 100.0,
            "range_cm": point_z_m * 100.0,
            "point_m": point_m,
            "point_cm": [point_x_m * 100.0, point_y_m * 100.0, point_z_m * 100.0],
            "pixel_dist": pixel_dist,
            "pixel_error": [pixel_ex, pixel_ey],
            "depth_error_cm": range_error_m * 100.0,
            "pose_source": pose_source,
            "pose_debug": pose_debug,
            "qr_euler_deg": euler_deg,
            "qr_edge_px": edge_px,
            "qr_real_size_cm": QR_REAL_SIZE_M * 100.0,
            "correction_count": correction_count,
            "debug_phase": "max_correction_stop",
            "debug_reason": "최대 보정 횟수에 도달해서 안전상 정지했습니다. MoveJ는 작은 각도 보정을 반복하므로 r 키로 재시작하거나 MAX_CORRECTION_COUNT를 더 늘릴 수 있습니다.",
            "control_method": "IBVS + MoveJ/send_angles",
        }

    current_angles = safe_get_angles(mc)
    if not valid_angles(current_angles):
        return {
            "ok": False,
            "state": "angle_fail",
            "message": "get_angles failed",
            "debug_phase": "angle_fail",
            "debug_reason": "현재 관절각을 읽지 못해서 MoveJ 목표를 만들 수 없습니다.",
            "control_method": "IBVS + MoveJ/send_angles",
        }

    target_angles, joint_plan_debug = build_ibvs_movej_target(current_angles, pixel_ex, pixel_ey, qr_edge_mean_px, desired_edge_px)

    # 너무 작은 변화면 완료 또는 대기 처리합니다.
    planned_delta = angle_delta_norm_deg(current_angles, target_angles)
    if planned_delta is not None and planned_delta < MIN_ANGLE_STEP_DEG:
        safe_stop(mc)
        approach_done = True
        return {
            "ok": True,
            "state": "small_angle_step_stop",
            "message": "angle step below minimum; treated as done",
            "distance_m": point_z_m,
            "distance_cm": point_z_m * 100.0,
            "point_cm": [point_x_m * 100.0, point_y_m * 100.0, point_z_m * 100.0],
            "pixel_dist": pixel_dist,
            "depth_error_cm": range_error_m * 100.0,
            "pose_source": pose_source,
            "pose_debug": pose_debug,
            "qr_real_size_cm": QR_REAL_SIZE_M * 100.0,
            "correction_count": correction_count,
            "joint_plan_debug": joint_plan_debug,
            "debug_phase": "small_angle_step_stop",
            "debug_reason": "계산된 관절 이동량이 최소 이동량보다 작아서 완료로 처리했습니다.",
            "control_method": "IBVS + MoveJ/send_angles",
        }

    correction_count += 1

    print(
        f"[IBVS MOVEJ REFINE] count={correction_count}/{MAX_CORRECTION_COUNT} "
        f"px=({pixel_ex:+.1f},{pixel_ey:+.1f}) distPx={pixel_dist:.1f} "
        f"poseZ={point_z_m * 100:.1f}cm final={FINAL_STANDOFF_M * 100:.1f}cm "
        f"errZ_ref={range_error_m * 100:+.1f}cm edge={qr_edge_mean_px} desiredEdge={desired_edge_px} sizeErr={size_error_px} target_angles={target_angles} "
        f"joint_plan={joint_plan_debug} speed={MOVEJ_SPEED}"
    )

    linear_move_flag = True
    move_start_time = time.time()
    try:
        before_angles, after_angles, before_coords, after_coords, move_exec_state, moved_delta_deg, move_debug = execute_joint_move(
            mc, target_angles, MOVEJ_SPEED
        )
    finally:
        last_move_time = time.time()
        linear_move_flag = False

    if moved_delta_deg is not None and moved_delta_deg < MOVE_DETECT_THRESHOLD_DEG:
        print(
            f"[MOVEJ NOT STARTED] before_angles={before_angles} after_angles={after_angles} "
            f"target_angles={target_angles} moved={moved_delta_deg:.2f}deg"
        )

    return {
        "ok": True,
        "state": "correcting_movej",
        "message": "MoveJ moved once; send next QR position for deadzone check",
        "distance_m": point_z_m,
        "distance_cm": point_z_m * 100.0,
        "range_cm": point_z_m * 100.0,
        "point_m": point_m,
        "point_cm": [point_x_m * 100.0, point_y_m * 100.0, point_z_m * 100.0],
        "pixel_dist": pixel_dist,
        "pixel_error": [pixel_ex, pixel_ey],
        "depth_error_cm": range_error_m * 100.0,
        "approach_cm": max(range_error_m, 0.0) * 100.0,
        "pose_source": pose_source,
        "pose_debug": pose_debug,
        "qr_euler_deg": euler_deg,
        "qr_edge_px": edge_px,
        "qr_size_distance_cm": approx_z_from_size_m * 100.0 if approx_z_from_size_m else None,
        "qr_real_size_cm": QR_REAL_SIZE_M * 100.0,
        "ibvs": {
            "qr_edge_mean_px": qr_edge_mean_px,
            "desired_edge_px": desired_edge_px,
            "size_error_px": size_error_px,
            "size_tol_px": IBVS_SIZE_TOL_PX,
        },
        "control_method": "IBVS + MoveJ/send_angles",
        "target_angles": target_angles,
        "before_angles": before_angles,
        "after_angles": after_angles,
        "before_coords": before_coords,
        "after_coords": after_coords,
        "moved_delta_deg": moved_delta_deg,
        "move_exec_state": move_exec_state,
        "joint_plan_debug": joint_plan_debug,
        "speed": MOVEJ_SPEED,
        "move_debug": move_debug,
        "debug_phase": move_debug.get("phase") if isinstance(move_debug, dict) else "unknown",
        "debug_reason": move_debug.get("reason") if isinstance(move_debug, dict) else "move debug unavailable",
        "correction_count": correction_count,
        "deadzone": {
            "center_px": CENTER_TOL_PX,
            "range_tol_cm": RANGE_TOL_M * 100.0,
            "center_in_deadzone": center_in_deadzone,
            "size_in_deadzone": size_in_deadzone if 'size_in_deadzone' in locals() else None,
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
            "angles": safe_get_angles(mc),
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

    print("[ROBOT] MyCobot 연결 중... movej_qr5cm_pose_debug")

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