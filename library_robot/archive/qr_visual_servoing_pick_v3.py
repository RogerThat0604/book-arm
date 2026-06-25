#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qr_visual_servoing_pick_v3.py
v2 대비: RealSense depth를 추가해 '진짜 3D'로 책을 향해 팔을 뻗는다.

3축 분해
  - 방위각(좌우)  : base 회전 정렬          (v2와 동일)
  - 반경(앞뒤)    : depth로 잰 거리만큼 뻗기  (★ 신규)
  - 높이(상하)    : 책상 높이 상수 Z_PICK     (★ 신규)

★ 데모 전에 맞춰야 할 상수 2개 (아래 ▼CALIBRATE 섹션):
    D_TOUCH  : 카메라→그리퍼 끝 전방 오프셋(mm). 책에 닿았을 때의 depth 값
    Z_PICK   : 책을 잡는 책상 높이(mm). 팔을 책 위에 내려 잡는 높이의 z 좌표
  REACH_GAIN : 카메라 기울기 보정(≈0.8~1.0). 덜 뻗으면 ↑, 더 뻗으면 ↓
"""

import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from pymycobot import MyCobot280 as MyCobot

# ───────────────────────── 설정 ─────────────────────────
PORT = "/dev/ttyJETCOBOT"
BAUD = 1000000

# 정렬(방위각)
CENTER_THRESHOLD = 20
KP               = 0.05        # 오버슈트 줄이려 0.06->0.05
MAX_MOVE         = 6           # 진동 줄이려 8->6
DIRECTION        = -1
CONFIRM_FRAMES   = 2
SETTLE_SEC       = 0.8
ALIGN_TIMEOUT    = 35
MAX_ADJUST       = 25

# ▼─────────────── CALIBRATE: 여기 3개를 실측해 맞추세요 ───────────────▼
D_TOUCH    = 120.0   # mm. 카메라→그리퍼 끝 전방 거리. 자로 재거나 시행착오로
REACH_GAIN = 1.0     # 카메라 기울기 보정. 덜 뻗으면 키우고, 더 뻗으면 줄이기
Z_PICK     = 200.0   # mm. 책 잡는 책상 높이. 팔을 수동으로 내려 잡히는 z를 읽어 넣기
# ▲──────────────────────────────────────────────────────────────────▲

APPROACH_SPEED = 30
DESCEND_SPEED  = 20
COORD_MODE     = 1
DEPTH_WIN      = 7            # depth 샘플 윈도우(px). 노이즈/구멍 대비 median

FRAME_W, FRAME_H = 640, 480

_qr = cv2.QRCodeDetector()
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ───────────────────── 로봇 / 카메라 ─────────────────────
def connect_robot():
    mc = MyCobot(PORT, BAUD)
    mc.thread_lock = True
    mc.power_on()
    time.sleep(1)
    return mc


def start_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, 30)  # ★ depth 추가
    pipeline.start(config)
    align = rs.align(rs.stream.color)   # depth를 color 좌표계에 정렬
    print("RealSense 시작 성공 (color + depth)")
    time.sleep(1)
    return pipeline, align


def get_frames(pipeline, align):
    frames = align.process(pipeline.wait_for_frames())
    color = frames.get_color_frame()
    depth = frames.get_depth_frame()
    if not color or not depth:
        return None, None
    return np.asanyarray(color.get_data()), depth


# ───────────────────── QR 위치 + 깊이 ─────────────────────
def detect_qr(frame):
    """QR 위치만 검출. 반환: (dx, cx, cy) 또는 None"""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = _clahe.apply(gray)

    ok, pts = _qr.detect(gray)
    if not ok or pts is None:
        big = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        ok, pts = _qr.detect(big)
        if ok and pts is not None:
            pts = pts / 1.5
    if not ok or pts is None:
        return None

    p = pts.reshape(-1, 2)
    cx = float(np.mean(p[:, 0]))
    cy = float(np.mean(p[:, 1]))
    return cx - (w / 2.0), cx, cy


def sample_depth_mm(depth_frame, cx, cy):
    """QR 중심 주변 윈도우의 0이 아닌 depth 중앙값(mm)."""
    xs = int(round(cx)); ys = int(round(cy))
    half = DEPTH_WIN // 2
    vals = []
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            x = min(max(xs + dx, 0), FRAME_W - 1)
            y = min(max(ys + dy, 0), FRAME_H - 1)
            d = depth_frame.get_distance(x, y)   # meters
            if d > 0:
                vals.append(d * 1000.0)          # -> mm
    if not vals:
        return None
    return float(np.median(vals))


# ───────────────────── Visual Servoing(방위각) ─────────────────────
def align_to_qr(mc, pipeline, align):
    print("QR 중앙 정렬 시작")
    t0 = time.time(); last_seen = t0; warned = False
    confirm = 0; adjust = 0
    last_cxcy = None

    while time.time() - t0 < ALIGN_TIMEOUT and adjust < MAX_ADJUST:
        frame, depth = get_frames(pipeline, align)
        if frame is None:
            time.sleep(0.1); continue

        res = detect_qr(frame)
        if res is None:
            if not warned and time.time() - last_seen > 4:
                print("  ⚠ QR이 안 보임 — 화면 안/조명/크기 점검(qr_probe.png)")
                warned = True
            time.sleep(0.08); continue

        last_seen = time.time(); warned = False
        dx, cx, cy = res
        last_cxcy = (cx, cy, depth)
        print(f"  dx={dx:.1f}, cx={cx:.1f}")

        if abs(dx) < CENTER_THRESHOLD:
            confirm += 1
            if confirm >= CONFIRM_FRAMES:
                print("  방위각 정렬 완료")
                return last_cxcy          # (cx, cy, depth_frame) 반환
            time.sleep(0.12); continue
        confirm = 0

        current = mc.get_angles()
        if not current or len(current) < 6:
            print("  각도 읽기 실패:", current); return None

        move = round(dx * KP)
        move = max(min(move, MAX_MOVE), -MAX_MOVE)
        if move == 0:
            move = 1 if dx > 0 else -1
        current[0] = current[0] + (DIRECTION * move)
        print(f"    base -> {current[0]:.1f} (move={move}) [{adjust+1}/{MAX_ADJUST}]")
        mc.send_angles(current, 15)
        adjust += 1
        time.sleep(SETTLE_SEC)

    print("방위각 정렬 실패")
    return None


# ───────────────────── 3D 집기 ─────────────────────
def pick_3d(mc, cx, cy, depth_frame):
    """방위각 정렬 후: depth로 반경을 정해 팔을 뻗고, Z_PICK까지 내려 집는다."""
    depth_mm = sample_depth_mm(depth_frame, cx, cy)
    if depth_mm is None:
        print("depth 측정 실패 → 집기 취소"); return
    print(f"QR 거리 depth = {depth_mm:.1f} mm")

    coords = mc.get_coords()
    if not coords or len(coords) < 6:
        print("좌표 읽기 실패 → 집기 취소"); return
    x, y, z, rx, ry, rz = coords
    print(f"현재 좌표 {[round(c,1) for c in coords]}")

    # 현재 반경/방위 유지한 채, depth만큼 앞으로 더 뻗기
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    reach = REACH_GAIN * (depth_mm - D_TOUCH)
    target_r = r + reach
    tx = target_r * math.cos(theta)
    ty = target_r * math.sin(theta)
    print(f"반경 {r:.1f} -> {target_r:.1f} mm (앞으로 {reach:+.1f})")

    print("그리퍼 열기")
    mc.set_gripper_state(0, 50); time.sleep(1)

    # 1) 책 위로 뻗기(현재 높이 유지)
    print("책 위로 뻗기")
    mc.send_coords([tx, ty, z, rx, ry, rz], APPROACH_SPEED, COORD_MODE); time.sleep(3)

    # 2) 책상 높이까지 하강
    print(f"하강 (Z -> {Z_PICK})")
    mc.send_coords([tx, ty, Z_PICK, rx, ry, rz], DESCEND_SPEED, COORD_MODE); time.sleep(3)

    print("그리퍼 닫기")
    mc.set_gripper_state(1, 50); time.sleep(2)

    # 3) 들어올리기
    print("들어올리기")
    mc.send_coords([tx, ty, z, rx, ry, rz], DESCEND_SPEED, COORD_MODE); time.sleep(3)
    print("집기 완료")


# ───────────────────────── main ─────────────────────────
def main():
    mc = connect_robot()
    pipeline, align = start_realsense()
    try:
        res = align_to_qr(mc, pipeline, align)
        if res is not None:
            cx, cy, depth_frame = res
            print("정렬 성공 → 3D 집기 시작")
            pick_3d(mc, cx, cy, depth_frame)
        else:
            print("집기 취소")
    except KeyboardInterrupt:
        print("중단")
    finally:
        pipeline.stop()
        print("RealSense 종료")


if __name__ == "__main__":
    main()
