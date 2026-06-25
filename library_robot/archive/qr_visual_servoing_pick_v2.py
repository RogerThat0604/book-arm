#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qr_visual_servoing_pick_v2.py
v1 대비 변경점 (검출률 문제 해결):
  - detectAndDecode() -> detect() 사용. 정렬에는 QR 위치만 필요(디코딩 불필요)
  - CLAHE 대비 보정 + 실패 시 1.5배 업스케일 재시도로 검출률↑
  - "QR 미검출" 프레임은 시도 횟수에서 차감 안 함 (timeout + 보정횟수로 제어)
  - QR을 오래 못 보면 경고 출력 (위치/조명 점검 안내)
"""

import time
import cv2
import numpy as np
import pyrealsense2 as rs
from pymycobot import MyCobot280 as MyCobot

# ───────────────────────── 설정 ─────────────────────────
PORT = "/dev/ttyJETCOBOT"
BAUD = 1000000

CENTER_THRESHOLD = 20
KP               = 0.06
MAX_MOVE         = 8
DIRECTION        = -1          # 발산하면 +1
CONFIRM_FRAMES   = 2
SETTLE_SEC       = 0.8

ALIGN_TIMEOUT    = 35          # 초. 전체 정렬 제한시간
MAX_ADJUST       = 25          # 실제 base 보정 횟수 상한(미검출은 제외)

DESCEND_Z_MM     = 60
COORD_SPEED      = 30
COORD_MODE       = 1

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
    pipeline.start(config)
    print("RealSense 시작 성공")
    time.sleep(1)
    return pipeline


def get_frame(pipeline):
    frames = pipeline.wait_for_frames()
    color = frames.get_color_frame()
    if not color:
        return None
    return np.asanyarray(color.get_data())


# ───────────────────── QR 위치 검출(디코딩 X) ─────────────────────
def detect_qr_dx(frame):
    """
    QR '위치'만 검출해 dx를 계산. 디코딩 실패와 무관하게 동작.
    반환: (dx, cx) 또는 None
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = _clahe.apply(gray)

    ok, pts = _qr.detect(gray)

    if not ok or pts is None:                 # 실패 시 1.5배 키워 재시도
        big = cv2.resize(gray, None, fx=1.5, fy=1.5,
                         interpolation=cv2.INTER_CUBIC)
        ok, pts = _qr.detect(big)
        if ok and pts is not None:
            pts = pts / 1.5

    if not ok or pts is None:
        return None

    p = pts.reshape(-1, 2)
    cx = float(np.mean(p[:, 0]))
    dx = cx - (w / 2.0)
    return dx, cx


# ───────────────────── Visual Servoing ─────────────────────
def align_to_qr(mc, pipeline):
    print("QR 중앙 정렬 시작")
    t0 = time.time()
    last_seen = t0
    warned = False
    confirm = 0
    adjust = 0

    while time.time() - t0 < ALIGN_TIMEOUT and adjust < MAX_ADJUST:
        frame = get_frame(pipeline)
        if frame is None:
            time.sleep(0.1)
            continue

        res = detect_qr_dx(frame)
        if res is None:
            # 미검출은 시도 횟수에 넣지 않음. 다만 오래 안 보이면 1회 경고
            if not warned and time.time() - last_seen > 4:
                print("  ⚠ QR이 4초째 안 보임 — 화면 안/조명/크기 점검(qr_probe.png)")
                warned = True
            time.sleep(0.08)
            continue

        last_seen = time.time()
        warned = False
        dx, cx = res
        print(f"  dx={dx:.1f}, cx={cx:.1f}")

        if abs(dx) < CENTER_THRESHOLD:
            confirm += 1
            if confirm >= CONFIRM_FRAMES:
                print("  정렬 완료(연속 확인)")
                return mc.get_angles()
            time.sleep(0.12)
            continue
        confirm = 0

        current = mc.get_angles()
        if not current or len(current) < 6:
            print("  각도 읽기 실패:", current)
            return None

        move = round(dx * KP)
        move = max(min(move, MAX_MOVE), -MAX_MOVE)
        if move == 0:
            move = 1 if dx > 0 else -1

        current[0] = current[0] + (DIRECTION * move)
        side = "왼쪽" if dx < 0 else "오른쪽"
        print(f"    base -> {current[0]:.1f} (move={move}, {side} 보정) [{adjust+1}/{MAX_ADJUST}]")

        mc.send_angles(current, 15)
        adjust += 1
        time.sleep(SETTLE_SEC)

    print("중앙 정렬 실패(시간 초과 또는 보정횟수 초과)")
    return None


# ───────────────────── 집기 시퀀스 ─────────────────────
def pick_at_current_pose(mc):
    print("그리퍼 열기")
    mc.set_gripper_state(0, 50)
    time.sleep(1)

    coords = mc.get_coords()
    if coords and len(coords) >= 6:
        print(f"현재 좌표 {[round(c,1) for c in coords]}")
        down = list(coords)
        down[2] -= DESCEND_Z_MM
        print(f"하강(Z {coords[2]:.1f} -> {down[2]:.1f})")
        mc.send_coords(down, COORD_SPEED, COORD_MODE)
        time.sleep(3)

        print("그리퍼 닫기")
        mc.set_gripper_state(1, 50)
        time.sleep(2)

        print("들어올리기")
        mc.send_coords(list(coords), COORD_SPEED, COORD_MODE)
        time.sleep(3)
    else:
        print("좌표 읽기 실패 → 관절각 폴백(base 보존)")
        cur = mc.get_angles()
        base = cur[0] if cur else 0
        above = [base, -20, -20, 0, 30, 0]
        pick  = [base, -35, -35, 0, 40, 0]
        lift  = [base, -15, -15, 0, 30, 0]
        mc.send_angles(above, 15); time.sleep(3)
        mc.send_angles(pick, 10);  time.sleep(3)
        mc.set_gripper_state(1, 50); time.sleep(2)
        mc.send_angles(lift, 10);  time.sleep(3)

    print("집기 완료")


# ───────────────────────── main ─────────────────────────
def main():
    mc = connect_robot()
    pipeline = start_realsense()
    try:
        aligned = align_to_qr(mc, pipeline)
        if aligned is not None:
            print("정렬 성공 → 집기 시작")
            pick_at_current_pose(mc)
        else:
            print("집기 취소")
    except KeyboardInterrupt:
        print("중단")
    finally:
        pipeline.stop()
        print("RealSense 종료")


if __name__ == "__main__":
    main()
