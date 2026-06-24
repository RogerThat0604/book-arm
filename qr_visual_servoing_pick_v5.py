#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qr_visual_servoing_pick_v5.py

v4 대비 정렬 안정화:
  - 검출 검증: 화면 밖 좌표 / 비정상 크기 사각형 버림 (cx=-39 같은 쓰레기 차단)
  - look-then-move: 멈춘 상태에서 여러 프레임 median으로 안정된 dx 확보 후 1회 보정
  - 이상치 제거: 직전 추정과 너무 동떨어진 읽기 무시
  - QR 분실 시 복구: 마지막 좋은 방향으로 되돌려 재탐색
집기/depth 게이트는 v4와 동일(티칭 관절각 + soft 게이트).
"""

import time
import cv2
import numpy as np
import pyrealsense2 as rs
from pymycobot import MyCobot280 as MyCobot

PORT = "/dev/ttyJETCOBOT"
BAUD = 1000000

# 정렬(look-then-move)
CENTER_THRESHOLD = 25
KP               = 0.045
MAX_MOVE         = 5          # 한 번에 크게 돌려 QR이 프레임 밖으로 나가지 않게
DIRECTION        = -1
CONFIRM_FRAMES   = 2
SETTLE_SEC       = 1.2        # 보낸 뒤 완전히 멈출 때까지 대기(움직이며 읽지 않기)
MAX_ADJUST       = 30
SAMPLES_PER_READ = 10         # 한 번 결정할 때 모을 프레임 수
MIN_VALID        = 3          # 이만큼 유효 검출돼야 신뢰
OUTLIER_PX       = 120        # 직전 median과 이만큼 벗어나면 이상치로 버림
LOST_LIMIT       = 3          # 연속 분실 횟수 → 복구 동작
RECOVER_DEG      = 6          # 복구 시 되돌릴 각도

# depth 게이트 (v4와 동일)
DEPTH_MIN, DEPTH_MAX = 120.0, 450.0
GATE_MODE   = "soft"          # soft / hard / off
DEPTH_FRAMES = 8
DEPTH_RADIUS = 45

# ▼ TEACH: 실측한 잡기 자세(관절각). j0(base)는 정렬값으로 덮임 ▼
ABOVE_J = [0, -20, -20, 0, 30, 0]
PICK_J  = [0, -35, -35, 0, 40, 0]
LIFT_J  = [0, -15, -15, 0, 30, 0]
# ▲────────────────────────────────────────────────────────────▲

FRAME_W, FRAME_H = 640, 480
AREA_MIN = 200                # px^2. 너무 작은 사각형 = 노이즈
AREA_MAX = 0.7 * FRAME_W * FRAME_H   # 너무 큰 것 = 오검출

_qr = cv2.QRCodeDetector()
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


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
    config.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
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


def detect_qr_validated(frame):
    """
    검증된 QR 검출만 반환. 반환: (dx, cx, cy) 또는 None
    - 화면 밖 좌표 / 비정상 크기 / 비볼록 사각형은 버림
    """
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

    p = pts.reshape(-1, 2).astype(np.float32)
    cx = float(np.mean(p[:, 0]))
    cy = float(np.mean(p[:, 1]))

    # ── 검증 ──
    if not (0 <= cx < w and 0 <= cy < h):
        return None                              # 화면 밖 → 쓰레기
    area = cv2.contourArea(p)
    if not (AREA_MIN < area < AREA_MAX):
        return None                              # 크기 비정상
    if not cv2.isContourConvex(p.astype(np.int32)):
        return None                              # 깨진 사각형

    return cx - (w / 2.0), cx, cy


def stable_read(pipeline, align, prev_dx=None):
    """
    멈춘 상태에서 여러 프레임을 모아 검증·이상치 제거 후 median dx.
    반환: (dx, cx, cy) 또는 None
    """
    dxs, cxs, cys = [], [], []
    for _ in range(SAMPLES_PER_READ):
        frame, _ = get_frames(pipeline, align)
        if frame is None:
            continue
        res = detect_qr_validated(frame)
        if res is None:
            continue
        dx, cx, cy = res
        if prev_dx is not None and abs(dx - prev_dx) > OUTLIER_PX and dxs:
            continue                             # 직전과 너무 다르면 이상치
        dxs.append(dx); cxs.append(cx); cys.append(cy)
        time.sleep(0.02)

    if len(dxs) < MIN_VALID:
        return None
    return float(np.median(dxs)), float(np.median(cxs)), float(np.median(cys))


def align_to_qr(mc, pipeline, align):
    print("QR 방위각 정렬 시작 (look-then-move)")
    cur = mc.get_angles()
    if not cur or len(cur) < 6:
        print("각도 읽기 실패"); return None, None
    confirm = 0; lost = 0; prev_dx = None; last_cxcy = None
    last_good_dir = 1

    for i in range(MAX_ADJUST):
        read = stable_read(pipeline, align, prev_dx)

        if read is None:
            lost += 1
            print(f"  QR 안정 검출 실패 ({lost}/{LOST_LIMIT})")
            if lost >= LOST_LIMIT:
                # 복구: 마지막으로 움직인 반대 방향으로 살짝 되돌려 재탐색
                cur[0] = cur[0] - last_good_dir * RECOVER_DEG
                print(f"  복구: base -> {cur[0]:.1f} 로 되돌려 재탐색")
                mc.send_angles(cur, 15); time.sleep(SETTLE_SEC)
                lost = 0; prev_dx = None
            continue

        lost = 0
        dx, cx, cy = read
        last_cxcy = (cx, cy)
        prev_dx = dx
        print(f"  [{i+1}] dx={dx:.1f} (median, cx={cx:.1f})")

        if abs(dx) < CENTER_THRESHOLD:
            confirm += 1
            if confirm >= CONFIRM_FRAMES:
                print("  방위각 정렬 완료")
                return mc.get_angles(), last_cxcy
            continue
        confirm = 0

        move = round(dx * KP)
        move = max(min(move, MAX_MOVE), -MAX_MOVE)
        if move == 0:
            move = 1 if dx > 0 else -1
        last_good_dir = 1 if (DIRECTION * move) > 0 else -1
        cur[0] = cur[0] + (DIRECTION * move)
        print(f"     base -> {cur[0]:.1f} (move={move})")
        mc.send_angles(cur, 15)
        time.sleep(SETTLE_SEC)        # 완전히 멈춘 뒤 다음 읽기

    print("방위각 정렬 실패")
    return None, None


def measure_depth_mm(pipeline, align, cx, cy):
    xs, ys = int(round(cx)), int(round(cy))
    vals = []; step = 6
    for _ in range(DEPTH_FRAMES):
        frames = align.process(pipeline.wait_for_frames())
        depth = frames.get_depth_frame()
        if not depth:
            continue
        for dy in range(-DEPTH_RADIUS, DEPTH_RADIUS + 1, step):
            for dx in range(-DEPTH_RADIUS, DEPTH_RADIUS + 1, step):
                x = min(max(xs + dx, 0), FRAME_W - 1)
                y = min(max(ys + dy, 0), FRAME_H - 1)
                d = depth.get_distance(x, y)
                if 0.05 < d < 4.0:
                    vals.append(d * 1000.0)
    return float(np.median(vals)) if vals else None


def taught_pick(mc, base_angle):
    def wb(j):
        out = list(j); out[0] = base_angle; return out
    print(f"티칭 집기 시작 (base={base_angle:.1f})")
    mc.set_gripper_state(0, 50); time.sleep(1); print("그리퍼 열기")
    mc.send_angles(wb(ABOVE_J), 20); time.sleep(3); print("책 위로 접근")
    mc.send_angles(wb(PICK_J), 15);  time.sleep(3); print("내려서 잡는 높이")
    mc.set_gripper_state(1, 50); time.sleep(2); print("그리퍼 닫기")
    mc.send_angles(wb(LIFT_J), 15);  time.sleep(3); print("들어올리기")
    print("집기 완료")


def main():
    mc = connect_robot()
    pipeline, align = start_realsense()
    try:
        angles, last = align_to_qr(mc, pipeline, align)
        if angles is None:
            print("집기 취소(정렬 실패)"); return

        base_angle = angles[0]
        cx, cy = last
        depth_mm = measure_depth_mm(pipeline, align, cx, cy)
        print(f"QR 거리 depth = {depth_mm if depth_mm is None else round(depth_mm,1)} mm")

        proceed = True
        if GATE_MODE != "off":
            if depth_mm is None:
                if GATE_MODE == "hard":
                    proceed = False; print("depth 실패 → 취소")
                else:
                    print("depth 못 읽음 → 게이트 건너뜀")
            elif depth_mm < DEPTH_MIN or depth_mm > DEPTH_MAX:
                print(f"depth 범위 밖({depth_mm:.0f})")
                proceed = (GATE_MODE == "soft")
        if not proceed:
            print("집기 취소"); return

        print("정렬 OK → 집기")
        taught_pick(mc, base_angle)
    except KeyboardInterrupt:
        print("중단")
    finally:
        pipeline.stop()
        print("RealSense 종료")


if __name__ == "__main__":
    main()
