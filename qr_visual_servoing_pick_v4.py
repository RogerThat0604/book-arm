#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qr_visual_servoing_pick_v4.py  (데모용 권장)

전략: 방위각 정렬(카메라) + 티칭 집기(좌표 추정 X) + depth 게이트
  - MyCobot280은 반경 ~281mm가 한계 → 책은 '닿는 호' 위에 놓는다(각도 자유)
  - 카메라는 base 회전으로 방위각만 맞춤 (책 쪽으로 팔을 돌림)
  - 집기는 손으로 가르쳐 둔 관절각으로 실행 (base만 정렬값으로 교체)
  - depth는 "책이 집을 수 있는 거리에 있나" 확인용으로만 사용

★ 데모 전 할 일 (아래 ▼TEACH / ▼GATE):
  1) 책을 닿는 위치에 놓고 팔을 손으로 움직여 잡기 좋은 자세 3개를 mc.get_angles()로 기록
  2) 그 값을 ABOVE/PICK/LIFT 의 [j1..j5]에 넣기 (base=j0는 코드가 정렬값으로 덮어씀)
  3) 책을 제 위치에 놓고 한 번 돌려 출력되는 depth를 보고 DEPTH_MIN/MAX 범위 잡기
"""

import time
import cv2
import numpy as np
import pyrealsense2 as rs
from pymycobot import MyCobot280 as MyCobot

PORT = "/dev/ttyJETCOBOT"
BAUD = 1000000

# 방위각 정렬
CENTER_THRESHOLD = 20
KP               = 0.05
MAX_MOVE         = 6
DIRECTION        = -1
CONFIRM_FRAMES   = 2
SETTLE_SEC       = 0.8
ALIGN_TIMEOUT    = 35
MAX_ADJUST       = 25

# ▼─────────────── GATE: 책이 닿는 거리인지 확인 ───────────────▼
DEPTH_MIN = 120.0      # mm. 책 놓고 출력되는 depth 보고 조정
DEPTH_MAX = 450.0      # mm.
# "soft" : depth 못 읽으면 경고만 하고 집기 진행(권장, 데모 안전)
# "hard" : 범위 밖/측정 실패면 집기 중단
# "off"  : depth 게이트 완전 비활성
GATE_MODE = "soft"
DEPTH_FRAMES = 8       # depth 측정 시 누적할 프레임 수
DEPTH_RADIUS = 45      # px. QR 중심 주변 이 반경을 격자로 훑음(흰 여백 포함)
# ▲────────────────────────────────────────────────────────────▲

# ▼─────────────── TEACH: 손으로 가르친 집기 자세(관절각, deg) ───────────────▼
# j0(base)는 무시되고 정렬값으로 대체됨. j1~j5만 의미 있음.
# 예시값 — 반드시 mc.get_angles()로 실측해 교체할 것!
ABOVE_J = [0, -20, -20, 0, 30, 0]   # 책 위 접근
PICK_J  = [0, -35, -35, 0, 40, 0]   # 책 잡는 높이
LIFT_J  = [0, -15, -15, 0, 30, 0]   # 들어올리기
# ▲──────────────────────────────────────────────────────────────────────────▲

DEPTH_WIN = 7
FRAME_W, FRAME_H = 640, 480

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


def detect_qr(frame):
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


def measure_depth_mm(pipeline, align, cx, cy):
    """
    여러 프레임에 걸쳐 QR 중심 주변 넓은 격자를 훑어 유효 깊이의 median(mm).
    QR 검정 패턴(IR 흡수)으로 중앙이 0이어도, 흰 여백/책 표지에서 깊이를 줍는다.
    실패 시 None.
    """
    xs, ys = int(round(cx)), int(round(cy))
    vals = []
    step = 6
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
                if 0.05 < d < 4.0:          # 5cm~4m 사이만 유효
                    vals.append(d * 1000.0)
    if not vals:
        return None
    return float(np.median(vals))


def align_to_qr(mc, pipeline, align):
    print("QR 방위각 정렬 시작")
    t0 = time.time(); last_seen = t0; warned = False
    confirm = 0; adjust = 0; last = None
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
        last = (cx, cy, depth)
        print(f"  dx={dx:.1f}, cx={cx:.1f}")
        if abs(dx) < CENTER_THRESHOLD:
            confirm += 1
            if confirm >= CONFIRM_FRAMES:
                print("  방위각 정렬 완료")
                return mc.get_angles(), last
            time.sleep(0.12); continue
        confirm = 0
        cur = mc.get_angles()
        if not cur or len(cur) < 6:
            print("  각도 읽기 실패:", cur); return None, None
        move = round(dx * KP)
        move = max(min(move, MAX_MOVE), -MAX_MOVE)
        if move == 0:
            move = 1 if dx > 0 else -1
        cur[0] = cur[0] + (DIRECTION * move)
        print(f"    base -> {cur[0]:.1f} (move={move}) [{adjust+1}/{MAX_ADJUST}]")
        mc.send_angles(cur, 15)
        adjust += 1
        time.sleep(SETTLE_SEC)
    print("방위각 정렬 실패")
    return None, None


def taught_pick(mc, base_angle):
    """base만 정렬값으로 덮은 티칭 집기. 좌표 추정 없음."""
    def with_base(j):
        out = list(j); out[0] = base_angle; return out

    print(f"티칭 집기 시작 (base={base_angle:.1f})")
    print("그리퍼 열기")
    mc.set_gripper_state(0, 50); time.sleep(1)
    print("책 위로 접근")
    mc.send_angles(with_base(ABOVE_J), 20); time.sleep(3)
    print("내려서 잡는 높이")
    mc.send_angles(with_base(PICK_J), 15); time.sleep(3)
    print("그리퍼 닫기")
    mc.set_gripper_state(1, 50); time.sleep(2)
    print("들어올리기")
    mc.send_angles(with_base(LIFT_J), 15); time.sleep(3)
    print("집기 완료")


def main():
    mc = connect_robot()
    pipeline, align = start_realsense()
    try:
        angles, last = align_to_qr(mc, pipeline, align)
        if angles is None:
            print("집기 취소(정렬 실패)"); return

        base_angle = angles[0]
        cx, cy, _ = last
        depth_mm = measure_depth_mm(pipeline, align, cx, cy)
        print(f"QR 거리 depth = {depth_mm if depth_mm is None else round(depth_mm,1)} mm")

        proceed = True
        if GATE_MODE != "off":
            if depth_mm is None:
                if GATE_MODE == "hard":
                    print("depth 측정 실패 → 집기 취소"); proceed = False
                else:
                    print("depth 못 읽음(QR 검정 패턴/근접). 게이트 건너뛰고 진행")
            elif depth_mm < DEPTH_MIN:
                print(f"책이 너무 가까움({depth_mm:.0f}<{DEPTH_MIN:.0f})")
                proceed = (GATE_MODE == "soft")
            elif depth_mm > DEPTH_MAX:
                print(f"책이 너무 멈({depth_mm:.0f}>{DEPTH_MAX:.0f})")
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
