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
import math
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

# ▼ TOP-DOWN GRASP 설정 (teach_vertical.py 로 측정) ▼
# 그리퍼가 책상을 향해 수직으로 내려가는 자세의 (rx, ry, rz).
# 책 위치가 바뀌어도 이 자세는 그대로 재사용.
VERTICAL_RPY = (179.5, 0.2,-89.6)   # ← teach_vertical.py 출력값으로 교체

# 책상 높이(mm). 책 잡는 z. teach_vertical 에서 측정.
Z_PICK     = 80.0                    # ← 교체
Z_APPROACH = Z_PICK + 80.0           # 책 위 80mm 접근 자세
Z_LIFT     = Z_PICK + 120.0          # 잡은 뒤 들어올리는 높이

# 책이 놓일 수 있는 반경 범위(mm). MyCobot280 작업영역 안에서.
R_MIN, R_MAX = 130.0, 240.0

# 카메라 → 그리퍼 평면 오프셋 (mm)
DX_CAM = 0.0     # 카메라가 그리퍼 앞쪽으로 mm 떨어진 거리(+면 앞)
DY_CAM = 0.0     # 좌우 오프셋
# ▲────────────────────────────────────────────────────────────▲

# (남겨두는) 폴백용 — top-down 실패 시에만 사용
ABOVE_J = [-1.1, -10.8, -6.4, -4.0, 8.4, 3.6]
PICK_J  = [-1.1, -72.5, 9.5, 45.5, 4.5, 5.0]
LIFT_J  = [-1.1, -66.6, 9.9, 33.0, 4.5, 4.9]
BASE_OFFSET_DEG = 0.0

# ▼ REACH: depth로 책 반경 추정 ▼
DEPTH_REF   = 254.0   # mm. 기준 depth (이 거리에서 R_BASE 로 잡힘)
R_BASE      = 180.0   # mm. DEPTH_REF 일 때의 책 반경(작업거리)
REACH_GAIN  = 0.8     # depth 변화 → 반경 변화 비율
COORD_LIMIT = 275.0   # mm. MyCobot280 좌표 한계

FRAME_W, FRAME_H = 640, 480
AREA_MIN = 200                # px^2. 너무 작은 사각형 = 노이즈
AREA_MAX = 0.7 * FRAME_W * FRAME_H   # 너무 큰 것 = 오검출

# 예시(미교체) 값 감지용 — 이대로면 집기를 막아 팔이 내리꽂는 사고 방지
_PLACEHOLDER = ([0, -20, -20, 0, 30, 0],
                [0, -35, -35, 0, 40, 0],
                [0, -15, -15, 0, 30, 0])


def is_untaught():
    return [ABOVE_J, PICK_J, LIFT_J] == [list(p) for p in _PLACEHOLDER]

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


def top_down_pick(mc, base_angle, depth_mm):
    """
    Top-down (수직) 파지: 그리퍼가 책상에 수직으로 내려가 책 등을 정면에서 잡는다.
      1) depth → 책 반경 R 추정
      2) (R, base_angle) → 책 위치 (x_book, y_book)
      3) 책 위 Z_APPROACH 로 수직 자세 접근
      4) Z만 Z_PICK 으로 하강
      5) 잡고 Z_LIFT 로 상승
    """
    import math
    # 1) 반경 R 추정 (depth가 멀면 책도 멀다)
    if depth_mm:
        r = R_BASE + REACH_GAIN * (depth_mm - DEPTH_REF)
    else:
        r = R_BASE
    r = max(min(r, R_MAX), R_MIN)
    print(f"top-down: depth={depth_mm} → 반경 R={r:.1f}mm, base={base_angle:.1f}°")

    # 2) 책 위치(평면)
    th = math.radians(base_angle)
    x_book = r * math.cos(th) + DX_CAM
    y_book = r * math.sin(th) + DY_CAM

    if abs(x_book) > COORD_LIMIT or abs(y_book) > COORD_LIMIT:
        print(f"  좌표 한계 초과(x={x_book:.0f}, y={y_book:.0f}) → 집기 취소")
        return False

    rx, ry, rz = VERTICAL_RPY
    approach = [x_book, y_book, Z_APPROACH, rx, ry, rz]
    pick     = [x_book, y_book, Z_PICK,     rx, ry, rz]
    lift     = [x_book, y_book, Z_LIFT,     rx, ry, rz]

    print(f"  책 평면 위치: ({x_book:.0f}, {y_book:.0f})")
    print(f"  Z: 접근 {Z_APPROACH:.0f} → 잡기 {Z_PICK:.0f} → 들기 {Z_LIFT:.0f}")

    print("그리퍼 열기")
    mc.set_gripper_state(0, 50); time.sleep(1)

    print("책 위 수직 접근")
    mc.send_coords(approach, 30, 1); time.sleep(3)

    print("수직 하강")
    mc.send_coords(pick, 15, 1); time.sleep(3)

    print("그리퍼 닫기")
    mc.set_gripper_state(1, 50); time.sleep(2)

    print("수직 상승")
    mc.send_coords(lift, 25, 1); time.sleep(3)
    print("집기 완료")
    return True


def taught_pick(mc, base_angle, depth_mm):
    """top-down 시도 → 실패 시 관절 폴백."""
    ok = False
    try:
        ok = top_down_pick(mc, base_angle, depth_mm)
    except Exception as e:
        print(f"top-down 실패: {e} → 관절 폴백")
        ok = False
    if ok:
        return
    # ── 폴백: 옛 티칭 자세 ──
    def wb(j):
        out = list(j); out[0] = base_angle; return out
    print("[폴백] 관절 티칭 파지")
    mc.set_gripper_state(0, 50); time.sleep(1)
    mc.send_angles(wb(ABOVE_J), 20); time.sleep(3)
    mc.send_angles(wb(PICK_J), 10);  time.sleep(3)
    mc.set_gripper_state(1, 50); time.sleep(2)
    mc.send_angles(wb(LIFT_J), 15);  time.sleep(3)
    print("집기 완료(폴백)")


def main():
    mc = connect_robot()
    pipeline, align = start_realsense()
    try:
        angles, last = align_to_qr(mc, pipeline, align)
        if angles is None:
            print("집기 취소(정렬 실패)"); return

        base_angle = angles[0] + BASE_OFFSET_DEG
        print(f"정렬 base={angles[0]:.1f} + 오프셋 {BASE_OFFSET_DEG:.1f} -> {base_angle:.1f}")
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
        if is_untaught():
            print("⚠ 티칭값이 아직 예시값입니다 — 그대로 두면 팔이 위험하게 내리꽂습니다.")
            print("  먼저 'python jog_teach.py' 로 자세를 가르친 뒤,")
            print("  출력된 ABOVE_J/PICK_J/LIFT_J 세 줄을 이 파일에 붙여넣으세요.")
            print("  안전을 위해 집기를 생략합니다.")
            return
        taught_pick(mc, base_angle, depth_mm)
    except KeyboardInterrupt:
        print("중단")
    finally:
        pipeline.stop()
        print("RealSense 종료")


if __name__ == "__main__":
    main()
