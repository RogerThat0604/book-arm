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

# 카테고리별 바구니 위치: 반드시 실제로 티칭해서 수정하기
BASKET_POSES = {
    "문학": [35, -20, -20, 0, 30, 0],
    "과학": [0, -20, -20, 0, 30, 0],
    "역사": [-35, -20, -20, 0, 30, 0],
}

HOME_J = [0, 0, 0, 0, 0, 0]

# depth 게이트 (v4와 동일)
DEPTH_MIN, DEPTH_MAX = 120.0, 450.0
GATE_MODE   = "soft"          # soft / hard / off
DEPTH_FRAMES = 8
DEPTH_RADIUS = 45

# ▼ TEACH: 실측한 잡기 자세(관절각). j0(base)는 정렬값으로 덮임 ▼
ABOVE_J = [-1.1, -10.8, -6.4, -4.0, 8.4, 3.6]
PICK_J  = [-1.1, -72.5, 9.5, 45.5, 4.5, 5.0]
LIFT_J  = [-1.1, -66.6, 9.9, 33.0, 4.5, 4.9]
# 카메라-그리퍼 방위 오프셋(deg). 정렬 후 그리퍼가 책보다
#   왼쪽에 서면 + 방향, 오른쪽에 서면 - 방향으로 조정(보통 ±몇 도)
BASE_OFFSET_DEG = 0.0
# ▲────────────────────────────────────────────────────────────▲

# ▼ REACH: 뻗는 거리를 depth로 상대 보정 (선택) ▼
#   REACH_GAIN = 0  이면 보정 끔 → v6와 완전히 동일(고정 거리, 안전)
#   켜려면: 기준 위치에서 출력된 depth를 DEPTH_REF에 넣고, REACH_GAIN을 0.8 부터 시작
DEPTH_REF   = 254.0   # mm. 기준(잘 잡히는) 위치에서의 depth
REACH_GAIN  = 0.8     # 0=고정. 0.8~1.0 권장(덜 뻗으면 ↑, 더 뻗으면 ↓)
REACH_MAX   = 80.0    # mm. 한 번에 보정할 최대 거리(안전 한계)
COORD_LIMIT = 275.0   # mm. MyCobot280 좌표 한계(±281) 안쪽
LIFT_Z_MM   = 90.0    # mm. 집은 뒤 수직으로 들어올리는 높이
# ▲────────────────────────────────────────────────────────────▲

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

def decode_qr_category(frame):
    data, pts, _ = _qr.detectAndDecode(frame)

    if not data:
        return None

    data = data.strip()

    if "문학" in data:
        return "문학"
    if "과학" in data:
        return "과학"
    if "역사" in data:
        return "역사"

    return None


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

def read_qr_category_stable(pipeline, align, tries=30):
    categories = []

    for i in range(tries):
        frame, _ = get_frames(pipeline, align)
        if frame is None:
            continue

        # 원본 시도
        data, pts, _ = _qr.detectAndDecode(frame)

        # 실패하면 grayscale + CLAHE로 재시도
        if not data:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = _clahe.apply(gray)
            data, pts, _ = _qr.detectAndDecode(gray)

        # 실패하면 확대해서 재시도
        if not data:
            big = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            data, pts, _ = _qr.detectAndDecode(big)

        if data:
            text = data.strip()
            print(f"QR decode raw: {text}")

            if "문학" in text:
                categories.append("문학")
            elif "과학" in text:
                categories.append("과학")
            elif "역사" in text:
                categories.append("역사")

        time.sleep(0.05)

    if not categories:
        return None

    return max(set(categories), key=categories.count)

def stable_read(pipeline, align, prev_dx=None):
    dxs, cxs, cys = [], [], []
    categories = []

    for _ in range(SAMPLES_PER_READ):
        frame, _ = get_frames(pipeline, align)
        if frame is None:
            continue

        category = decode_qr_category(frame)
        if category:
            categories.append(category)

        res = detect_qr_validated(frame)
        if res is None:
            continue

        dx, cx, cy = res

        if prev_dx is not None and abs(dx - prev_dx) > OUTLIER_PX and dxs:
            continue

        dxs.append(dx)
        cxs.append(cx)
        cys.append(cy)
        time.sleep(0.02)

    if len(dxs) < MIN_VALID:
        return None

    category = None
    if categories:
        category = max(set(categories), key=categories.count)

    return (
        float(np.median(dxs)),
        float(np.median(cxs)),
        float(np.median(cys)),
        category
    )


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
        dx, cx, cy, category = read
        last_cxcy = (cx, cy)
        prev_dx = dx
        print(f"  [{i+1}] dx={dx:.1f} (median, cx={cx:.1f})")

        if abs(dx) < CENTER_THRESHOLD:
            confirm += 1
            if confirm >= CONFIRM_FRAMES:
                print("  방위각 정렬 완료")
                return mc.get_angles(), last_cxcy, category
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
    return None, None, None


def measure_depth_mm(pipeline, align, cx, cy):
    """
    QR 중심 주변에서 '가까운 표면(책)'의 깊이만 골라 median(mm).
    넓은 창에 섞인 배경(먼 벽 ~1m)을 배제해 1089 같은 오측정을 막는다.
    """
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
    if not vals:
        return None
    vals = np.array(vals)
    near_ref = np.percentile(vals, 5)          # 가장 가까운 쪽 = 책 표면
    near = vals[vals <= near_ref + 60.0]        # 그로부터 6cm 이내만 = 책
    return float(np.median(near))


def taught_pick(mc, base_angle, depth_mm):
    def wb(j):
        out = list(j); out[0] = base_angle; return out
    print(f"티칭 집기 시작 (base={base_angle:.1f})")
    mc.set_gripper_state(0, 50); time.sleep(1); print("그리퍼 열기")
    mc.send_angles(wb(ABOVE_J), 20); time.sleep(3); print("책 위로 접근")
    mc.send_angles(wb(PICK_J), 10);  time.sleep(3); print("내려서 잡는 높이")

    # ── 뻗는 거리 상대 보정(선택) ──
    if REACH_GAIN and DEPTH_REF and depth_mm:
        coords = mc.get_coords()
        if coords and len(coords) >= 6:
            x, y, z, rx, ry, rz = coords
            dr = REACH_GAIN * (depth_mm - DEPTH_REF)
            dr = max(min(dr, REACH_MAX), -REACH_MAX)
            r = math.hypot(x, y)
            th = math.atan2(y, x)
            x2 = (r + dr) * math.cos(th)
            y2 = (r + dr) * math.sin(th)
            if abs(x2) <= COORD_LIMIT and abs(y2) <= COORD_LIMIT:
                print(f"  거리 보정 dr={dr:+.1f}mm (depth {depth_mm:.0f} vs 기준 {DEPTH_REF:.0f})")
                mc.send_coords([x2, y2, z, rx, ry, rz], 20, 1); time.sleep(2)
            else:
                print(f"  보정값이 좌표 한계 초과 → 보정 생략(고정 위치로 집기)")

    mc.set_gripper_state(1, 50); time.sleep(2); print("그리퍼 닫기")

    # 들어올리기: 현재 위치에서 Z만 위로(확실한 수직 상승)
    coords = mc.get_coords()
    if coords and len(coords) >= 6:
        up = list(coords); up[2] += LIFT_Z_MM
        print(f"들어올리기 (Z {coords[2]:.0f} -> {up[2]:.0f})")
        mc.send_coords(up, 25, 1); time.sleep(3)
    else:
        print("들어올리기 (관절 폴백)")
        mc.send_angles(wb(LIFT_J), 15); time.sleep(3)
    print("집기 완료")

def place_to_category(mc, category):
    if category not in BASKET_POSES:
        print(f"알 수 없는 카테고리: {category}")
        return False

    target = BASKET_POSES[category]

    print(f"📦 {category} 바구니로 이동")
    mc.send_angles(target, 20)
    time.sleep(3)

    print("🖐 책 놓기")
    mc.set_gripper_state(0, 50)
    time.sleep(1.5)

    print("🏠 HOME 복귀")
    mc.send_angles(HOME_J, 20)
    time.sleep(2)

    print(f"✅ {category} 분류 완료")
    return True

def main():
    mc = connect_robot()
    pipeline, align = start_realsense()

    try:
        angles, last, category = align_to_qr(mc, pipeline, align)

        if angles is None:
            print("집기 취소(정렬 실패)")
            return

        if category is None:
            print("정렬 중 카테고리 인식 실패 → 정렬 후 재인식 시도")
            category = read_qr_category_stable(pipeline, align)

        if category is None:
            print("카테고리 최종 인식 실패")
            return

        print(f"📚 QR 카테고리 인식: {category}")

        base_angle = angles[0] + BASE_OFFSET_DEG
        print(f"정렬 base={angles[0]:.1f} + 오프셋 {BASE_OFFSET_DEG:.1f} -> {base_angle:.1f}")

        cx, cy = last
        depth_mm = measure_depth_mm(pipeline, align, cx, cy)
        print(f"QR 거리 depth = {depth_mm if depth_mm is None else round(depth_mm,1)} mm")

        proceed = True

        if GATE_MODE != "off":
            if depth_mm is None:
                if GATE_MODE == "hard":
                    proceed = False
                    print("depth 실패 → 취소")
                else:
                    print("depth 못 읽음 → 게이트 건너뜀")
            elif depth_mm < DEPTH_MIN or depth_mm > DEPTH_MAX:
                print(f"depth 범위 밖({depth_mm:.0f})")
                proceed = (GATE_MODE == "soft")

        if not proceed:
            print("집기 취소")
            return

        print("정렬 OK → 집기")

        if is_untaught():
            print("티칭값 필요")
            return

        taught_pick(mc, base_angle, depth_mm)

        print("집기 성공 → 카테고리 바구니 이동")
        place_to_category(mc, category)

    except KeyboardInterrupt:
        print("중단")

    finally:
        pipeline.stop()
        print("RealSense 종료")

if __name__ == "__main__":
    main()
