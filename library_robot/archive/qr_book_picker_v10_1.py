#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qr_book_picker_v10.py  —  Top-down 시나리오 본 파이프라인

흐름:
  HOME → OBSERVE 자세(카메라가 바닥을 내려다봄)
       → QR 검출 + 카테고리 디코딩
       → 픽셀(cx,cy) → 로봇 좌표(x_book, y_book) 변환
       → 책 위 수직 접근 → 수직 하강 → 잡기 → 상승
       → 카테고리 바구니로 이동 → 놓기 → HOME

캘리브레이션 필요:
  calibrate_topdown.py 를 한 번 돌려 출력값을 아래 'CALIBRATION' 블록에 붙여넣기.
"""

import time
import cv2
import numpy as np
import pyrealsense2 as rs
from pymycobot import MyCobot280 as MyCobot

PORT = "/dev/ttyJETCOBOT"
BAUD = 1000000

# ─────────────── CALIBRATION (calibrate_topdown.py 결과로 교체) ───────────────
OBSERVE_J     = [2.5, -14.1, -0.9, -44.0, -4.8, 6.0]   # ← 교체
DEPTH_REF     = 219.5                       # mm
Z_TABLE       = 86.9                        # mm. 책상 표면 z
MM_PER_PX_X   = 0.019                       # cx 1px → x_robot mm/px (부호 주의)
MM_PER_PX_Y   = 0.019                        # cy 1px → y_robot mm/px
REF_CX        = 249.5                       # 기준 픽셀
REF_CY        = 323.0
REF_X_ROBOT   = 147.0                    # 기준 로봇 좌표
REF_Y_ROBOT   = -61.8
# 추가 오프셋 (책 잡으러 가는 위치를 측정값에서 mm만큼 이동)
OFFSET_X = 80.0   # +면 책 앞쪽으로(로봇 본체에서 멀어지는 방향)
OFFSET_Y = 0.0
# ───────────────────────────────────────────────────────────────────────────────

# Top-down 자세 — calibrate에선 OBSERVE 자세의 (rx,ry,rz)를 그대로 쓰는 게 안전
# 책 잡을 때는 VERTICAL_RPY 로 강제 수직(아래 자동 측정).
# 일단 OBSERVE 좌표 측정 시점의 rpy를 그대로 사용(코드가 자동으로 읽음).

# 동작 파라미터
Z_APPROACH_OFFSET = 80.0     # 책 위 80mm 접근
Z_LIFT_OFFSET     = 120.0    # 잡은 뒤 들어올림
COORD_LIMIT       = 275.0    # mm. 작업영역 한계
APPROACH_SPEED    = 30
DESCEND_SPEED     = 15
LIFT_SPEED        = 25

# 바구니 위치 (관절각, 실측 후 교체)
BASKET_POSES = {
    "문학": [35, -20, -20, 0, 30, 0],
    "과학": [0,  -20, -20, 0, 30, 0],
    "역사": [-35, -20, -20, 0, 30, 0],
}
HOME_J = [0, 0, 0, 0, 0, 0]

# QR / 검출
FRAME_W, FRAME_H = 640, 480
AREA_MIN = 200
AREA_MAX = 0.7 * FRAME_W * FRAME_H
SAMPLES_PER_READ = 15
MIN_VALID        = 5
DEPTH_RADIUS     = 30
DEPTH_FRAMES     = 5

_qr = cv2.QRCodeDetector()
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ───────────────────── 로봇/카메라 ─────────────────────
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


def grab(pipeline, align):
    frames = align.process(pipeline.wait_for_frames())
    color = frames.get_color_frame()
    depth = frames.get_depth_frame()
    if not color or not depth:
        return None, None
    return np.asanyarray(color.get_data()), depth


# ───────────────────── QR 검출 ─────────────────────
def detect_qr_validated(frame):
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
    cx = float(np.mean(p[:, 0])); cy = float(np.mean(p[:, 1]))
    if not (0 <= cx < w and 0 <= cy < h):
        return None
    area = cv2.contourArea(p)
    if not (AREA_MIN < area < AREA_MAX):
        return None
    if not cv2.isContourConvex(p.astype(np.int32)):
        return None
    return cx, cy


def decode_category(frame):
    """pyzbar 기반 디코딩. 여러 전처리를 시도해 작은/흐릿한 QR도 잡는다."""
    try:
        from pyzbar.pyzbar import decode as zbar_decode
    except ImportError:
        # pyzbar 없으면 cv2 폴백
        data, _, _ = _qr.detectAndDecode(frame)
        if data:
            text = data.strip()
            for cat in ("문학", "과학", "역사"):
                if cat in text:
                    return cat
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 여러 전처리 후보를 순서대로 시도
    candidates = [
        gray,                                          # 원본 흑백
        _clahe.apply(gray),                            # 대비 보정
        cv2.resize(gray, None, fx=2.0, fy=2.0,        # 2배 확대
                   interpolation=cv2.INTER_CUBIC),
        cv2.resize(_clahe.apply(gray), None,           # 2배 + CLAHE
                   fx=2.0, fy=2.0,
                   interpolation=cv2.INTER_CUBIC),
    ]

    for img in candidates:
        results = zbar_decode(img)
        for r in results:
            text = r.data.decode("utf-8", errors="ignore").strip()
            for cat in ("문학", "과학", "역사"):
                if cat in text:
                    return cat
            # 매칭 안 돼도 디코딩은 됐으니 로그 남김
            if text:
                print(f"  (디코딩됨: '{text}' — 카테고리 매칭 안됨)")
    return None


def stable_observe(pipeline, align):
    """OBSERVE 자세에서 QR 픽셀 위치 + 카테고리를 안정 측정."""
    cxs, cys, cats = [], [], []
    for _ in range(SAMPLES_PER_READ):
        frame, _ = grab(pipeline, align)
        if frame is None:
            continue
        cat = decode_category(frame)
        if cat:
            cats.append(cat)
        res = detect_qr_validated(frame)
        if res is None:
            continue
        cxs.append(res[0]); cys.append(res[1])
        time.sleep(0.04)
    if len(cxs) < MIN_VALID:
        return None
    cx = float(np.median(cxs)); cy = float(np.median(cys))
    category = max(set(cats), key=cats.count) if cats else None
    return cx, cy, category


def measure_depth(pipeline, align, cx, cy):
    """책 표면 깊이(mm). 가까운 표면만 골라 배경 배제."""
    xs, ys = int(round(cx)), int(round(cy))
    vals = []; step = 5
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
    near = vals[vals <= np.percentile(vals, 15) + 30.0]  # 가까운 표면만
    return float(np.median(near))


# ───────────────────── 픽셀 → 로봇 좌표 매핑 ─────────────────────
def pixel_to_robot(cx, cy, depth_mm):
    scale = (depth_mm / DEPTH_REF) if (depth_mm and DEPTH_REF) else 1.0
    x = REF_X_ROBOT + (cx - REF_CX) * MM_PER_PX_X * scale
    y = REF_Y_ROBOT + (cy - REF_CY) * MM_PER_PX_Y * scale
    return x + OFFSET_X, y + OFFSET_Y    # ← 변경

# ───────────────────── Top-down 파지 ─────────────────────
def topdown_pick(mc, x_book, y_book):
    """현재 자세의 (rx,ry,rz)를 유지한 채 책 위로 가서 수직 하강."""
    if abs(x_book) > COORD_LIMIT or abs(y_book) > COORD_LIMIT:
        print(f"  좌표 한계 초과 ({x_book:.0f},{y_book:.0f}) → 취소")
        return False

    coords = mc.get_coords()
    if not coords or len(coords) < 6:
        print("  좌표 읽기 실패 → 취소"); return False
    _, _, _, rx, ry, rz = coords
    print(f"  현재 자세 rpy=({rx:.0f},{ry:.0f},{rz:.0f}) — 수직 자세 유지")

    z_approach = Z_TABLE + Z_APPROACH_OFFSET
    z_pick     = Z_TABLE
    z_lift     = Z_TABLE + Z_LIFT_OFFSET

    print("그리퍼 열기")
    mc.set_gripper_state(0, 50); time.sleep(1)

    print(f"책 위로 접근 → ({x_book:.0f}, {y_book:.0f}, {z_approach:.0f})")
    mc.send_coords([x_book, y_book, z_approach, rx, ry, rz], APPROACH_SPEED, 1)
    time.sleep(3)

    print(f"수직 하강 → z={z_pick:.0f}")
    mc.send_coords([x_book, y_book, z_pick, rx, ry, rz], DESCEND_SPEED, 1)
    time.sleep(3)

    print("그리퍼 닫기")
    mc.set_gripper_state(1, 50); time.sleep(2)

    print(f"수직 상승 → z={z_lift:.0f}")
    mc.send_coords([x_book, y_book, z_lift, rx, ry, rz], LIFT_SPEED, 1)
    time.sleep(3)
    print("집기 완료")
    return True


# ───────────────────── 카테고리 배치 ─────────────────────
def place_to_basket(mc, category):
    if category not in BASKET_POSES:
        print(f"⚠ 알 수 없는 카테고리: {category} → HOME")
        mc.send_angles(HOME_J, 25); time.sleep(2)
        return False
    print(f"📦 {category} 바구니로 이동")
    mc.send_angles(BASKET_POSES[category], 25); time.sleep(3)
    print("🖐 책 놓기")
    mc.set_gripper_state(0, 50); time.sleep(1.5)
    print("🏠 HOME 복귀")
    mc.send_angles(HOME_J, 25); time.sleep(2)
    print(f"✅ {category} 분류 완료")
    return True


# ───────────────────── main ─────────────────────
def main():
    mc = connect_robot()
    pipeline, align = start_realsense()
    try:
        print("HOME")
        mc.send_angles(HOME_J, 25); time.sleep(2)

        print("OBSERVE 자세로 이동 — 카메라가 바닥을 내려다봅니다")
        mc.send_angles(OBSERVE_J, 25); time.sleep(3)

        print("책 탐색 중...")
        read = stable_observe(pipeline, align)
        if read is None:
            print("⚠ QR 검출 실패 — 책이 시야 안에 있는지 확인"); return
        cx, cy, category = read
        print(f"  QR 픽셀: ({cx:.1f}, {cy:.1f})")

        depth_mm = measure_depth(pipeline, align, cx, cy)
        print(f"  표면 깊이: {depth_mm:.1f} mm" if depth_mm else "  깊이 측정 실패(기준값 사용)")

        x_book, y_book = pixel_to_robot(cx, cy, depth_mm or DEPTH_REF)
        print(f"  책 위치(로봇 좌표): ({x_book:.1f}, {y_book:.1f}) mm")

        # 카테고리 재시도(정렬 단계가 없으니 보강)
        if category is None:
            print("  카테고리 재인식 시도...")
            for _ in range(20):
                frame, _ = grab(pipeline, align)
                if frame is None: continue
                category = decode_category(frame)
                if category: break
                time.sleep(0.05)
        if category is None:
            print("⚠ 카테고리 인식 실패 → 집기만 하고 HOME"); return
        print(f"  카테고리: {category}")

        ok = topdown_pick(mc, x_book, y_book)
        if not ok:
            print("집기 취소"); return

        place_to_basket(mc, category)

    except KeyboardInterrupt:
        print("중단")
    finally:
        pipeline.stop()
        print("RealSense 종료")


if __name__ == "__main__":
    main()
