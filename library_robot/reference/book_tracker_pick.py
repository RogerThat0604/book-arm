#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QR 중심점 기반 Visual Servoing + Pick 시퀀스
도서관 책 분류 로봇 (JetCobot / MyCobot280)

기존 detect_book_dx()의 "밝은 영역 = 책" 오인식 문제를 제거하고,
이미 동작 중인 QR을 정렬 기준점으로 재사용한다.

수정한 핵심 버그
  (1) 정렬 후 HOME 복귀로 base 각도를 날리던 문제 -> 정렬각 유지
  (2) int(dx*KP)의 비대칭 절삭 -> round() 사용
  (3) 단일 프레임 정렬 판정 -> 2프레임 연속 확인
  (4) 하드코딩 관절각 하강 -> 좌표(Z) 기반 하강 + 관절각 폴백

의존성:  pip install pyrealsense2 pymycobot opencv-python numpy
QR 검출: OpenCV 내장 cv2.QRCodeDetector (별도 설치 불필요)
         더 견고하게 하려면 pyzbar 사용(주석 참고)
"""

import time
import cv2
import numpy as np
import pyrealsense2 as rs
from pymycobot import MyCobot280 as MyCobot

# ───────────────────────── 설정 ─────────────────────────
PORT = "/dev/ttyJETCOBOT"      # 레포 robot_controller_topic.py 기준
BAUD = 1000000

# Visual Servoing 파라미터
CENTER_THRESHOLD = 20          # px. 35 -> 20으로 강화(D 참고). 작업거리에 맞게 보정
KP               = 0.06        # 비례 게인
MAX_MOVE         = 8           # deg. 한 스텝 최대 회전
MAX_TRY          = 25
DIRECTION        = -1          # 발산하면 +1로 뒤집기 (부호 캘리브레이션)
CONFIRM_FRAMES   = 2           # 연속 N프레임 정렬되어야 완료(노이즈 방지)
SETTLE_SEC       = 0.8         # 회전 후 정착 대기

# 좌표 기반 하강 파라미터
DESCEND_Z_MM     = 60          # 현재 위치에서 Z를 얼마나 내릴지(mm). 실측 보정 필요
COORD_SPEED      = 30
COORD_MODE       = 1           # 0=각도보간, 1=직선보간

FRAME_W, FRAME_H = 640, 480


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


# ───────────────────── QR 중심점 검출 ─────────────────────
_qr = cv2.QRCodeDetector()


def detect_qr_dx(frame):
    """
    QR 코드의 중심점으로 dx(화면 중심과의 가로 오차)를 계산한다.
    반환: (dx, cx, data) 또는 None
    """
    h, w = frame.shape[:2]

    # 흑백 + 약한 블러가 QR 검출 안정성을 높인다
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    data, points, _ = _qr.detectAndDecode(gray)
    if points is None:
        return None

    pts = points.reshape(-1, 2)          # (4,2) 코너 좌표
    cx = float(np.mean(pts[:, 0]))       # 코너 평균 = QR 중심 x
    dx = cx - (w / 2.0)
    return dx, cx, data

    # ── pyzbar 대안(여러 QR/작은 QR에 더 강함) ──
    # from pyzbar.pyzbar import decode
    # for obj in decode(frame):
    #     x, y, bw, bh = obj.rect
    #     cx = x + bw / 2.0
    #     return cx - w / 2.0, cx, obj.data.decode()
    # return None


# ───────────────────── Visual Servoing ─────────────────────
def align_to_qr(mc, pipeline):
    """
    base joint를 회전시켜 QR을 화면 중앙에 맞춘다.
    성공 시 정렬된 관절각(list)을 반환, 실패 시 None.
    """
    print("QR 중앙 정렬 시작")
    confirm = 0

    for i in range(MAX_TRY):
        frame = get_frame(pipeline)
        if frame is None:
            print("  프레임 실패")
            time.sleep(0.2)
            continue

        res = detect_qr_dx(frame)
        if res is None:
            print("  QR 미검출")
            confirm = 0                  # 놓치면 확인 카운터 리셋
            time.sleep(0.2)
            continue

        dx, cx, data = res
        print(f"  [{i+1}/{MAX_TRY}] dx={dx:.1f}, cx={cx:.1f}, qr='{data}'")

        # ── 2프레임 연속 확인 ──
        if abs(dx) < CENTER_THRESHOLD:
            confirm += 1
            if confirm >= CONFIRM_FRAMES:
                print("  정렬 완료(연속 확인)")
                return mc.get_angles()
            time.sleep(0.15)
            continue
        confirm = 0

        current = mc.get_angles()
        if not current or len(current) < 6:
            print("  각도 읽기 실패:", current)
            return None

        # P-제어: round로 비대칭 절삭 제거
        move = round(dx * KP)
        move = max(min(move, MAX_MOVE), -MAX_MOVE)
        if move == 0:                    # dead-band: 최소 1deg는 움직여 수렴 보장
            move = 1 if dx > 0 else -1

        current[0] = current[0] + (DIRECTION * move)
        side = "왼쪽" if dx < 0 else "오른쪽"
        print(f"    base -> {current[0]:.1f} (move={move}, {side} 보정)")

        mc.send_angles(current, 15)
        time.sleep(SETTLE_SEC)

    print("중앙 정렬 실패(MAX_TRY 초과)")
    return None


# ───────────────────── 집기 시퀀스 ─────────────────────
def pick_at_current_pose(mc):
    """
    정렬된 현재 자세를 유지한 채 집는다. HOME 복귀 없음(정렬각 보존).
    좌표 기반 하강을 우선 시도하고, 실패하면 관절각 폴백.
    """
    print("그리퍼 열기")
    mc.set_gripper_state(0, 50)
    time.sleep(1)

    coords = mc.get_coords()
    if coords and len(coords) >= 6:
        # ── 좌표 기반: 현재 위치에서 Z만 하강 → 집기 → 상승 ──
        print(f"현재 좌표 {[(round(c,1)) for c in coords]}")

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
        # ── 폴백: 좌표 못 읽을 때만 base 각도 보존하며 관절각 하강 ──
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