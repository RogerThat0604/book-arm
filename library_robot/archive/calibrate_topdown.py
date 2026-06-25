#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_topdown.py — 내려다보는(top-down) 시나리오용 캘리브레이션

수행 작업:
  1) OBSERVE 자세 기록 — 카메라가 책상 일정 구역을 내려다보는 팔 자세
  2) 평면 매핑 자동 측정 — 픽셀 (cx,cy) → 로봇 좌표 (x,y) mm/px 비율
     • QR을 두 위치에 두고 측정하여 비율과 부호를 자동 결정
  3) Z_TABLE 측정 — 책상 표면 z(mm)

사용 흐름:
  ▶ python calibrate_topdown.py
  ① OBSERVE 자세 만들기: 팔을 손으로 옮겨 카메라가 책상의 중앙을 내려다보게 한 뒤 Enter
     - 책상 위 20~30cm 정도 떠 있는 게 적당
     - 그 자세 그대로 재고정됨(처짐 방지)
  ② 평면 매핑: QR을 책상 위 두 위치(A, B)에 차례로 놓고 각각 Enter
     - A 위치: 화면 왼쪽 위쪽
     - B 위치: 화면 오른쪽 아래쪽 (두 점이 서로 멀수록 정밀)
  ③ Z_TABLE: 그리퍼 끝이 책상 표면에 살짝 닿을 정도로 손으로 옮긴 뒤 Enter

출력: v10_config 블록을 그대로 복사해 v10 파일에 붙여넣으면 끝.
"""

import time
import cv2
import numpy as np
import pyrealsense2 as rs
from pymycobot import MyCobot280 as MyCobot

PORT = "/dev/ttyJETCOBOT"
BAUD = 1000000
FRAME_W, FRAME_H = 640, 480

_qr = cv2.QRCodeDetector()
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def start_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    time.sleep(1)
    return pipeline, align


def grab(pipeline, align):
    frames = align.process(pipeline.wait_for_frames())
    color = frames.get_color_frame()
    depth = frames.get_depth_frame()
    if not color or not depth:
        return None, None
    return np.asanyarray(color.get_data()), depth


def find_qr_stable(pipeline, align, tries=40):
    """QR이 안정적으로 잡힐 때까지 시도. 픽셀 중심 + depth(mm) 반환."""
    cxs, cys, dms = [], [], []
    for _ in range(tries):
        frame, depth = grab(pipeline, align)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = _clahe.apply(gray)
        ok, pts = _qr.detect(gray)
        if not ok or pts is None:
            time.sleep(0.05); continue
        p = pts.reshape(-1, 2)
        cx, cy = float(np.mean(p[:, 0])), float(np.mean(p[:, 1]))
        if not (0 <= cx < FRAME_W and 0 <= cy < FRAME_H):
            continue
        # 주변 깊이 median
        vals = []
        for dy in range(-30, 31, 6):
            for dx in range(-30, 31, 6):
                x = min(max(int(cx + dx), 0), FRAME_W - 1)
                y = min(max(int(cy + dy), 0), FRAME_H - 1)
                d = depth.get_distance(x, y)
                if 0.05 < d < 4.0:
                    vals.append(d * 1000.0)
        if not vals:
            continue
        cxs.append(cx); cys.append(cy)
        dms.append(float(np.median(vals)))
        time.sleep(0.04)
        if len(cxs) >= 15:
            break
    if len(cxs) < 5:
        return None
    return float(np.median(cxs)), float(np.median(cys)), float(np.median(dms))


def main():
    mc = MyCobot(PORT, BAUD)
    mc.thread_lock = True
    mc.power_on()
    time.sleep(1)
    pipeline, align = start_realsense()
    print("RealSense 시작")

    # ── ① OBSERVE 자세 ──
    print("\n[1/3] OBSERVE 자세")
    print("  카메라가 책상의 중앙을 내려다보는 자세로 팔을 옮겨주세요.")
    print("  (책상 위 20~30cm 떠 있게)")
    input("  준비됐으면 Enter → 팔 힘 풀림 (한 손으로 받칠 준비!)... ")
    mc.release_all_servos(); time.sleep(0.3)
    input("  자세 잡았으면 Enter... ")
    observe_j = None
    for _ in range(6):
        observe_j = mc.get_angles()
        if observe_j and len(observe_j) == 6: break
        time.sleep(0.3)
    observe_j = [round(a, 1) for a in observe_j]
    mc.send_angles(observe_j, 30); time.sleep(2)
    print(f"  OBSERVE_J = {observe_j}")

    # OBSERVE 자세에서의 좌표(참고용)
    observe_c = mc.get_coords()
    if observe_c:
        observe_c = [round(c, 1) for c in observe_c]
        print(f"  OBSERVE 좌표 = {observe_c}")

    # ── ② 평면 매핑: 두 점 측정 ──
    print("\n[2/3] 평면 매핑 — QR을 두 위치에 놓고 좌표를 측정합니다.")
    samples = []
    for label, hint in [
        ("A", "QR을 책상 위, 화면 '왼쪽 위' 쪽에 놓고 Enter"),
        ("B", "QR을 책상 위, 화면 '오른쪽 아래' 쪽에 놓고 Enter (A에서 멀수록 좋음)"),
    ]:
        input(f"  [{label}] {hint}... ")
        # 책 옆에 그리퍼 끝을 손으로 옮겨 정확한 로봇 좌표를 얻는다
        print(f"  팔 힘을 풉니다. 그리퍼 끝을 'QR 바로 위 책상 표면'에 살짝 닿게 옮긴 뒤 Enter")
        mc.release_all_servos(); time.sleep(0.3)
        input("  옮겼으면 Enter... ")
        coords = mc.get_coords()
        if not coords or len(coords) < 6:
            print("  좌표 읽기 실패. 다시 시도하세요."); return
        x_robot, y_robot = round(coords[0], 1), round(coords[1], 1)
        z_table = round(coords[2], 1)
        print(f"    측정 로봇 좌표: ({x_robot}, {y_robot}), z(책상)={z_table}")

        # 다시 OBSERVE 자세로 복귀시켜 카메라로 QR 픽셀 측정
        print("  OBSERVE 자세로 복귀 후 카메라로 QR 픽셀 측정...")
        mc.send_angles(observe_j, 30); time.sleep(3)
        found = find_qr_stable(pipeline, align)
        if not found:
            print("  QR 검출 실패. 위치 조정 후 다시 시도하세요."); return
        cx, cy, depth_mm = found
        print(f"    측정 픽셀: ({cx:.1f}, {cy:.1f}), depth={depth_mm:.1f}mm")
        samples.append({
            "label": label,
            "cx": cx, "cy": cy, "depth": depth_mm,
            "x_robot": x_robot, "y_robot": y_robot, "z_table": z_table,
        })

    # ── ③ 평면 매핑 계산 ──
    A, B = samples
    dcx = B["cx"] - A["cx"]
    dcy = B["cy"] - A["cy"]
    dxr = B["x_robot"] - A["x_robot"]
    dyr = B["y_robot"] - A["y_robot"]
    print(f"\n[3/3] 매핑 계산")
    print(f"  Δ픽셀 = ({dcx:+.1f}, {dcy:+.1f})")
    print(f"  Δ로봇 = ({dxr:+.1f}, {dyr:+.1f}) mm")

    # 간단한 선형 매핑(축 정렬 가정).
    # 실제론 cx/cy 와 x_robot/y_robot 축이 교차할 수 있어 4개 비율을 분리해 둔다.
    MM_PER_PX_X = dxr / dcx if abs(dcx) > 1 else 0.0   # cx 변화 → x_robot mm/px
    MM_PER_PX_Y = dyr / dcy if abs(dcy) > 1 else 0.0   # cy 변화 → y_robot mm/px
    z_table = float(np.mean([A["z_table"], B["z_table"]]))
    depth_ref = float(np.mean([A["depth"], B["depth"]]))

    print("\n──────── v10_config 블록 (v10 파일에 붙여넣기) ────────")
    print(f"OBSERVE_J     = {observe_j}")
    print(f"DEPTH_REF     = {depth_ref:.1f}   # mm, OBSERVE 자세에서 책상까지")
    print(f"Z_TABLE       = {z_table:.1f}    # mm, 책상 표면 높이")
    print(f"MM_PER_PX_X   = {MM_PER_PX_X:.3f}  # cx 1px → x_robot {MM_PER_PX_X:.3f}mm")
    print(f"MM_PER_PX_Y   = {MM_PER_PX_Y:.3f}  # cy 1px → y_robot {MM_PER_PX_Y:.3f}mm")
    print(f"REF_CX        = {A['cx']:.1f}     # 기준 픽셀(샘플 A)")
    print(f"REF_CY        = {A['cy']:.1f}")
    print(f"REF_X_ROBOT   = {A['x_robot']:.1f} # 기준 로봇좌표(샘플 A)")
    print(f"REF_Y_ROBOT   = {A['y_robot']:.1f}")
    print("────────────────────────────────────────────────────────")

    pipeline.stop()
    print("\n완료. 위 블록을 v10 파일의 CALIBRATION 섹션에 그대로 붙여넣으세요.")


if __name__ == "__main__":
    main()
