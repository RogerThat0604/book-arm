#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qr_probe.py — QR 검출 진단 도구 (정렬 전에 먼저 실행)

60프레임 동안 두 방식의 검출 성공률을 비교한다.
  - detect()           : 위치만 (정렬에 필요한 것)
  - detectAndDecode()  : 위치 + 내용 디코딩 (어려움)
중간에 주석 달린 스냅샷(qr_probe.png)을 저장해서
QR이 화면에 제대로/충분히 크게 보이는지 눈으로 확인한다.

실행:  python qr_probe.py
확인:  qr_probe.png 를 열어볼 것
"""

import time
import cv2
import numpy as np
import pyrealsense2 as rs

N = 60
W, H = 640, 480

qr = cv2.QRCodeDetector()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
pipeline.start(config)
time.sleep(1)
print("RealSense 시작. 60프레임 진단 시작...")

det_hits = 0      # detect() 성공
dec_hits = 0      # detectAndDecode() 성공(내용까지)
snap_saved = False

try:
    for i in range(N):
        frames = pipeline.wait_for_frames()
        cf = frames.get_color_frame()
        if not cf:
            continue
        frame = np.asanyarray(cf.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)

        ok, pts = qr.detect(gray)
        data, _, _ = qr.detectAndDecode(gray)

        if ok and pts is not None:
            det_hits += 1
        if data:
            dec_hits += 1

        # 중간 프레임에 검출되면 주석 스냅샷 저장
        if not snap_saved and ok and pts is not None:
            vis = frame.copy()
            cv2.line(vis, (W // 2, 0), (W // 2, H), (0, 255, 0), 1)  # 화면 중앙선
            p = pts.reshape(-1, 2).astype(int)
            cv2.polylines(vis, [p], True, (0, 0, 255), 2)            # QR 박스
            cx = int(np.mean(p[:, 0]))
            dx = cx - W // 2
            cv2.putText(vis, f"dx={dx}  data='{data}'", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imwrite("qr_probe.png", vis)
            snap_saved = True
            print(f"  스냅샷 저장: qr_probe.png  (dx={dx})")
finally:
    pipeline.stop()

print("─" * 40)
print(f"detect()          성공률: {det_hits}/{N}  ({det_hits*100//N}%)")
print(f"detectAndDecode() 성공률: {dec_hits}/{N}  ({dec_hits*100//N}%)")
print("─" * 40)
if det_hits < N * 0.5:
    print("⚠ detect() 성공률이 낮음 → QR을 더 크게/가깝게/밝게,")
    print("  화면 중앙 쪽에 오도록 배치하세요. qr_probe.png 확인.")
elif dec_hits < det_hits * 0.5:
    print("→ 위치는 잘 잡힘. 정렬은 detect() 방식으로 충분합니다(권장).")
else:
    print("→ 검출 양호. v2 스크립트로 정렬 진행하세요.")
