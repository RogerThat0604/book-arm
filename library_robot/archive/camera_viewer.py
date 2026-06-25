#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_viewer.py — 캘리브 도우미: 카메라 화면 실시간 보기

기능:
  - OBSERVE 자세로 팔 보내고 그 상태에서 카메라 화면 표시
  - QR 박스 + 디코딩 텍스트 오버레이
  - 화면 중앙선 표시
  - 'q' 키로 종료

키 조작:
  q : 종료
  s : 현재 화면 스냅샷 저장 (camera_snap.png)
  r : 팔 힘 풀기/다시 굳히기 토글 (자세 조정용)

사용 흐름:
  1) 이 스크립트 실행 → 카메라 창 뜸
  2) 'r' 눌러서 팔 힘 풀고 자세 조정 (책상 위에서 한 손으로 받치며)
  3) 만족스러우면 'r' 다시 눌러 굳히기
  4) 화면 보면서 책 위치, 시야, QR 크기 확인
  5) 'q' 종료 후 mc.get_angles()로 자세 기록
"""

import time
import cv2
import numpy as np
import pyrealsense2 as rs
from pymycobot import MyCobot280 as MyCobot

# 이 자세에서 시작 (현재 OBSERVE_J)
START_J = [2.5, -14.1, -0.9, -44.0, -4.8, 6.0]

PORT = "/dev/ttyJETCOBOT"; BAUD = 1000000
FRAME_W, FRAME_H = 640, 480

mc = MyCobot(PORT, BAUD); mc.thread_lock = True; mc.power_on(); time.sleep(1)
print(f"시작 자세로 이동: {START_J}")
mc.send_angles(START_J, 25); time.sleep(3)

pipeline = rs.pipeline(); config = rs.config()
config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
pipeline.start(config); time.sleep(1)
print("카메라 시작. q=종료, s=스냅샷, r=팔 힘 풀기/굳히기")

try:
    from pyzbar.pyzbar import decode as zbar_decode
    has_zbar = True
except ImportError:
    qr = cv2.QRCodeDetector()
    has_zbar = False

released = False

while True:
    frames = pipeline.wait_for_frames()
    c = frames.get_color_frame()
    if not c: continue
    frame = np.asanyarray(c.get_data())

    # QR 검출 + 오버레이
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if has_zbar:
        results = zbar_decode(gray)
        for r in results:
            pts = np.array([[p.x, p.y] for p in r.polygon], dtype=int)
            if len(pts) >= 4:
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            text = r.data.decode("utf-8", errors="ignore")
            x, y, w, h = r.rect.left, r.rect.top, r.rect.width, r.rect.height
            cv2.putText(frame, f"'{text}' ({w}x{h}px)",
                        (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
    else:
        ok, pts = qr.detect(gray)
        if ok and pts is not None:
            p = pts.reshape(-1, 2).astype(int)
            cv2.polylines(frame, [p], True, (0, 255, 0), 2)

    # 중앙선 + 안내
    cv2.line(frame, (FRAME_W // 2, 0), (FRAME_W // 2, FRAME_H), (100, 100, 100), 1)
    cv2.line(frame, (0, FRAME_H // 2), (FRAME_W, FRAME_H // 2), (100, 100, 100), 1)
    status = "RELEASED (자세 조정 가능)" if released else "HELD"
    color = (0, 165, 255) if released else (255, 255, 255)
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, "q: quit  s: snap  r: release/hold",
                (10, FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("camera", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("camera_snap.png", frame)
        print("저장: camera_snap.png")
    elif key == ord('r'):
        released = not released
        if released:
            print("팔 힘 풀림. 손으로 받치고 자세 조정하세요")
            mc.release_all_servos()
        else:
            ang = mc.get_angles()
            if ang and len(ang) == 6:
                print(f"현재 자세 굳히기: {[round(a, 1) for a in ang]}")
                mc.send_angles(ang, 30); time.sleep(1.5)

pipeline.stop()
cv2.destroyAllWindows()

final = mc.get_angles()
if final:
    print(f"\n최종 자세 (OBSERVE_J로 사용 가능):\n  {[round(a, 1) for a in final]}")
