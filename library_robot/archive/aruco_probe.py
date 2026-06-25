#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aruco_probe.py — ArUco 검출 + DB 조회 진단 도구

기능:
  - RealSense 카메라로 ArUco 마커 검출
  - 검출된 ID마다 DB 조회 → 책 제목/카테고리 표시
  - 60프레임 통계 출력 + 스냅샷 저장
  - 'q' 종료, 's' 스냅샷, 'l' 라이브 종료 후 통계 출력

사용:
  python aruco_probe.py

사전:
  - books_db_init.py 로 DB와 마커 생성됨
  - 마커가 인쇄돼 책에 붙여져 있음(또는 화면에 띄워둠)
"""

import time
from collections import Counter

import cv2
import numpy as np
import pyrealsense2 as rs

from books_db import lookup_book

DICT_NAME = cv2.aruco.DICT_4X4_50
FRAME_W, FRAME_H = 640, 480


def detect_markers(frame, detector):
    """반환: list of (id, center_cxcy, size_px)"""
    corners, ids, _ = detector.detectMarkers(frame)
    out = []
    if ids is None:
        return out
    for marker_id, c in zip(ids.flatten(), corners):
        pts = c.reshape(4, 2)
        cx = float(pts[:, 0].mean()); cy = float(pts[:, 1].mean())
        size = float(np.linalg.norm(pts[0] - pts[2]))  # 대각 길이 ≈ 마커 크기
        out.append((int(marker_id), (cx, cy), size, pts.astype(int)))
    return out


def main():
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_NAME)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
    pipeline.start(config)
    time.sleep(1)
    print("ArUco 검출 시작. q=종료, s=스냅샷")

    seen = Counter()
    sizes = []
    N = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            cf = frames.get_color_frame()
            if not cf:
                continue
            frame = np.asanyarray(cf.get_data())
            N += 1

            results = detect_markers(frame, detector)
            for mid, (cx, cy), size, pts in results:
                seen[mid] += 1
                sizes.append(size)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                # DB 조회
                info = lookup_book(mid)
                if info:
                    label = f"ID:{mid} {info['title']}/{info['category']}"
                else:
                    label = f"ID:{mid} (DB 미등록)"
                cv2.putText(frame, label, (pts[0][0], pts[0][1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"{int(size)}px",
                            (int(cx) + 8, int(cy) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.line(frame, (FRAME_W // 2, 0), (FRAME_W // 2, FRAME_H), (80, 80, 80), 1)
            cv2.line(frame, (0, FRAME_H // 2), (FRAME_W, FRAME_H // 2), (80, 80, 80), 1)
            cv2.putText(frame, f"frames={N}  unique={len(seen)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("aruco_probe", frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("aruco_snap.png", frame)
                print("저장: aruco_snap.png")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    # ── 통계 ──
    print("\n" + "─" * 50)
    print(f"총 프레임: {N}")
    print(f"발견된 마커 ID: {sorted(seen.keys())}")
    for mid in sorted(seen.keys()):
        info = lookup_book(mid)
        title = f"{info['title']} ({info['category']})" if info else "(DB 미등록)"
        print(f"  ID {mid:3d}: {seen[mid]:4d}회 검출 ({seen[mid]*100//max(N,1):3d}%)  → {title}")
    if sizes:
        print(f"마커 크기 px : median {int(np.median(sizes))}, "
              f"min {int(min(sizes))}, max {int(max(sizes))}")
    print("─" * 50)
    if sizes and np.median(sizes) < 40:
        print("⚠ 마커가 작음 → 더 크게 인쇄하거나 가까이 두세요")
    elif not seen:
        print("⚠ 마커가 전혀 안 잡힘 → 인쇄 품질/조명/딕셔너리 확인")
    else:
        print("✅ 검출 양호. v11 파이프라인 진행 가능")


if __name__ == "__main__":
    main()
