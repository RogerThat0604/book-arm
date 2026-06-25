#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aruco_lib.py — 재사용 가능한 ArUco 검출 모듈

v11 본 파이프라인에서 사용. 검출 + DB 조회 + 안정화 측정을 한 곳에서 처리.

주요 함수:
  detect_markers(frame)              → 모든 마커 검출
  stable_detect(pipeline, align, ..) → 여러 프레임 누적해 안정적 검출
  best_book(detections)              → 가장 가깝거나 큰 마커 선택
"""

import time
from collections import defaultdict

import cv2
import numpy as np

from books_db import lookup_book

DICT_NAME = cv2.aruco.DICT_4X4_50
FRAME_W, FRAME_H = 640, 480

_aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_NAME)
_aruco_params = cv2.aruco.DetectorParameters()
_detector = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)


def detect_markers(frame):
    """
    한 프레임에서 모든 ArUco 마커 검출.
    반환: list of dict
        {"id": int, "cx": float, "cy": float, "size": float,
         "pts": (4,2) ndarray, "book": dict or None}
    """
    corners, ids, _ = _detector.detectMarkers(frame)
    out = []
    if ids is None:
        return out
    for marker_id, c in zip(ids.flatten(), corners):
        pts = c.reshape(4, 2)
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        size = float(np.linalg.norm(pts[0] - pts[2]))   # 대각 길이 ≈ 마커 크기
        out.append({
            "id": int(marker_id),
            "cx": cx, "cy": cy, "size": size,
            "pts": pts,
            "book": lookup_book(int(marker_id)),
        })
    return out


def stable_detect(pipeline, align, samples=15, min_hits=5):
    """
    여러 프레임 누적해 안정적인 마커 검출.
    같은 ID가 min_hits 이상 잡힌 마커만 신뢰 → 위치는 median.
    반환: list of dict (안정 검출된 마커들)
    """
    by_id = defaultdict(lambda: {"cxs": [], "cys": [], "sizes": [], "pts_last": None})

    for _ in range(samples):
        frames = align.process(pipeline.wait_for_frames()) if align else pipeline.wait_for_frames()
        cf = frames.get_color_frame()
        if not cf:
            continue
        frame = np.asanyarray(cf.get_data())
        for m in detect_markers(frame):
            d = by_id[m["id"]]
            d["cxs"].append(m["cx"])
            d["cys"].append(m["cy"])
            d["sizes"].append(m["size"])
            d["pts_last"] = m["pts"]
        time.sleep(0.04)

    results = []
    for mid, d in by_id.items():
        if len(d["cxs"]) < min_hits:
            continue
        results.append({
            "id": mid,
            "cx": float(np.median(d["cxs"])),
            "cy": float(np.median(d["cys"])),
            "size": float(np.median(d["sizes"])),
            "hits": len(d["cxs"]),
            "samples": samples,
            "pts": d["pts_last"],
            "book": lookup_book(mid),
        })
    # 큰 순서로 정렬(가까운 게 먼저)
    results.sort(key=lambda r: -r["size"])
    return results


def best_book(detections, prefer="biggest"):
    """
    여러 검출 중 하나 선택.
      prefer = "biggest"  → 가장 큰(가까운) 마커
      prefer = "centered" → 화면 중앙에 가장 가까운 마커
    DB에 없는 마커는 자동 제외.
    """
    valid = [d for d in detections if d.get("book")]
    if not valid:
        return None
    if prefer == "centered":
        cx0, cy0 = FRAME_W / 2, FRAME_H / 2
        valid.sort(key=lambda d: (d["cx"] - cx0) ** 2 + (d["cy"] - cy0) ** 2)
    else:
        valid.sort(key=lambda d: -d["size"])
    return valid[0]


def measure_depth_mm(depth_frame, cx, cy, radius=20):
    """마커 주변 깊이 median(mm). 가까운 표면만 채택해 배경 배제."""
    xs, ys = int(round(cx)), int(round(cy))
    vals = []
    for dy in range(-radius, radius + 1, 4):
        for dx in range(-radius, radius + 1, 4):
            x = min(max(xs + dx, 0), FRAME_W - 1)
            y = min(max(ys + dy, 0), FRAME_H - 1)
            d = depth_frame.get_distance(x, y)
            if 0.05 < d < 4.0:
                vals.append(d * 1000.0)
    if not vals:
        return None
    vals = np.array(vals)
    near = vals[vals <= np.percentile(vals, 20) + 30.0]
    return float(np.median(near))


if __name__ == "__main__":
    # 단독 테스트
    print("ArUco 라이브러리 모듈. v11에서 import해서 사용하세요.")
    print(f"사용 딕셔너리: DICT_4X4_50 (최대 ID 50)")
    print(f"등록 함수: detect_markers, stable_detect, best_book, measure_depth_mm")
