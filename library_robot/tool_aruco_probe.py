#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aruco_probe_web.py — ArUco 검출 진단 (웹 버전, GUI 불필요)

기능:
  - RealSense 카메라로 ArUco 검출
  - 검출된 ID마다 DB 조회 → 책 제목/카테고리 표시
  - 누적 통계 실시간 표시

사용:
  python aruco_probe_web.py
  → 브라우저에서 http://<라즈베리IP>:5001  접속

의존성: pip install flask
"""

import io
import time
import threading
from collections import Counter

import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, render_template_string, jsonify, send_file

from books_db import lookup_book
from hangul_draw import put_hangul

DICT_NAME = cv2.aruco.DICT_6X6_250
FRAME_W, FRAME_H = 640, 480
PORT_WEB = 5001

print("RealSense 시작...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
pipeline.start(config)
time.sleep(1)
print("카메라 시작")

aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_NAME)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

state = {
    "seen": Counter(),
    "sizes": [],
    "frames": 0,
    "last_results": [],
    "lock": threading.Lock(),
}


def detect_and_draw(frame):
    corners, ids, _ = detector.detectMarkers(frame)
    results = []
    if ids is not None:
        for marker_id, c in zip(ids.flatten(), corners):
            pts = c.reshape(4, 2)
            cx = float(pts[:, 0].mean()); cy = float(pts[:, 1].mean())
            size = float(np.linalg.norm(pts[0] - pts[2]))
            results.append({
                "id": int(marker_id),
                "cx": cx, "cy": cy, "size": size,
                "pts": pts.astype(int),
            })

    for r in results:
        info = lookup_book(r["id"])
        cv2.polylines(frame, [r["pts"]], True, (0, 255, 0), 2)
        if info:
            label = f"ID:{r['id']} {info['title']} [{info['category']}]"
        else:
            label = f"ID:{r['id']} (DB 미등록)"
        # 한글 그리기 (배경 박스 포함, 가독성↑)
        label_pos = (int(r["pts"][0][0]), max(int(r["pts"][0][1]) - 28, 5))
        frame = put_hangul(frame, label, label_pos, size=18,
                           color=(255, 255, 255), bg=(0, 120, 0))
        cv2.circle(frame, (int(r["cx"]), int(r["cy"])), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"{int(r['size'])}px",
                    (int(r["cx"]) + 8, int(r["cy"]) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.line(frame, (FRAME_W // 2, 0), (FRAME_W // 2, FRAME_H), (80, 80, 80), 1)
    cv2.line(frame, (0, FRAME_H // 2), (FRAME_W, FRAME_H // 2), (80, 80, 80), 1)

    # 누적 통계 갱신
    with state["lock"]:
        state["frames"] += 1
        for r in results:
            state["seen"][r["id"]] += 1
            state["sizes"].append(r["size"])
            if len(state["sizes"]) > 500:
                state["sizes"] = state["sizes"][-500:]
        state["last_results"] = [
            {"id": r["id"], "size": int(r["size"])} for r in results
        ]

    return frame


def make_frame():
    frames = pipeline.wait_for_frames()
    cf = frames.get_color_frame()
    if not cf:
        return None
    frame = np.asanyarray(cf.get_data())
    return detect_and_draw(frame)


def mjpeg_gen():
    while True:
        f = make_frame()
        if f is None:
            time.sleep(0.05); continue
        ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + jpg.tobytes() + b"\r\n")


app = Flask(__name__)

PAGE = """
<!doctype html><html><head><meta charset="utf-8"><title>ArUco Probe</title>
<style>
body{background:#111;color:#eee;font-family:sans-serif;text-align:center;margin:18px}
img{max-width:100%;border:1px solid #444}
#stats{margin:10px auto;max-width:640px;text-align:left;background:#1a1a1a;
       padding:14px;border-radius:8px;font-family:monospace;font-size:14px}
.row{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #333}
.now{color:#4f4;margin-top:8px;font-weight:bold}
button{padding:8px 16px;margin:6px;background:#08c;color:#fff;border:none;border-radius:6px;cursor:pointer}
</style></head>
<body>
<h2>ArUco Detection Probe</h2>
<img src="/stream">
<div><button onclick="reset()">통계 리셋</button></div>
<div id="stats">로딩...</div>
<script>
function fmt(s){return Object.entries(s).map(([k,v])=>`<div class="row"><span>ID ${k}</span><span>${v.title||'(DB 미등록)'}</span><span>${v.count}회 (${v.pct}%)</span></div>`).join('');}
function refresh(){
  fetch('/stats').then(r=>r.json()).then(d=>{
    let html = `<div class="row"><b>총 프레임</b><span>${d.frames}</span></div>`;
    html += `<div class="row"><b>마커 크기(px)</b><span>median ${d.size_median}</span></div>`;
    html += fmt(d.seen);
    if(d.now.length>0){
      html += '<div class="now">지금 보이는 마커: ' + d.now.map(x=>`ID ${x.id} (${x.size}px)`).join(', ') + '</div>';
    } else {
      html += '<div class="now" style="color:#888">지금 보이는 마커: 없음</div>';
    }
    document.getElementById('stats').innerHTML = html;
  });
}
function reset(){ fetch('/reset',{method:'POST'}).then(refresh); }
setInterval(refresh, 500); refresh();
</script>
</body></html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/stream")
def stream():
    return Response(mjpeg_gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    with state["lock"]:
        N = max(state["frames"], 1)
        seen_dict = {}
        for mid, cnt in sorted(state["seen"].items()):
            info = lookup_book(mid)
            seen_dict[mid] = {
                "title": f"{info['title']} [{info['category']}]" if info else None,
                "count": cnt,
                "pct": cnt * 100 // N,
            }
        size_med = int(np.median(state["sizes"])) if state["sizes"] else 0
        return jsonify({
            "frames": state["frames"],
            "seen": seen_dict,
            "size_median": size_med,
            "now": state["last_results"],
        })


@app.route("/reset", methods=["POST"])
def reset():
    with state["lock"]:
        state["seen"].clear()
        state["sizes"].clear()
        state["frames"] = 0
    return jsonify({"ok": True})


if __name__ == "__main__":
    print(f"브라우저에서 접속: http://<라즈베리IP>:{PORT_WEB}")
    print("  IP 확인: hostname -I")
    try:
        app.run(host="0.0.0.0", port=PORT_WEB, threaded=True, debug=False)
    finally:
        pipeline.stop()
