#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_web.py — 카메라 화면을 웹브라우저로 스트리밍

사용:
  1) 라즈베리파이에서 실행: python camera_web.py
  2) 같은 네트워크의 PC/폰 브라우저에서 접속:
        http://<라즈베리파이IP>:5000
     (IP 모르면 라즈베리에서 'hostname -I' 명령으로 확인)

브라우저 화면:
  - 실시간 카메라 영상 (QR 박스 + 디코딩 텍스트 오버레이)
  - 팔 힘 풀기/굳히기 버튼
  - 현재 관절각 표시
  - 스냅샷 다운로드 버튼

의존성: pip install flask
"""

import io
import time
import threading
import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, render_template_string, jsonify, send_file
from pymycobot import MyCobot280 as MyCobot

START_J = [2.5, -14.1, -0.9, -44.0, -4.8, 6.0]
PORT_MC = "/dev/ttyJETCOBOT"; BAUD = 1000000
FRAME_W, FRAME_H = 640, 480
PORT_WEB = 5000

# ── 로봇/카메라 초기화 ──
print("로봇 연결...")
mc = MyCobot(PORT_MC, BAUD); mc.thread_lock = True; mc.power_on(); time.sleep(1)
print(f"시작 자세로 이동: {START_J}")
mc.send_angles(START_J, 25); time.sleep(3)

pipeline = rs.pipeline(); config = rs.config()
config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
pipeline.start(config); time.sleep(1)
print("카메라 시작")

try:
    from pyzbar.pyzbar import decode as zbar_decode
    has_zbar = True
except ImportError:
    qr_det = cv2.QRCodeDetector()
    has_zbar = False

state = {"released": False, "lock": threading.Lock()}


def make_frame():
    frames = pipeline.wait_for_frames()
    c = frames.get_color_frame()
    if not c:
        return None
    frame = np.asanyarray(c.get_data())

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if has_zbar:
        for r in zbar_decode(gray):
            pts = np.array([[p.x, p.y] for p in r.polygon], dtype=int)
            if len(pts) >= 4:
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            text = r.data.decode("utf-8", errors="ignore")
            x, y, w, h = r.rect.left, r.rect.top, r.rect.width, r.rect.height
            cv2.putText(frame, f"'{text}' ({w}x{h}px)",
                        (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
    else:
        ok, pts = qr_det.detect(gray)
        if ok and pts is not None:
            p = pts.reshape(-1, 2).astype(int)
            cv2.polylines(frame, [p], True, (0, 255, 0), 2)

    cv2.line(frame, (FRAME_W // 2, 0), (FRAME_W // 2, FRAME_H), (100, 100, 100), 1)
    cv2.line(frame, (0, FRAME_H // 2), (FRAME_W, FRAME_H // 2), (100, 100, 100), 1)
    status = "RELEASED" if state["released"] else "HELD"
    color = (0, 165, 255) if state["released"] else (255, 255, 255)
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def mjpeg_gen():
    while True:
        f = make_frame()
        if f is None:
            time.sleep(0.05); continue
        ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")


app = Flask(__name__)

PAGE = """
<!doctype html><html><head><meta charset="utf-8"><title>Cobot Camera</title>
<style>
body{background:#111;color:#eee;font-family:sans-serif;text-align:center;margin:20px}
img{max-width:100%;border:1px solid #444}
button{padding:10px 20px;margin:6px;font-size:16px;cursor:pointer;border:none;border-radius:6px}
.release{background:#c80;color:#fff} .hold{background:#0a0;color:#fff}
.snap{background:#08c;color:#fff}
#angles{font-family:monospace;background:#222;padding:8px;border-radius:6px;display:inline-block;margin-top:8px}
.status{margin:10px;font-weight:bold}
.released{color:#fa0} .held{color:#0f0}
</style></head>
<body>
<h2>JetCobot Camera</h2>
<img src="/stream" id="stream">
<div>
  <button class="release" onclick="toggle()">팔 힘 풀기/굳히기</button>
  <button class="snap" onclick="snap()">스냅샷</button>
</div>
<div class="status" id="status">상태: -</div>
<div id="angles">관절각: -</div>
<script>
function refresh(){
  fetch('/state').then(r=>r.json()).then(d=>{
    const s = document.getElementById('status');
    s.textContent = '상태: ' + (d.released ? 'RELEASED (자세 조정 가능)' : 'HELD');
    s.className = 'status ' + (d.released ? 'released' : 'held');
    document.getElementById('angles').textContent = '관절각: ' + JSON.stringify(d.angles);
  });
}
function toggle(){ fetch('/toggle',{method:'POST'}).then(refresh); }
function snap(){ window.open('/snap','_blank'); }
setInterval(refresh, 700); refresh();
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


@app.route("/state")
def get_state():
    ang = mc.get_angles()
    ang = [round(a, 1) for a in ang] if ang else None
    return jsonify({"released": state["released"], "angles": ang})


@app.route("/toggle", methods=["POST"])
def toggle():
    with state["lock"]:
        if state["released"]:
            ang = mc.get_angles()
            if ang and len(ang) == 6:
                mc.send_angles(ang, 30); time.sleep(0.5)
            state["released"] = False
        else:
            mc.release_all_servos()
            state["released"] = True
    return jsonify({"released": state["released"]})


@app.route("/snap")
def snap():
    f = make_frame()
    if f is None:
        return "no frame", 500
    ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return send_file(io.BytesIO(jpg.tobytes()),
                     mimetype="image/jpeg",
                     download_name="camera_snap.jpg")


if __name__ == "__main__":
    print(f"브라우저에서 접속: http://<라즈베리파이IP>:{PORT_WEB}")
    print("  IP 확인: 라즈베리에서 'hostname -I'")
    try:
        app.run(host="0.0.0.0", port=PORT_WEB, threaded=True, debug=False)
    finally:
        pipeline.stop()
        print("종료")
