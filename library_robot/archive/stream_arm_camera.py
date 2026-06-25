import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer

cap = cv2.VideoCapture(6)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]
            cv2.line(frame, (w//2, 0), (w//2, h), (0, 255, 0), 2)
            cv2.line(frame, (0, h//2), (w, h//2), (0, 255, 0), 2)

            _, jpg = cv2.imencode('.jpg', frame)

            self.wfile.write(b'--frame\r\n')
            self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
            self.wfile.write(jpg.tobytes())
            self.wfile.write(b'\r\n')

print("카메라 스트림 시작: http://192.168.0.37:8080")
server = HTTPServer(('0.0.0.0', 8080), Handler)
server.serve_forever()
