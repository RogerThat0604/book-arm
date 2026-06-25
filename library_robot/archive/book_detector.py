import cv2
import numpy as np

cap = cv2.VideoCapture(6)

if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

ret, frame = cap.read()

if not ret:
    print("프레임 읽기 실패")
    exit()

h, w = frame.shape[:2]

roi_x1 = int(w * 0.15)
roi_y1 = int(h * 0.25)
roi_x2 = int(w * 0.85)
roi_y2 = int(h * 0.95)

# ROI
roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

# 흰색/밝은 물체 마스크
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

lower = np.array([0, 0, 120])
upper = np.array([180, 80, 255])

mask = cv2.inRange(hsv, lower, upper)

# 노이즈 제거
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

best = None
best_area = 0

for cnt in contours:
    area = cv2.contourArea(cnt)

    if area < 300:
        continue

    x, y, bw, bh = cv2.boundingRect(cnt)
    ratio = bw / float(bh)

    # 책/표지 후보: 너무 얇거나 너무 길면 제외
    if 0.3 < ratio < 3.5 and area > best_area:
        best = (x, y, bw, bh)
        best_area = area

# 중앙 십자선
cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 2)
cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 0), 2)

# ROI 표시
cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 255), 2)

if best:
    x, y, bw, bh = best

    x1 = roi_x1 + x
    y1 = roi_y1 + y
    x2 = x1 + bw
    y2 = y1 + bh

    cx = x1 + bw // 2
    cy = y1 + bh // 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

    dx = cx - (w // 2)
    dy = cy - (h // 2)

    print(f"책 감지: cx={cx}, cy={cy}, dx={dx}, dy={dy}, area={best_area}")
else:
    print("책 감지 실패")

cv2.imwrite("/home/jetcobot/book_detect_result.jpg", frame)
print("저장 완료: /home/jetcobot/book_detect_result.jpg")

cap.release()