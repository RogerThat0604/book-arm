import cv2

cap = cv2.VideoCapture(6)

if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

ret, frame = cap.read()

if not ret:
    print("프레임 읽기 실패")
    exit()

h, w = frame.shape[:2]

# 중앙 십자선
cv2.line(frame, (w//2, 0), (w//2, h), (0, 255, 0), 2)
cv2.line(frame, (0, h//2), (w, h//2), (0, 255, 0), 2)

# 임시 책 영역: 화면 중앙 박스
box_w = int(w * 0.4)
box_h = int(h * 0.4)

x1 = w//2 - box_w//2
y1 = h//2 - box_h//2
x2 = w//2 + box_w//2
y2 = h//2 + box_h//2

cx = (x1 + x2) // 2
cy = (y1 + y2) // 2

cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

print(f"책 중심 후보: x={cx}, y={cy}")
cv2.imwrite("/home/jetcobot/book_center_test.jpg", frame)
print("저장 완료: /home/jetcobot/book_center_test.jpg")

cap.release()
