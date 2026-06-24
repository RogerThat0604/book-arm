import cv2
import time
from pymycobot.mycobot280 import MyCobot280

CAMERA_INDEX = 6

CENTER_THRESHOLD = 35
KP = 0.03
MAX_MOVE = 5

HOME_ANGLES = [0, 0, 0, 0, 0, 0]
SPEED = 20

mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
time.sleep(1)

mc.send_angles(HOME_ANGLES, SPEED)
time.sleep(2)

base_angle = 0

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

def detect_book_dx(frame):
    h, w, _ = frame.shape
    frame_center_x = w // 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh

        if area > 3000 and bh > 40 and bw > 30:
            candidates.append((x, y, bw, bh, area))

    if not candidates:
        return None

    x, y, bw, bh, area = max(candidates, key=lambda item: item[4])
    book_center_x = x + bw // 2

    dx = book_center_x - frame_center_x
    return dx

print("Visual Servoing 시작")

while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임 읽기 실패")
        continue

    dx = detect_book_dx(frame)

    if dx is None:
        print("책 감지 안 됨")
        time.sleep(0.3)
        continue

    print(f"dx={dx}")

    if abs(dx) < CENTER_THRESHOLD:
        print("중앙 정렬 완료")
        break

    move = int(dx * KP)
    move = max(min(move, MAX_MOVE), -MAX_MOVE)

    if move == 0:
        move = 1 if dx > 0 else -1

    if dx < 0:
        print("왼쪽 보정")
    else:
        print("오른쪽 보정")

    base_angle = base_angle - move

    mc.send_angles([base_angle, 0, 0, 0, 0, 0], SPEED)
    time.sleep(0.7)

cap.release()

print("Step 1 완료")