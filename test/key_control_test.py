import cv2
import numpy as np
import time

from pymycobot.mycobot280 import MyCobot280
from pymycobot.genre import Angle

# =========================
# JetCobot 연결
# =========================
mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
mc.thread_lock = True

print("로봇이 연결되었습니다.")

# =========================
# 설정값
# =========================
J1_ID = Angle.J1.value
J4_ID = Angle.J4.value

J1_MIN, J1_MAX = -168, 168
J4_MIN, J4_MAX = -145, 145

STEP = 2
SPEED = 30

angles = mc.get_angles()
print("현재 각도:", angles)

j1 = angles[0]
j4 = angles[3]

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))

def send_j1_j4():
    global j1, j4

    j1 = clamp(j1, J1_MIN, J1_MAX)
    j4 = clamp(j4, J4_MIN, J4_MAX)

    mc.send_angle(J1_ID, j1, SPEED)
    time.sleep(0.05)
    mc.send_angle(J4_ID, j4, SPEED)

    print(f"J1: {j1:.1f}, J4: {j4:.1f}")

# =========================
# OpenCV 키 입력 창
# =========================
while True:
    img = np.ones((430, 720, 3), dtype=np.uint8) * 255

    cv2.putText(img, "JetCobot J1 / J4 Manual Control",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 2)

    cv2.putText(img, f"J1 Left/Right : {j1:.1f}",
                (30, 120), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)

    cv2.putText(img, f"J4 Up/Down    : {j4:.1f}",
                (30, 170), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)

    cv2.putText(img, "A : J1 Left",
                (30, 240), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), 2)

    cv2.putText(img, "D : J1 Right",
                (30, 285), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), 2)

    cv2.putText(img, "W : J4 Up",
                (30, 330), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), 2)

    cv2.putText(img, "S : J4 Down",
                (30, 375), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), 2)

    cv2.putText(img, "R : Read angles    Q : Quit",
                (30, 415), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 0, 0), 2)

    cv2.imshow("JetCobot Control", img)

    key = cv2.waitKey(50) & 0xFF

    if key == ord('a'):
        j1 += STEP
        send_j1_j4()

    elif key == ord('d'):
        j1 -= STEP
        send_j1_j4()

    elif key == ord('w'):
        j4 += STEP
        send_j1_j4()

    elif key == ord('s'):
        j4 -= STEP
        send_j1_j4()

    elif key == ord('r'):
        angles = mc.get_angles()
        print("현재 각도:", angles)
        j1 = angles[0]
        j4 = angles[3]

    elif key == ord('q'):
        print("종료합니다.")
        mc.stop()
        break

cv2.destroyAllWindows()
