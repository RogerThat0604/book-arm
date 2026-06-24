import cv2
import json
import time
import threading
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image, ImageDraw, ImageFont
from pymycobot import MyCobot280

PORT = "/dev/ttyJETCOBOT"
BAUD = 1000000

CAMERA_PATH = "/dev/jetcocam0"
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

SPEED = 10
MODE_L = 1

# 목표 QR 크기: 현재 로그 기준 0.15 근처
TARGET_AREA = 0.150

# 허용 오차
TOL_X = 25
TOL_Y = 25
TOL_AREA = 0.015

# 직각 이동량(mm)
STEP_Y = 5.0
STEP_Z = 5.0
STEP_X = 8.0

# 안전 범위
X_MIN, X_MAX = 30, 230
Y_MIN, Y_MAX = -180, 100
Z_MIN, Z_MAX = 120, 410

# 방향이 반대면 True로 변경
INVERT_Y = False
INVERT_Z = False
INVERT_X = False

stop_flag = False


def keyboard_listener():
    global stop_flag
    input("\n[ENTER] 누르면 즉시 정지\n")
    stop_flag = True


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def valid_coords(coords):
    return isinstance(coords, list) and len(coords) == 6


def put_korean_text(frame, text, position, font_size=22, color=(0, 255, 0)):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, font_size)
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def parse_qr_data(data):
    try:
        book = json.loads(data)
        title = book.get("title", "")
        category = book.get("category", "")
        book_id = book.get("id", "")

        if title and category:
            return f"{title} / {category}"
        if title:
            return title
        return book_id if book_id else data

    except Exception:
        return data


def get_qr_info(frame):
    frame_h, frame_w = frame.shape[:2]
    frame_cx = frame_w // 2
    frame_cy = frame_h // 2
    frame_area = frame_w * frame_h

    codes = [code for code in decode(frame) if code.type == "QRCODE"]

    if not codes:
        return None

    code = max(codes, key=lambda c: c.rect.width * c.rect.height)

    data = code.data.decode("utf-8", errors="ignore")
    x, y, w, h = code.rect

    if code.polygon and len(code.polygon) >= 4:
        pts = [(p.x, p.y) for p in code.polygon]
        cx = int(sum(px for px, py in pts) / len(pts))
        cy = int(sum(py for px, py in pts) / len(pts))
    else:
        pts = None
        cx = x + w // 2
        cy = y + h // 2

    return {
        "data": data,
        "label": parse_qr_data(data),
        "rect": (x, y, w, h),
        "pts": pts,
        "center": (cx, cy),
        "frame_center": (frame_cx, frame_cy),
        "ex": cx - frame_cx,
        "ey": cy - frame_cy,
        "area": (w * h) / frame_area,
    }


def draw_info(frame, info):
    frame_cx, frame_cy = info["frame_center"]
    cx, cy = info["center"]
    x, y, w, h = info["rect"]

    cv2.circle(frame, (frame_cx, frame_cy), 6, (255, 0, 0), -1)
    cv2.line(frame, (frame_cx - 25, frame_cy), (frame_cx + 25, frame_cy), (255, 0, 0), 2)
    cv2.line(frame, (frame_cx, frame_cy - 25), (frame_cx, frame_cy + 25), (255, 0, 0), 2)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if info["pts"]:
        pts = info["pts"]
        for i in range(len(pts)):
            cv2.line(frame, pts[i], pts[(i + 1) % len(pts)], (255, 0, 0), 2)

    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
    cv2.line(frame, (frame_cx, frame_cy), (cx, cy), (0, 255, 255), 2)

    frame = put_korean_text(
        frame,
        f"QR: {info['label']}",
        (x, max(20, y - 35)),
        22,
        (0, 255, 0)
    )

    cv2.putText(
        frame,
        f"ex={info['ex']}, ey={info['ey']}, area={info['area']:.4f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    return frame


def move_once(mc, target):
    mc.send_coords(target, SPEED, MODE_L)
    time.sleep(0.6)


mc = MyCobot280(PORT, BAUD)
mc.thread_lock = True

time.sleep(1)
mc.power_on()
time.sleep(1)

mc.set_fresh_mode(0)
time.sleep(0.2)

home = mc.get_coords()
print("home:", home)

if not valid_coords(home):
    print("좌표 읽기 실패")
    exit()

fixed_rx = home[3]
fixed_ry = home[4]
fixed_rz = home[5]

current = home.copy()

camera = cv2.VideoCapture(CAMERA_PATH)

if not camera.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

threading.Thread(target=keyboard_listener, daemon=True).start()

print("\nQR 직각 이동 정렬 테스트 시작")
print("동작 순서: Y 좌우 정렬 → Z 상하 정렬 → X 거리 정렬")

try:
    while not stop_flag:
        success, frame = camera.read()

        if not success:
            print("프레임을 읽을 수 없습니다.")
            continue

        info = get_qr_info(frame)

        if info is None:
            cv2.putText(
                frame,
                "No QR",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.imshow("QR Robot Align", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            continue

        frame = draw_info(frame, info)

        ex = info["ex"]
        ey = info["ey"]
        area = info["area"]

        print(f"ex={ex}, ey={ey}, area={area:.4f}")

        target = current.copy()
        moved = False

        # 1단계: 좌우 정렬
        if abs(ex) > TOL_X:
            direction = 1 if ex > 0 else -1

            if INVERT_Y:
                direction *= -1

            target[1] += direction * STEP_Y
            moved = True
            print("[1] Y축 좌우 정렬")

        # 2단계: 상하 정렬
        elif abs(ey) > TOL_Y:
            direction = -1 if ey > 0 else 1

            if INVERT_Z:
                direction *= -1

            target[2] += direction * STEP_Z
            moved = True
            print("[2] Z축 상하 정렬")

        # 3단계: 거리 정렬
        elif abs(TARGET_AREA - area) > TOL_AREA:
            direction = 1 if area < TARGET_AREA else -1

            if INVERT_X:
                direction *= -1

            target[0] += direction * STEP_X
            moved = True
            print("[3] X축 거리 정렬")

        else:
            print("\n정렬 완료")
            cv2.putText(
                frame,
                "ALIGNED",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )
            cv2.imshow("QR Robot Align", frame)
            cv2.waitKey(800)
            break

        if moved:
            target[0] = clamp(target[0], X_MIN, X_MAX)
            target[1] = clamp(target[1], Y_MIN, Y_MAX)
            target[2] = clamp(target[2], Z_MIN, Z_MAX)

            # 카메라 방향 고정
            target[3] = fixed_rx
            target[4] = fixed_ry
            target[5] = fixed_rz

            print("target:", target)

            move_once(mc, target)
            current = target.copy()

        cv2.imshow("QR Robot Align", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    stop_flag = True
    mc.stop()
    camera.release()
    cv2.destroyAllWindows()

    print("\n최종 좌표:")
    print(mc.get_coords())
    print("종료")