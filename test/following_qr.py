import cv2
import json
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image, ImageDraw, ImageFont

CAMERA_PATH = "/dev/jetcocam0"
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"


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


camera = cv2.VideoCapture(CAMERA_PATH)

if not camera.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("QR Align Value Test Start - q 키로 종료")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("프레임을 읽을 수 없습니다.")
            continue

        frame_h, frame_w = frame.shape[:2]
        frame_cx = frame_w // 2
        frame_cy = frame_h // 2
        frame_area = frame_w * frame_h

        cv2.circle(frame, (frame_cx, frame_cy), 6, (255, 0, 0), -1)
        cv2.line(frame, (frame_cx - 20, frame_cy), (frame_cx + 20, frame_cy), (255, 0, 0), 2)
        cv2.line(frame, (frame_cx, frame_cy - 20), (frame_cx, frame_cy + 20), (255, 0, 0), 2)

        codes = [code for code in decode(frame) if code.type == "QRCODE"]

        if codes:
            code = max(codes, key=lambda c: c.rect.width * c.rect.height)

            data = code.data.decode("utf-8", errors="ignore")
            x, y, w, h = code.rect

            if code.polygon and len(code.polygon) >= 4:
                pts = [(p.x, p.y) for p in code.polygon]

                for i in range(len(pts)):
                    cv2.line(frame, pts[i], pts[(i + 1) % len(pts)], (255, 0, 0), 2)

                cx = int(sum(px for px, py in pts) / len(pts))
                cy = int(sum(py for px, py in pts) / len(pts))
            else:
                cx = x + w // 2
                cy = y + h // 2

            ex = cx - frame_cx
            ey = cy - frame_cy
            area_ratio = (w * h) / frame_area

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.line(frame, (frame_cx, frame_cy), (cx, cy), (0, 255, 255), 2)

            label = parse_qr_data(data)

            frame = put_korean_text(
                frame,
                f"QR: {label}",
                (x, max(20, y - 35)),
                font_size=22,
                color=(0, 255, 0)
            )

            cv2.putText(
                frame,
                f"ex={ex}, ey={ey}, area={area_ratio:.4f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            print(f"data={data}, ex={ex}, ey={ey}, area={area_ratio:.4f}")

        else:
            cv2.putText(
                frame,
                "No QR",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("QR Align Value Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()