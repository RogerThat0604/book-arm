import cv2
import easyocr
import numpy as np

from PIL import Image, ImageDraw, ImageFont

OCR_INTERVAL = 10
OCR_WIDTH = 320
OCR_HEIGHT = 240

FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"


def put_korean_text(frame, text, position, font_size=22, color=(0, 255, 0)):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, font_size)

    # BGR -> RGB
    rgb_color = (color[2], color[1], color[0])

    draw.text(position, text, font=font, fill=rgb_color)

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


reader = easyocr.Reader(
    ['ko', 'en'],
    gpu=False
)

camera = cv2.VideoCapture('/dev/jetcocam0')

if not camera.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("OCR Start - q 키로 종료")

frame_count = 0
last_text = ""
latest_results = []

try:
    while True:
        success, frame = camera.read()

        if not success:
            continue

        frame_count += 1

        original_h, original_w = frame.shape[:2]

        scale_x = original_w / OCR_WIDTH
        scale_y = original_h / OCR_HEIGHT

        if frame_count % OCR_INTERVAL == 0:
            ocr_frame = cv2.resize(
                frame,
                (OCR_WIDTH, OCR_HEIGHT)
            )

            latest_results = reader.readtext(
                ocr_frame,
                detail=1
            )

        texts = []

        for result in latest_results:
            box = result[0]
            text = result[1]
            score = result[2]

            if score < 0.3:
                continue

            pts = []

            for x, y in box:
                pts.append(
                    (
                        int(x * scale_x),
                        int(y * scale_y)
                    )
                )

            for i in range(4):
                cv2.line(
                    frame,
                    pts[i],
                    pts[(i + 1) % 4],
                    (0, 255, 0),
                    2
                )

            x, y = pts[0]

            label = f"{text} ({score:.2f})"

            frame = put_korean_text(
                frame,
                label,
                (x, max(20, y - 30)),
                font_size=22,
                color=(0, 255, 0)
            )

            texts.append(label)

        current_text = " | ".join(texts)

        if current_text and current_text != last_text:
            print("=" * 60)
            print(current_text)
            last_text = current_text

        cv2.imshow("JetCobot EasyOCR", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n종료")

finally:
    camera.release()
    cv2.destroyAllWindows()