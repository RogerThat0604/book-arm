import cv2
import easyocr

OCR_INTERVAL = 3

# 한국어 + 영어
reader = easyocr.Reader(
    ['ko', 'en'],
    gpu=False
)

camera = cv2.VideoCapture('/dev/jetcocam0')

if not camera.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("OCR Start (Ctrl+C 종료)")

frame_count = 0
last_text = ""

try:
    while True:

        success, frame = camera.read()

        if not success:
            continue

        frame_count += 1

        # 3프레임마다 OCR 수행
        if frame_count % OCR_INTERVAL != 0:
            continue

        # 속도 향상을 위해 축소
        frame = cv2.resize(
            frame,
            (480, 360)
        )

        results = reader.readtext(
            frame,
            detail=1
        )

        texts = []

        for result in results:

            text = result[1]
            score = result[2]

            if score < 0.3:
                continue

            texts.append(
                f"{text} ({score:.2f})"
            )

        current_text = " | ".join(texts)

        # 같은 결과 반복 출력 방지
        if current_text and current_text != last_text:

            print("=" * 60)
            print(current_text)

            last_text = current_text

except KeyboardInterrupt:

    print("\n종료")

finally:

    camera.release()