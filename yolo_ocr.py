import cv2
import easyocr
import mediapipe as mp
import requests
from ultralytics import YOLO
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

last_search_time = 0

searched_books = set()

NAVER_CLIENT_ID = "6UVCQPXczTv61_dYotMB"
NAVER_CLIENT_SECRET = "8uUw2LhVmR"

def clean_html(text):
    return text.replace("<b>", "").replace("</b>", "")

def get_book_info(title):
    clean = title.strip().replace(" ", "")

    url = "https://openapi.naver.com/v1/search/book.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {
        "query": clean,
        "display": 1,
        "sort": "sim",
    }

    try:
        r = requests.get(url, headers=headers, params=params, timeout=5)
        print("API 상태:", r.status_code)

        data = r.json()

        if r.status_code != 200 or not data.get("items"):
            print("검색 결과 없음")
            return None

        book = data["items"][0]

        return {
            "title": clean_html(book.get("title", "제목 없음")),
            "authors": clean_html(book.get("author", "저자 없음")),
            "publisher": clean_html(book.get("publisher", "출판사 없음")),
            "description": clean_html(book.get("description", "소개 없음"))[:300],
        }

    except Exception as e:
        print("API 오류:", e)
        return None

class BookCategoryPublisher(Node):
    def __init__(self):
        super().__init__('book_category_publisher')
        self.publisher = self.create_publisher(String, '/book_category', 10)

    def publish_category(self, category):
        msg = String()
        msg.data = category
        self.publisher.publish(msg)
        self.get_logger().info(f'📤 /book_category 발행: {category}')

class BookPublisher(Node):

    def __init__(self):
        super().__init__("book_publisher")

        self.book_pub = self.create_publisher(
            String,
            "/recognized_book",
            10
        )

        self.category_pub = self.create_publisher(
            String,
            "/book_category",
            10
        )

    def publish_book(self, title):
        msg = String()
        msg.data = title
        self.book_pub.publish(msg)
        print(f"📢 책 발행: {title}")

    def publish_category(self, category):
        msg = String()
        msg.data = category
        self.category_pub.publish(msg)
        print(f"📢 카테고리 발행: {category}")
# =====================
# 모델 준비
# =====================
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(["ko", "en"])

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

BOOK_CATEGORY = {
    "어린왕자": "문학",
    "데미안": "문학",
    "1984": "문학",
    "죄와벌": "문학",

    "코스모스": "과학",
    "사피엔스": "과학",
    "이기적유전자": "과학",

    "총균쇠": "역사",
}

rclpy.init()

book_publisher = BookPublisher()

# =====================
# 카메라 준비
# =====================
cap = cv2.VideoCapture(6)

if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

print("YOLO + Greeting + OCR + Book API 시작")
print("q 키 종료")

frame_count = 0
last_hand_x = None
wave_count = 0
greeting_triggered = False

last_searched_text = ""

# =====================
# 메인 루프
# =====================
while True:
    ret, frame = cap.read()

    if not ret:
        print("카메라 프레임 읽기 실패")
        break

    print("frame ok")

    frame_count += 1
    h, w, _ = frame.shape

    # =====================
    # 1. YOLO 사람 감지
    # =====================
    results = model(frame, verbose=False)
    person_detected = False

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])

            if class_name != "person" or conf < 0.5:
                continue

            person_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                frame,
                (int(w * 0.05), int(h * 0.35)),
                (int(w * 0.75), int(h * 0.95)),
                (255, 0, 255),
                2
            )
            cv2.putText(
                frame,
                f"person {conf:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    # =====================
    # 2. 손 인사 감지
    # =====================
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)

    if person_detected and hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[0]
            hand_x = int(wrist.x * w)

            if last_hand_x is not None:
                diff = hand_x - last_hand_x

                if abs(diff) > 25:
                    wave_count += 1

            last_hand_x = hand_x

            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

        if wave_count >= 6 and not greeting_triggered:
            print("👋 인사 감지됨: 안녕하세요!")
            greeting_triggered = True

        if greeting_triggered:
            cv2.putText(
                frame,
                "Greeting Detected: Hello!",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )
    else:
        last_hand_x = None
        wave_count = 0
        greeting_triggered = False

    # =====================
    # 3. OCR + 책 정보 검색
    # =====================
    if frame_count % 30 == 0:
        # OCR 영역 제한: 화면 왼쪽/아래쪽 또는 책 놓는 위치만 읽기
        roi = frame[int(h * 0.35):int(h * 0.95), int(w * 0.05):int(w * 0.75)]

        ocr_results = reader.readtext(roi)

        for _, text, ocr_conf in ocr_results:
            if ocr_conf < 0.5:
                continue

            clean_text = text.strip().replace(" ", "")

            OCR_FIX = {
                "스모스": "코스모스",
                "스모": "코스모스",
                "데미": "데미안",
                "미안": "데미안",
                "린왕자": "어린왕자",
                "총균": "총균쇠",
                "균쇠": "총균쇠",
            }

            for wrong, fixed in OCR_FIX.items():
                if wrong in clean_text:
                    print(f"🔧 OCR 보정: {clean_text} -> {fixed}")
                    clean_text = fixed
                    break

            # 1. 의미 없는 단어 먼저 제거
            ignore_words = ["person", "문학", "과학", "역사", "Error", "Hello"]
            if clean_text in ignore_words:
                continue

            # 2. 같은 책 중복 검색 방지
            if clean_text in searched_books:
                continue

            searched_books.add(clean_text)

            print(f"[OCR] {clean_text} / {ocr_conf:.2f}")

            # 3. 네이버 API 검색
            info = get_book_info(clean_text)

            if info:
                print("\n📚 책 정보")
                print("제목:", info["title"])
                print("저자:", info["authors"])
                print("출판사:", info["publisher"])
                print("소개:", info["description"])
                print("-" * 40)

                book_publisher.publish_book(
                    info["title"]
                )

                category = BOOK_CATEGORY.get(clean_text, "미분류")

                if category != "미분류":
                    book_publisher.publish_category(category)

                    rclpy.spin_once(
                        book_publisher,
                        timeout_sec=0
                    )
                else:
                    print("⚠️ 미분류라서 팔 이동 안 함")
                
                rclpy.spin_once(
                    book_publisher,
                    timeout_sec=0
                )    

    # =====================
    # 화면 출력
    # =====================
    
    #cv2.imshow("YOLO + Greeting + OCR + Book API", frame)

    #if cv2.waitKey(1) & 0xFF == ord("q"):
    #    break
    time.sleep(0.03)

# =====================
# 종료
# =====================
cap.release()
hands.close()
cv2.destroyAllWindows()

book_publisher.destroy_node()
rclpy.shutdown()