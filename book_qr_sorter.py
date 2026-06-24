import cv2
import numpy as np
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pyzbar.pyzbar import decode


class BookQRSorter(Node):
    def __init__(self):
        super().__init__("book_qr_sorter")
        self.pub = self.create_publisher(String, "/book_category", 10)
        self.last_publish_time = 0

    def publish_category(self, category):
        now = time.time()

        if now - self.last_publish_time < 8:
            return

        msg = String()
        msg.data = category
        self.pub.publish(msg)
        self.last_publish_time = now
        self.get_logger().info(f"📤 책 감지 + QR 분류 발행: {category}")


def detect_book(frame):
    h, w = frame.shape[:2]

    roi_x1 = int(w * 0.15)
    roi_y1 = int(h * 0.25)
    roi_x2 = int(w * 0.85)
    roi_y2 = int(h * 0.95)

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 120])
    upper = np.array([180, 90, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 250:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        if area > best_area:
            best = (x, y, bw, bh)
            best_area = area

    if best is None:
        return None

    x, y, bw, bh = best

    cx = roi_x1 + x + bw // 2
    cy = roi_y1 + y + bh // 2

    dx = cx - (w // 2)
    dy = cy - (h // 2)

    return cx, cy, dx, dy, best_area


def detect_qr(frame):
    qrs = decode(frame)

    for qr in qrs:
        data = qr.data.decode("utf-8").strip()

        if data in ["문학", "과학", "역사"]:
            return data

    return None


def main():
    rclpy.init()
    node = BookQRSorter()

    cap = cv2.VideoCapture(6)

    if not cap.isOpened():
        print("카메라 열기 실패")
        return

    print("📚 책 감지 + QR 분류 시작")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임 읽기 실패")
            break

        book = detect_book(frame)
        category = detect_qr(frame)

        if book:
            cx, cy, dx, dy, area = book
            print(f"책 감지: dx={dx}, dy={dy}, area={area}")

            if category:
                print(f"QR 카테고리: {category}")

                if abs(dx) < 60:
                    print("✅ 중앙 정렬됨 → 분류 발행")
                    node.publish_category(category)
                else:
                    print("⚠️ 책을 중앙에 맞춰주세요")
            else:
                print("QR 없음")

        else:
            print("책 감지 실패")

        rclpy.spin_once(node, timeout_sec=0)
        time.sleep(0.5)

    cap.release()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()