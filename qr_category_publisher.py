import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pyzbar.pyzbar import decode


class QRCategoryPublisher(Node):
    def __init__(self):
        super().__init__("qr_category_publisher")
        self.publisher = self.create_publisher(String, "/book_category", 10)
        self.last_data = None

    def publish_category(self, category):
        msg = String()
        msg.data = category
        self.publisher.publish(msg)
        self.get_logger().info(f"📤 QR 카테고리 발행: {category}")


def main():
    rclpy.init()
    node = QRCategoryPublisher()

    cap = cv2.VideoCapture(6)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print("QR 분류 시작")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임 읽기 실패")
            break

        qr_codes = decode(frame)

        for qr in qr_codes:
            data = qr.data.decode("utf-8").strip()

            if data in ["문학", "과학", "역사"] and data != node.last_data:
                node.publish_category(data)
                node.last_data = data

        rclpy.spin_once(node, timeout_sec=0)

    cap.release()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()