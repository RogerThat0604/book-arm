import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pymycobot.mycobot import MyCobot


class ArmSorterNode(Node):
    def __init__(self):
        super().__init__("arm_sorter_node")

        self.mc = MyCobot("/dev/ttyUSB0", 1000000)
        self.mc.power_on()
        time.sleep(1)

        self.subscription = self.create_subscription(
            String,
            "/book_category",
            self.category_callback,
            10
        )

        self.get_logger().info("📚 Arm Sorter Node 시작")
        self.get_logger().info("/book_category 구독 중...")

    def category_callback(self, msg):
        category = msg.data.strip()
        self.get_logger().info(f"📥 카테고리 수신: {category}")
        self.move_to_category(category)

    def move_to_category(self, category):
        self.get_logger().info(f"🤖 실제 팔 이동 시작: {category}")

        if category == "문학":
            target = [30, 0, 0, 0, 0, 0]
        elif category == "과학":
            target = [0, 30, 0, 0, 0, 0]
        elif category == "역사":
            target = [-30, 0, 0, 0, 0, 0]
        else:
            self.get_logger().warn(f"알 수 없는 카테고리: {category}")
            return

        self.mc.send_angles(target, 20)
        time.sleep(3)

        self.mc.send_angles([0, 0, 0, 0, 0, 0], 20)
        time.sleep(3)

        self.get_logger().info("✅ 실제 팔 이동 완료")


def main(args=None):
    rclpy.init(args=args)
    node = ArmSorterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
