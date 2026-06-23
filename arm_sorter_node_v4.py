import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pymycobot.mycobot import MyCobot


class ArmSorterV4(Node):
    def __init__(self):
        super().__init__("arm_sorter_node_v4")

        self.mc = MyCobot("/dev/ttyUSB0", 1000000)
        self.mc.power_on()
        time.sleep(1)

        self.busy = False

        self.home = [0, 0, 0, 0, 0, 0]

        self.above_book = [0, -20, -20, 0, 30, 0]
        self.pick_book = [0, -35, -35, 0, 40, 0]
        self.lift_book = [0, -15, -15, 0, 30, 0]

        self.bin_positions = {
            "문학": [45, -10, -15, 0, 30, 0],
            "과학": [0, -10, -15, 0, 30, 0],
            "역사": [-45, -10, -15, 0, 30, 0],
        }

        self.sub = self.create_subscription(
            String,
            "/book_category",
            self.category_callback,
            10
        )

        self.get_logger().info("🤖 Arm Sorter V4 시작")
        self.get_logger().info("📚 /book_category 구독 중...")

    def move(self, angles, speed=15, wait=3):
        self.mc.send_angles(angles, speed)
        time.sleep(wait)

    def open_gripper(self):
        self.get_logger().info("✋ 그리퍼 열기")
        self.mc.set_gripper_state(0, 50)
        time.sleep(1)

    def close_gripper(self):
        self.get_logger().info("🤏 그리퍼 닫기")
        self.mc.set_gripper_state(1, 50)
        time.sleep(2)

    def pick_book_sequence(self):
        self.get_logger().info("📍 책 위로 이동")
        self.move(self.above_book, 15, 3)

        self.get_logger().info("⬇️ 책 잡는 위치로 이동")
        self.move(self.pick_book, 10, 3)

        self.close_gripper()

        self.get_logger().info("⬆️ 책 들기")
        self.move(self.lift_book, 10, 3)

    def place_book_sequence(self, category):
        self.get_logger().info(f"🚚 {category} 바구니로 이동")
        self.move(self.bin_positions[category], 15, 3)

        self.get_logger().info("📖 책 놓기")
        self.open_gripper()

    def go_home(self):
        self.get_logger().info("🏠 HOME 복귀")
        self.move(self.home, 20, 3)

    def category_callback(self, msg):
        category = msg.data.strip()

        if self.busy:
            self.get_logger().warn("⚠️ 작업 중이라 명령 무시")
            return

        if category not in self.bin_positions:
            self.get_logger().warn(f"❌ 알 수 없는 카테고리: {category}")
            return

        self.busy = True
        self.get_logger().info(f"📥 카테고리 수신: {category}")

        try:
            self.go_home()
            self.open_gripper()
            self.pick_book_sequence()
            self.place_book_sequence(category)
            self.go_home()
            self.get_logger().info("✅ V4 Pick & Place 분류 완료")

        except Exception as e:
            self.get_logger().error(f"❌ 작업 실패: {e}")

        finally:
            self.busy = False


def main(args=None):
    rclpy.init(args=args)
    node = ArmSorterV4()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()