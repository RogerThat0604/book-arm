import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pymycobot.mycobot import MyCobot


class ArmSorterNode(Node):
    def __init__(self):
        super().__init__('arm_sorter_node_v2')

        self.mc = MyCobot('/dev/ttyUSB0', 1000000)
        self.mc.power_on()
        time.sleep(1)

        self.subscription = self.create_subscription(
            String,
            '/book_category',
            self.category_callback,
            10
        )

        self.get_logger().info('🤖 Arm Sorter V2 시작')
        self.get_logger().info('📚 /book_category 구독 중...')

    def category_callback(self, msg):
        category = msg.data.strip()
        self.get_logger().info(f'📥 카테고리 수신: {category}')
        self.move_to_category(category)

    def move_to_category(self, category):
        home = [0, 0, 0, 0, 0, 0]

        if category == '문학':
            target = [45, 0, 0, 0, 0, 0]

        elif category == '과학':
            target = [0, 25, 0, 0, 0, 0]

        elif category == '역사':
            target = [-45, 0, 0, 0, 0, 0]
        else:
            self.get_logger().warn(f'알 수 없는 카테고리: {category}')
            return

        self.get_logger().info(f'🚀 {category} 바구니 이동')
        self.mc.send_angles(target, 20)
        time.sleep(4)

        self.get_logger().info('🏠 HOME 복귀')
        self.mc.send_angles(home, 20)
        time.sleep(4)

        self.get_logger().info('✅ 작업 완료')


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


if __name__ == '__main__':
    main()
