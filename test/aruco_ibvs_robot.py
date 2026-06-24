import time
from pymycobot import MyCobot280

PORT = "/dev/ttyJETCOBOT"
BAUD = 1000000

mc = MyCobot280(PORT, BAUD)
time.sleep(1)

base = mc.get_angles()
print("기준 각도:", base)

for joint_index in range(6):
    print(f"\n[J{joint_index + 1}] 테스트")

    target = base.copy()
    target[joint_index] += 2.0

    input(f"Enter: J{joint_index + 1} +2도")
    mc.send_angles(target, 8)
    time.sleep(2.0)
    print("이동 후:", mc.get_angles())

    input("Enter: 기준 각도로 복귀")
    mc.send_angles(base, 8)
    time.sleep(2.5)
    print("복귀 후:", mc.get_angles())

print("완료")
mc.stop()