from pymycobot.mycobot import MyCobot
import time

mc = MyCobot("/dev/ttyUSB0", 1000000)
mc.power_on()
time.sleep(1)

home = [0, 0, 0, 0, 0, 0]

# 임시값: 책 위/책 잡는 위치/들기/바구니 위치
above_book = [0, -20, -20, 0, 30, 0]
pick_book  = [0, -35, -35, 0, 40, 0]
lift_book  = [0, -15, -15, 0, 30, 0]

literature_bin = [45, -10, -15, 0, 30, 0]

print("HOME")
mc.send_angles(home, 20)
time.sleep(2)

print("그리퍼 열기")
mc.set_gripper_state(0, 50)
time.sleep(1)

print("책 위로 이동")
mc.send_angles(above_book, 15)
time.sleep(3)

print("책 잡는 위치로 하강")
mc.send_angles(pick_book, 10)
time.sleep(3)

print("그리퍼 닫기")
mc.set_gripper_state(1, 50)
time.sleep(2)

print("들기")
mc.send_angles(lift_book, 10)
time.sleep(3)

print("문학 바구니로 이동")
mc.send_angles(literature_bin, 15)
time.sleep(3)

print("그리퍼 열기")
mc.set_gripper_state(0, 50)
time.sleep(2)

print("HOME 복귀")
mc.send_angles(home, 20)
time.sleep(3)

print("완료")
