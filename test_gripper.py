from pymycobot.mycobot import MyCobot
import time

mc = MyCobot("/dev/ttyUSB0", 1000000)
mc.power_on()
time.sleep(1)

print("그리퍼 열기")
mc.set_gripper_state(0, 50)
time.sleep(2)

print("그리퍼 닫기")
mc.set_gripper_state(1, 50)
time.sleep(2)

print("완료")
