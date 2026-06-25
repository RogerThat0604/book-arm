import time
from pymycobot.mycobot import MyCobot

PORT = "/dev/ttyUSB0"
BAUD = 1000000

mc = MyCobot(PORT, BAUD)
time.sleep(1)

print("power on")
mc.power_on()
time.sleep(1)

print("angles:", mc.get_angles())
print("coords:", mc.get_coords())

print("관절1 테스트")
mc.send_angle(1, 30, 30)
time.sleep(3)

print("angles after:", mc.get_angles())

print("home")
mc.send_angles([0, 0, 0, 0, 0, 0], 30)
time.sleep(3)