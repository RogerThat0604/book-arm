from pymycobot.mycobot import MyCobot

# 로봇 연결 설정
mc = MyCobot('/dev/ttyJETCOBOT', 1000000)
mc.thread_lock = True

# 로봇을 초기 위치로 리셋
initial_angles = [0, 0, 0, 0, 0, 0]
speed = 30

mc.send_angles(initial_angles, speed)
print("리셋 완료")
