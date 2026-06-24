import time
import threading
from pymycobot import MyCobot280

PORT = "/dev/ttyJETCOBOT"
BAUD = 1000000

SPEED = 15
MODE_L = 1  # 1 = Linear

# 부드럽게 보이도록 작은 이동량
DX_STEP = 1.0     # 앞으로 mm
DZ_STEP = -0.8    # 아래로 mm

# 명령 주기
INTERVAL = 0.06

MAX_FORWARD = 120
MAX_DOWN = 120

stop_flag = False


def keyboard_listener():
    global stop_flag
    input("\n[ENTER] 누르면 즉시 정지\n")
    stop_flag = True


def valid_coords(coords):
    return isinstance(coords, list) and len(coords) == 6


mc = MyCobot280(PORT, BAUD)
mc.thread_lock = True

time.sleep(1)

mc.power_on()
time.sleep(1)

# 최신 명령 우선 실행 모드
mc.set_fresh_mode(1)
time.sleep(0.2)

home = mc.get_coords()
print("home:", home)

if not valid_coords(home):
    print("좌표 읽기 실패")
    exit()

# 카메라 방향 고정
fixed_rx = home[3]
fixed_ry = home[4]
fixed_rz = home[5]

start_x = home[0]
start_z = home[2]

current = home.copy()

threading.Thread(target=keyboard_listener, daemon=True).start()

try:
    print("\n자연스러운 전진 + 하강 테스트 시작")

    while not stop_flag:
        forward_dist = current[0] - start_x
        down_dist = start_z - current[2]

        if forward_dist >= MAX_FORWARD:
            print("\n[LIMIT] 최대 전진 거리 도달")
            break

        if down_dist >= MAX_DOWN:
            print("\n[LIMIT] 최대 하강 거리 도달")
            break

        current[0] += DX_STEP
        current[2] += DZ_STEP

        # 카메라 자세 고정
        current[3] = fixed_rx
        current[4] = fixed_ry
        current[5] = fixed_rz

        mc.send_coords(current, SPEED, MODE_L)

        print(
            f"x={current[0]:.1f}, z={current[2]:.1f}, "
            f"forward={forward_dist:.1f}, down={down_dist:.1f}"
        )

        time.sleep(INTERVAL)

finally:
    stop_flag = True
    mc.stop()

    time.sleep(0.5)

    print("\n최종 좌표:")
    print(mc.get_coords())

    print("\n종료")