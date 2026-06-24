"""
책꽂이 스캔 - 직선 이동 예시 코드 (1단계: 아르코마커 제외)
=================================================================
목표:
  - 로봇 손을 책꽂이 앞으로 뻗은 상태에서
  - 자세(rx, ry, rz)를 고정한 채
  - y축으로만 좌->우 직선 이동 (책꽂이로부터 x거리 유지)

좌표계 가정 (MyCobot280 base 좌표계):
  x : 로봇 ↔ 책꽂이 거리 (고정)
  y : 좌우 이동축 (왼쪽 -y -> 오른쪽 +y, 혹은 반대 - 실측 후 부호 조정)
  z : 스캔할 책장 높이 (고정)
  rx, ry, rz : 손목 자세 (고정 - 손이 항상 책꽂이를 정면으로 바라봄)

이 단계에서는 아르코마커 인식 없이 "정해진 시작/끝 좌표 사이를
일정 속도로 직선 이동하는 동작"만 검증한다.
다음 단계에서 아르코마커 좌표를 y_start/y_end 자리에 동적으로 꽂으면 된다.

실행 전 설치:
  pip install pymycobot
"""

import time
from pymycobot import MyCobot280

# ──────────────────────────────────────────
# 설정값 (환경에 맞게 반드시 실측 후 수정)
# ──────────────────────────────────────────

ROBOT_PORT = '/dev/ttyJETCOBOT'  # JetCobot 전용 udev 심볼릭 링크 (공식 문서 기준)
ROBOT_BAUD = 1000000             # JetCobot은 1,000,000 baud

# 손을 뻗어 책꽂이를 바라보는 기준 자세
# 실제 로봇에서 jog로 원하는 자세를 잡은 뒤 get_coords()로 읽어서 채워 넣을 것
BASE_X  = 200.0    # 책꽂이로부터 유지할 거리 (mm) - 고정
BASE_Z  = 150.0    # 스캔할 책장 높이 (mm) - 고정
FIXED_RX = 0.0     # 손목 roll - 고정
FIXED_RY = 0.0     # 손목 pitch - 고정
FIXED_RZ = 0.0     # 손목 yaw - 고정

# 스캔 구간 (y축 좌->우)
Y_LEFT  = -150.0   # 책꽂이 왼쪽 끝에 대응하는 y 좌표
Y_RIGHT =  150.0   # 책꽂이 오른쪽 끝에 대응하는 y 좌표

# 이동 속도 / 모드
MOVE_SPEED = 15      # 1~100, 느릴수록 카메라 흔들림 적음
MOVE_MODE  = 1        # 0: angular(moveJ), 1: linear(moveL) - 직선 이동이므로 1
STEP_MM    = 10.0     # 한 스텝당 y 이동 거리 (mm) - 작을수록 부드러움
STEP_DELAY = 0.3      # 스텝 사이 대기 시간 (초) - 카메라가 안정될 시간


def goto_start_pose(mc: MyCobot280):
    """책꽂이 앞, 왼쪽 끝 시작 위치로 이동"""
    start_coords = [BASE_X, Y_LEFT, BASE_Z, FIXED_RX, FIXED_RY, FIXED_RZ]
    print(f"[Move] 시작 위치로 이동: {start_coords}")
    mc.sync_send_coords(start_coords, MOVE_SPEED, MOVE_MODE, timeout=20)


def scan_left_to_right(mc: MyCobot280):
    """
    y축만 변경하며 왼쪽 -> 오른쪽으로 스텝 이동.
    x, z, rx, ry, rz는 고정되므로 손의 자세(=카메라 방향)는 항상 동일하게 유지된다.
    """
    y = Y_LEFT
    direction = 1 if Y_RIGHT > Y_LEFT else -1

    while (direction > 0 and y < Y_RIGHT) or (direction < 0 and y > Y_RIGHT):
        coords = [BASE_X, y, BASE_Z, FIXED_RX, FIXED_RY, FIXED_RZ]
        print(f"[Scan] y={y:.1f} -> 이동")
        mc.send_coords(coords, MOVE_SPEED, MOVE_MODE)

        # 다음 스텝까지 대기 (이동 완료 + 카메라 안정화 시간)
        time.sleep(STEP_DELAY)

        # 여기서 카메라 프레임을 캡처해 아르코마커 인식을 수행할 수 있다.
        # 다음 단계에서 이 지점에 OCR/ArUco 인식 콜백을 추가한다.
        # frame = cap.read() ...
        # detect_aruco(frame) ...

        y += STEP_MM * direction

    # 마지막 정확히 끝점으로 보정
    final_coords = [BASE_X, Y_RIGHT, BASE_Z, FIXED_RX, FIXED_RY, FIXED_RZ]
    print(f"[Scan] 종료 위치 보정: {final_coords}")
    mc.sync_send_coords(final_coords, MOVE_SPEED, MOVE_MODE, timeout=20)


def main():
    print(f"[Init] 로봇 연결: {ROBOT_PORT} @ {ROBOT_BAUD}")
    mc = MyCobot280(ROBOT_PORT, ROBOT_BAUD)
    mc.thread_lock = True   # JetCobot 공식 예제 기준 필수 설정
    time.sleep(0.5)
    # 공식 예제에는 power_on() 호출이 없음 - 안 움직이면 이 두 줄 빼고 테스트
    mc.power_on()
    time.sleep(0.5)

    try:
        goto_start_pose(mc)
        time.sleep(1.0)

        print("[Run] 좌->우 스캔 시작")
        scan_left_to_right(mc)
        print("[Run] 스캔 완료")

    except KeyboardInterrupt:
        print("[Stop] 사용자 중단")
        mc.stop()

    finally:
        mc.power_off()
        print("[Done] 종료")


if __name__ == '__main__':
    main()
