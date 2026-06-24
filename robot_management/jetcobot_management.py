import argparse
import itertools
import sys
import time
import subprocess
from pymycobot.mycobot280 import MyCobot280

PORT = "/dev/ttyJETCOBOT"
BAUD = 1000000

HOME_ANGLES = [0, 0, 0, 0, 0, 0]
DEFAULT_SPEED = 50

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def countdown_spinner(seconds, message):
    spinner = itertools.cycle([
        "⠋", "⠙", "⠹", "⠸",
        "⠼", "⠴", "⠦", "⠧",
        "⠇", "⠏"
    ])

    end_time = time.time() + seconds

    while True:
        remain = end_time - time.time()

        if remain <= 0:
            break

        sys.stdout.write(
            f"\r{next(spinner)} {message:<20} {remain:0.1f}s"
        )
        sys.stdout.flush()
        time.sleep(0.08)

    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()


def success(message):
    print(f"{GREEN}✓{RESET} {message}")


def warn(message):
    print(f"{YELLOW}⚠{RESET} {message}")


def error(message):
    print(f"{RED}■{RESET} {message}")


def connect_robot():
    mc = MyCobot280(PORT, BAUD)
    mc.thread_lock = True
    return mc


def print_robot_status(mc):
    angles = mc.get_angles()
    coords = mc.get_coords()
    encoders = mc.get_encoders()

    print("현재 각도:", angles)
    print("현재 좌표:", coords)
    print("인코더:", encoders)


def release_servos(mc):
    countdown_spinner(3, "서보 토크 해제 준비중")
    mc.release_all_servos()
    time.sleep(1)
    success("서보 토크 해제 완료")


def focus_servos(mc):
    countdown_spinner(3, "서보 토크 활성화 준비중")
    mc.focus_all_servos()
    time.sleep(1)
    success("서보 토크 활성화 완료")


def move_home(mc):
    warn("로봇팔이 움직입니다. 주변을 확인해주세요")
    countdown_spinner(3, "초기 위치 이동 준비중")

    mc.focus_all_servos()
    time.sleep(1)

    mc.send_angles(HOME_ANGLES, DEFAULT_SPEED)
    time.sleep(3)

    mc.set_gripper_value(100, DEFAULT_SPEED)
    time.sleep(1)

    success("초기 위치 이동 완료")


def print_header(title="JETCOBOT CONTROL SEQUENCE"):
    print()
    print("=========================================")
    print(f"      {title}")
    print("=========================================")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "shutdown",
            "servooff",
            "servoon",
            "home",
            "status"
        ],
        required=True
    )

    args = parser.parse_args()

    try:
        print_header()

        if args.mode == "shutdown":
            subprocess.run(["sudo", "-v"], check=True)

        if args.mode in ["shutdown", "servooff"]:
            warn("로봇팔을 잡고 대기해주세요")
            print("  Ctrl + C 로 종료를 취소할 수 있습니다")
            print()

        mc = connect_robot()

        if args.mode == "shutdown":
            release_servos(mc)

            print()

            countdown_spinner(3, "시스템 종료 준비중")
            success("시스템 종료 요청 완료")

            print()
            print("안전 종료가 완료되었습니다.")
            print("Goodbye.")
            print()

            subprocess.run(["sudo", "shutdown", "-h", "now"])

        elif args.mode == "servooff":
            release_servos(mc)
            print()
            print("모터 비활성화만 완료되었습니다.")
            print()

        elif args.mode == "servoon":
            focus_servos(mc)
            print()
            print_robot_status(mc)
            print()

        elif args.mode == "home":
            move_home(mc)
            print()
            print_robot_status(mc)
            print()

        elif args.mode == "status":
            print_robot_status(mc)
            print()

    except KeyboardInterrupt:
        print()
        error("작업이 취소되었습니다.")
        print()

    except Exception as e:
        print()
        error(f"오류 발생: {e}")
        print("안전을 위해 작업을 중단했습니다.")
        print()


if __name__ == "__main__":
    main()
