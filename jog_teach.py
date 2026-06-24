#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jog_teach.py — 팔에 힘을 준 채로 관절을 조금씩 움직여 자세를 가르친다.
              (드래그 티칭처럼 팔이 처지지 않아 훨씬 쉽고 정밀)

실행:  python jog_teach.py

명령어 (입력 후 Enter):
  1 5       → 1번 관절 +5도   (1~6번, 음수도 가능: 3 -10)
  g o       → 그리퍼 열기
  g c       → 그리퍼 닫기
  s         → 현재 각도 보기
  a / p / l → 지금 자세를 ABOVE / PICK / LIFT 로 저장
  q         → 끝내고 결과 출력

권장 순서:
  1) 관절을 움직여 '책 바로 위'를 만든 뒤  a
  2) 더 내려 '잡는 높이'를 만든 뒤        p
  3) 다시 들어올린 자세에서               l
  4) q 로 끝내고 출력값을 v6에 붙여넣기
"""

import time
from pymycobot import MyCobot280 as MyCobot

PORT = "/dev/ttyJETCOBOT"
BAUD = 1000000
SPEED = 20
JOINT_MIN, JOINT_MAX = -160, 160     # 안전용 소프트 제한

mc = MyCobot(PORT, BAUD)
mc.thread_lock = True
mc.power_on()
time.sleep(1)

angles = mc.get_angles()
if not angles or len(angles) != 6:
    angles = [0, 0, 0, 0, 0, 0]
mc.send_angles(angles, SPEED)
time.sleep(2)
print("준비 완료. 현재 각도:", [round(a, 1) for a in angles])
print("명령: '관절번호 각도'(예: 1 5 / 3 -10), g o, g c, s, a, p, l, q")

saved = {"ABOVE_J": None, "PICK_J": None, "LIFT_J": None}

while True:
    cmd = input("명령> ").strip().lower()
    if cmd in ("q", "quit"):
        break
    if cmd == "s":
        print("  현재:", [round(a, 1) for a in angles]); continue
    if cmd in ("a", "p", "l"):
        key = {"a": "ABOVE_J", "p": "PICK_J", "l": "LIFT_J"}[cmd]
        saved[key] = [round(a, 1) for a in angles]
        print(f"  {key} 저장:", saved[key]); continue
    if cmd.startswith("g"):
        parts = cmd.split()
        if len(parts) == 2 and parts[1] in ("o", "open"):
            mc.set_gripper_state(0, 50); print("  그리퍼 열기")
        elif len(parts) == 2 and parts[1] in ("c", "close"):
            mc.set_gripper_state(1, 50); print("  그리퍼 닫기")
        else:
            print("  사용법: g o  또는  g c")
        continue

    # 관절 이동: "번호 각도"
    parts = cmd.split()
    if len(parts) == 2 and parts[0] in "123456":
        try:
            idx = int(parts[0]) - 1
            delta = float(parts[1])
        except ValueError:
            print("  형식 오류. 예: 1 5"); continue
        new = angles[idx] + delta
        if not (JOINT_MIN <= new <= JOINT_MAX):
            print(f"  제한 초과({new:.1f}). 무시."); continue
        angles[idx] = new
        try:
            mc.send_angles(angles, SPEED)
            time.sleep(1.0)
            print(f"  J{idx+1} -> {angles[idx]:.1f}")
        except Exception as e:
            print("  전송 실패:", e)
        continue

    print("  알 수 없는 명령. (1 5 / g o / g c / s / a / p / l / q)")

print("\n──────── v6 파일의 TEACH 블록에 붙여넣으세요 ────────")
print(f"ABOVE_J = {saved['ABOVE_J']}")
print(f"PICK_J  = {saved['PICK_J']}")
print(f"LIFT_J  = {saved['LIFT_J']}")
print("─────────────────────────────────────────────────────")
