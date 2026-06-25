#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
teach_poses.py — 드래그 티칭(손으로 자세 잡기). 처짐 방지 개선판.

각 자세를 기록한 직후 그 자세로 팔을 '재고정'하므로,
다음 자세를 잡을 때까지 팔이 축 처지지 않는다.

실행:  python teach_poses.py

⚠ 힘이 풀리는 순간(메시지가 그렇게 안내함)에는 반드시 한 손으로 팔을 받치세요.
   책은 데모 때와 같은 위치에 놓고 가르치세요.
"""

import time
from pymycobot import MyCobot280 as MyCobot

PORT = "/dev/ttyJETCOBOT"
BAUD = 1000000

mc = MyCobot(PORT, BAUD)
mc.thread_lock = True
mc.power_on()
time.sleep(1)

# 현재 자세에서 시작(팔을 굳혀 둠)
cur = mc.get_angles()
if cur and len(cur) == 6:
    mc.send_angles(cur, 30)
    time.sleep(1.5)

steps = [
    ("ABOVE_J", "그리퍼를 열고, 책 '바로 위'에 떠 있는 자세"),
    ("PICK_J",  "책을 실제로 '잡는 높이'까지 내린 자세"),
    ("LIFT_J",  "책을 집어 '살짝 들어올린' 자세"),
]

saved = {}
mc.set_gripper_state(0, 50)   # 그리퍼 열어두고 시작
time.sleep(1)

for key, desc in steps:
    print(f"\n========== {key} ==========")
    print(f"목표 자세: {desc}")
    input("준비됐으면 Enter → 팔 힘이 풀립니다. (한 손으로 받칠 준비!) ")
    mc.release_all_servos()
    time.sleep(0.3)
    print("이제 손으로 팔을 움직여 자세를 잡으세요.")
    input("자세를 잡은 채로(흔들리지 않게) Enter를 누르세요... ")

    ang = None
    for _ in range(6):
        ang = mc.get_angles()
        if ang and len(ang) == 6:
            break
        time.sleep(0.3)
    saved[key] = [round(a, 1) for a in ang] if ang else None
    print(f"  기록: {saved[key]}")

    # 즉시 그 자세로 재고정(처짐 방지)
    if ang and len(ang) == 6:
        mc.send_angles(ang, 30)
        time.sleep(1.5)
        print("  팔을 그 자세로 다시 굳혔어요. (이제 안 처짐)")
    else:
        print("  ⚠ 각도 읽기 실패 — 이 자세는 다시 시도하세요.")

print("\n──────── 아래 세 줄을 v6 파일의 TEACH 블록에 붙여넣으세요 ────────")
print(f"ABOVE_J = {saved.get('ABOVE_J')}")
print(f"PICK_J  = {saved.get('PICK_J')}")
print(f"LIFT_J  = {saved.get('LIFT_J')}")
print("──────────────────────────────────────────────────────────────")
print("\n끝! 팔은 마지막 자세로 굳어 있어요. 그대로 두면 됩니다.")
