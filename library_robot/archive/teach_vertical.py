#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
teach_vertical.py — top-down grasp의 수직 자세(orientation) 기록

목적: 그리퍼가 책상에 '수직'으로 내려와 책을 정면에서 잡는 자세의 (rx, ry, rz)를 알아낸다.
      이 세 값은 책 위치가 바뀌어도 그대로 재사용할 수 있는 자세 기준이다.

사용:
  1) 책상 위에 책을 한 권 놓는다.
  2) 실행하면 팔 힘이 풀린다. 손으로 그리퍼를 책 등 바로 위로 옮긴 뒤,
     그리퍼가 '바닥을 향해 똑바로(수직)' 내려가는 자세로 맞춘다.
     (그리퍼 손가락이 책 등을 수직으로 감싸는 모양)
  3) 그 자세 그대로 잡고 Enter.
  4) 출력되는 VERTICAL_RPY 와 Z_PICK 을 v9 파일에 붙여넣는다.
"""

import time
from pymycobot import MyCobot280 as MyCobot

PORT = "/dev/ttyJETCOBOT"
BAUD = 1000000

mc = MyCobot(PORT, BAUD)
mc.thread_lock = True
mc.power_on()
time.sleep(1)

print("팔 힘을 풉니다. 그리퍼가 책상에 '수직'으로 내려가는 자세로 만들어 주세요.")
print("  (그리퍼 손가락이 책 등을 정면에서 감싸는 모양)")
input("준비됐으면 Enter (한 손으로 받칠 준비!)... ")
mc.release_all_servos()
time.sleep(0.3)

input("자세 잡았으면 Enter... ")

coords = None
angles = None
for _ in range(6):
    coords = mc.get_coords()
    angles = mc.get_angles()
    if coords and len(coords) == 6 and angles and len(angles) == 6:
        break
    time.sleep(0.3)

if coords:
    coords = [round(c, 1) for c in coords]
if angles:
    angles = [round(a, 1) for a in angles]

# 잡힌 자세 그대로 재고정
if angles:
    mc.send_angles(angles, 30)
    time.sleep(1.5)

print("\n──────── v9 파일에 붙여넣으세요 ────────")
print(f"# 좌표 전체: {coords}")
print(f"VERTICAL_RPY = ({coords[3]}, {coords[4]}, {coords[5]})   # (rx, ry, rz)")
print(f"Z_PICK = {coords[2]}   # mm. 책 잡는 책상 높이")
print(f"# 참고 관절각: {angles}")
print("────────────────────────────────────────")
