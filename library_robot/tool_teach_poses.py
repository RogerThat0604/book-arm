#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
teach_shelf_poses.py — 책장 시나리오용 다중 자세 티칭

처짐 방지: 각 자세를 기록한 직후 그 자세로 재고정.
camera_web.py 와 함께 사용하면 좋음 (다른 터미널에서 띄워두고 화면 보며 자세 조정).

티칭할 자세:
  HOME              - 안전한 대기 자세
  OBSERVE_A         - 책장 A(입력)를 보는 자세, base 0° 근처
  OBSERVE_B         - 책장 B(분류)를 보는 자세, base 180° 근처
  BASKET_과학       - 과학 칸에 꽂는 자세
  BASKET_문학       - 문학 칸
  BASKET_역사       - 역사 칸

결과: shelf_poses.py 파일에 저장 → v11에서 import해서 사용

실행:
  python teach_shelf_poses.py
  (camera_web 같이 띄워두면 시야 보며 조정 가능)
"""

import time
import os
from pymycobot import MyCobot280 as MyCobot

PORT = "/dev/ttyJETCOBOT"; BAUD = 1000000
OUTPUT_FILE = "shelf_poses.py"


POSES_TO_TEACH = [
    ("HOME",         "안전한 대기 자세 (모든 동작 시작·끝 자세)"),
    ("OBSERVE_A",    "책장 A(입력)를 카메라로 보는 자세"),
    ("OBSERVE_B",    "책장 B(분류)를 카메라로 보는 자세 (보통 OBSERVE_A에서 base 180°)"),
    ("BASKET_과학",  "과학 카테고리 칸 앞 자세 (책을 거기로 가져가서 꽂는 위치)"),
    ("BASKET_문학",  "문학 카테고리 칸 앞 자세"),
    ("BASKET_역사",  "역사 카테고리 칸 앞 자세"),
]


def teach_one(mc, name, desc):
    print(f"\n{'='*50}")
    print(f"[{name}]")
    print(f"  {desc}")
    print(f"{'='*50}")
    input(f"  팔을 손으로 받칠 준비됐으면 Enter → 힘 풀림 ")
    mc.release_all_servos()
    time.sleep(0.3)
    print(f"  손으로 자세를 잡으세요. (camera_web 화면 확인하면서)")
    input(f"  자세 잡았으면 Enter (흔들리지 않게)... ")

    angles = None
    for _ in range(6):
        angles = mc.get_angles()
        if angles and len(angles) == 6:
            break
        time.sleep(0.3)

    if not angles or len(angles) != 6:
        print(f"  ⚠ 각도 읽기 실패 → 이 자세는 건너뜀")
        return None

    angles = [round(a, 1) for a in angles]
    coords = mc.get_coords()
    coords = [round(c, 1) for c in coords] if coords else None

    print(f"  ✅ 기록: {angles}")
    if coords:
        print(f"     좌표: {coords}")

    # 처짐 방지: 그 자세로 재고정
    mc.send_angles(angles, 30)
    time.sleep(1.5)
    print(f"  (자세 재고정 완료)")
    return angles


def save_poses(poses):
    """결과를 shelf_poses.py 로 저장 (v11에서 import 가능하게)"""
    lines = [
        "# -*- coding: utf-8 -*-",
        "# shelf_poses.py — teach_shelf_poses.py로 자동 생성",
        "# 책장 시나리오 자세 정의",
        "",
        "POSES = {",
    ]
    for name, angles in poses.items():
        if angles:
            lines.append(f"    {name!r}: {angles},")
    lines.append("}")
    lines.append("")
    # 편의용 별칭
    for name in poses:
        if poses[name]:
            lines.append(f"{name} = POSES[{name!r}]")

    content = "\n".join(lines)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n💾 저장 완료: {OUTPUT_FILE}")
    print(f"  v11에서 사용: from shelf_poses import HOME, OBSERVE_A, ...")


def main():
    mc = MyCobot(PORT, BAUD)
    mc.thread_lock = True
    mc.power_on()
    time.sleep(1)

    # 시작 시 현재 자세 굳히기
    cur = mc.get_angles()
    if cur and len(cur) == 6:
        mc.send_angles(cur, 30); time.sleep(1.5)

    print("🎯 책장 시나리오 자세 티칭 시작")
    print(f"   총 {len(POSES_TO_TEACH)}개 자세를 차례로 가르칩니다.")
    print(f"   각 자세에서: 힘 풀림 → 손으로 잡기 → Enter → 재고정")
    print(f"   ⚠ 팔이 떨어지지 않게 항상 한 손으로 받치세요!")
    print(f"   💡 다른 터미널에 camera_web.py 띄워두면 화면 보며 조정 가능")
    input("\n준비됐으면 Enter로 시작...")

    poses = {}
    for name, desc in POSES_TO_TEACH:
        ang = teach_one(mc, name, desc)
        poses[name] = ang

    # 결과 저장
    if any(poses.values()):
        save_poses(poses)
    else:
        print("\n저장된 자세가 없어 파일을 만들지 않았습니다.")

    print("\n" + "="*50)
    print("📋 티칭 결과 요약:")
    for name, angles in poses.items():
        status = f"{angles}" if angles else "❌ 실패"
        print(f"  {name:15s} : {status}")
    print("="*50)
    print(f"\n팔은 마지막 자세로 굳어있어요. 안전 자세로 보내려면:")
    print(f"  mc.send_angles(HOME, 25)")


if __name__ == "__main__":
    main()
