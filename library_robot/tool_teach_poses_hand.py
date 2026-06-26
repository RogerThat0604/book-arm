#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tool_teach_poses_hand.py — 손-손 시나리오 자세 티칭

가르칠 자세:
  HOME           안전 대기
  OBSERVE_A      왼손(입력) 보는 자세
  PICK_A         왼손에서 책 잡는 자세 (OBSERVE_A에서 그리퍼 앞으로)
  OBSERVE_B      오른손 영역 보는 자세 (base 반대편)
  PLACE_과학     오른손 → 과학 위치 (왼쪽)
  PLACE_문학     오른손 → 문학 위치 (가운데)
  PLACE_예술     오른손 → 예술 위치 (오른쪽)

결과: shelf_poses_hand.py 자동 생성 → v11_hand.py 에서 import

실행:
  python tool_teach_poses_hand.py
  (tool_camera_web.py 같이 띄워서 시야 확인하면 더 쉬움)
"""

import time
import os
from pymycobot import MyCobot280 as MyCobot

PORT = "/dev/ttyJETCOBOT"; BAUD = 1000000
OUTPUT_FILE = "shelf_poses_hand.py"


POSES_TO_TEACH = [
    ("HOME",
     "안전한 대기 자세 (시작·종료 자세)"),

    ("OBSERVE_A",
     "왼손에 든 책의 마커를 카메라가 잘 볼 수 있는 자세\n"
     "    그리퍼는 책에서 약 15cm 떨어진 위치"),

    ("PICK_A",
     "OBSERVE_A에서 그리퍼가 책 쪽으로 접근해 잡는 자세\n"
     "    책 중앙/아래쪽에 그리퍼가 위치 (호재님 손가락과 분리)"),

    ("OBSERVE_B",
     "base를 분류 책장 쪽으로 회전한 자세 (예: 180도)\n"
     "    호재님 오른손 영역이 카메라에 보이는 자세"),

    ("PLACE_과학",
     "오른손이 '과학' 위치(예: 왼쪽)에 있을 때 책을 놓을 자세\n"
     "    그리퍼가 오른손 손바닥 5cm 위쯤"),

    ("PLACE_문학",
     "오른손이 '문학' 위치(예: 가운데)에 있을 때 책을 놓을 자세"),

    ("PLACE_예술",
     "오른손이 '예술' 위치(예: 오른쪽)에 있을 때 책을 놓을 자세"),
]


def teach_one(mc, name, desc):
    print(f"\n{'='*55}")
    print(f"[{name}]")
    for line in desc.split("\n"):
        print(f"  {line}")
    print(f"{'='*55}")
    input(f"  팔 받칠 준비됐으면 Enter → 힘 풀림 ")
    mc.release_all_servos()
    time.sleep(0.3)
    print(f"  손으로 자세 잡으세요 (camera_web 화면 확인하면서)")
    input(f"  자세 잡았으면 Enter... ")

    angles = None
    for _ in range(6):
        angles = mc.get_angles()
        if angles and len(angles) == 6:
            break
        time.sleep(0.3)

    if not angles or len(angles) != 6:
        print(f"  ⚠ 각도 읽기 실패 → 건너뜀")
        return None

    angles = [round(a, 1) for a in angles]
    coords = mc.get_coords()
    coords = [round(c, 1) for c in coords] if coords else None

    print(f"  ✅ 기록: {angles}")
    if coords:
        print(f"     좌표: {coords}")

    # 재고정 (처짐 방지)
    mc.send_angles(angles, 30)
    time.sleep(1.5)
    print(f"  (자세 재고정 완료)")
    return angles


def save_poses(poses):
    lines = [
        "# -*- coding: utf-8 -*-",
        "# shelf_poses_hand.py — tool_teach_poses_hand.py로 자동 생성",
        "# 손-손 시나리오 자세 정의",
        "",
        "POSES = {",
    ]
    for name, angles in poses.items():
        if angles:
            lines.append(f"    {name!r}: {angles},")
    lines.append("}")
    lines.append("")
    for name in poses:
        if poses[name]:
            lines.append(f"{name} = POSES[{name!r}]")

    content = "\n".join(lines)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n💾 저장 완료: {OUTPUT_FILE}")
    print(f"  사용법: from shelf_poses_hand import HOME, OBSERVE_A, ...")


def main():
    mc = MyCobot(PORT, BAUD)
    mc.thread_lock = True
    mc.power_on()
    time.sleep(1)

    cur = mc.get_angles()
    if cur and len(cur) == 6:
        mc.send_angles(cur, 30); time.sleep(1.5)

    print("=" * 55)
    print("🎯 손-손 시나리오 자세 티칭")
    print("=" * 55)
    print(f"총 {len(POSES_TO_TEACH)}개 자세를 차례로 가르칩니다.")
    print("각 자세에서: 힘 풀림 → 손으로 잡기 → Enter → 재고정")
    print("⚠ 팔이 떨어지지 않게 항상 한 손으로 받치세요!")
    print("💡 다른 터미널에 camera_web 띄워두면 시야 보며 조정 가능")
    print("=" * 55)
    input("\n준비됐으면 Enter로 시작...")

    poses = {}
    for name, desc in POSES_TO_TEACH:
        ang = teach_one(mc, name, desc)
        poses[name] = ang

    if any(poses.values()):
        save_poses(poses)
    else:
        print("\n저장된 자세 없음.")

    print("\n" + "="*55)
    print("📋 티칭 결과:")
    for name, angles in poses.items():
        status = f"{angles}" if angles else "❌ 실패"
        print(f"  {name:15s} : {status}")
    print("="*55)
    print("\n팔은 마지막 자세로 굳어있어요.")
    print("이제: python v11_hand.py --manual --dry 로 검증 시작!")


if __name__ == "__main__":
    main()
