#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v11_hand.py — 손-손 데모 파이프라인 (책장 시뮬레이션)

시나리오:
  호재님 왼손(입력 책장 A) → 로봇이 책 잡기 → 카테고리별 호재님 오른손(분류 책장 B 칸)

장점: 책장 없이 v11 동작 전체 검증 가능
주의: 손이 작업영역 안에 있으므로 안전장치 필수

자세 정의 (shelf_poses_hand.py 에서 import):
  HOME           안전 대기
  OBSERVE_A      왼손 보는 자세
  PICK_A         왼손에서 책 잡는 자세 (OBSERVE_A보다 그리퍼 앞)
  OBSERVE_B      오른손 영역 보는 자세 (base 회전 후)
  PLACE_과학     오른손 왼쪽 위치
  PLACE_문학     오른손 가운데 위치
  PLACE_예술     오른손 오른쪽 위치

실행:
  python v11_hand.py             # 한 사이클
  python v11_hand.py --manual    # 각 단계마다 Enter 대기 (안전 권장)
  python v11_hand.py --dry       # 동작 없이 검출만
"""

import sys
import time
import argparse
import numpy as np
import pyrealsense2 as rs
from pymycobot import MyCobot280 as MyCobot

from aruco_lib import stable_detect, best_book, FRAME_W, FRAME_H
from books_db import lookup_book, log_robot_action

# ─── 자세 import ───
try:
    from shelf_poses_hand import (
        HOME, OBSERVE_A, PICK_A, OBSERVE_B,
        PLACE_과학, PLACE_문학, PLACE_예술,
    )
    POSES_OK = True
except ImportError:
    print("⚠ shelf_poses_hand.py 가 없습니다. tool_teach_poses_hand.py 로 자세를 먼저 가르치세요.")
    POSES_OK = False
    HOME = OBSERVE_A = PICK_A = OBSERVE_B = None
    PLACE_과학 = PLACE_문학 = PLACE_예술 = None

# ─── 로봇/시스템 설정 ───
PORT = "/dev/ttyJETCOBOT"
BAUD = 1000000

# 속도 (손 시나리오라 평소보다 느림)
SPEED_TRAVEL    = 20          # 자세 간 이동 (25→20)
SPEED_PICK      = 12          # 잡기 (15→12)
SPEED_PLACE     = 15          # 놓기

WAIT_AFTER_MOVE = 2.5
WAIT_GRIPPER    = 1.5

# 그리퍼 (스티로폼+손 보호)
GRIPPER_FORCE_CLOSE = 40
GRIPPER_FORCE_OPEN  = 50

# 안전 대기 (그리퍼 동작 직전 호재님이 손 정리할 시간)
SAFETY_PAUSE_PICK  = 3.0      # 잡기 직전 대기
SAFETY_PAUSE_PLACE = 3.0      # 놓기 직전 대기

# 영문 카테고리 → 한글 매핑 (cb_books.category)
CATEGORY_MAP = {
    "literature": "문학",
    "science":    "과학",
    "art":        "예술",
}

# 카테고리 → 자세 매핑
CATEGORY_POSES = {
    "과학": "PLACE_과학",
    "문학": "PLACE_문학",
    "예술": "PLACE_예술",
}

processed_books = set()


def get_place_pose(category):
    mapping = {
        "과학": PLACE_과학,
        "문학": PLACE_문학,
        "예술": PLACE_예술,
    }
    return mapping.get(category)


def move_to(mc, pose, label="", speed=SPEED_TRAVEL, wait=WAIT_AFTER_MOVE):
    if pose is None:
        print(f"⚠ 자세 '{label}' 가 None — 이동 취소"); return False
    print(f"→ {label}")
    mc.send_angles(pose, speed)
    time.sleep(wait)
    return True


def gripper_open(mc):
    mc.set_gripper_state(0, GRIPPER_FORCE_OPEN)
    time.sleep(WAIT_GRIPPER)


def gripper_close(mc):
    mc.set_gripper_state(1, GRIPPER_FORCE_CLOSE)
    time.sleep(WAIT_GRIPPER)


def manual_wait(manual, msg):
    """수동 모드면 Enter 대기, 아니면 그냥 진행."""
    if manual:
        input(f"⏸  {msg} — 준비됐으면 Enter ")


def safety_countdown(sec, msg):
    """카운트다운 + 안내 메시지."""
    print(f"⚠️  {msg}")
    for i in range(int(sec), 0, -1):
        print(f"   {i}...")
        time.sleep(1)


# ───────────────────── 검출 ─────────────────────
def detect_book_in_hand(pipeline, align):
    """OBSERVE_A 자세에서 왼손에 들린 책의 마커 검출."""
    print("📷 책 인식 중...")
    detections = stable_detect(pipeline, align, samples=15, min_hits=5)
    valid = [d for d in detections if d.get("book")]
    valid = [d for d in valid if d["id"] not in processed_books]

    if not valid:
        return None
    return valid[0]   # 가장 큰(가까운) 마커


# ───────────────────── 메인 사이클 ─────────────────────
def process_one_book(mc, pipeline, align, manual=False, dry_run=False):
    """한 권 처리: 왼손 인식 → 잡기 → 오른손에 놓기"""

    # 1) 왼손 관찰 자세
    manual_wait(manual, "왼손에 책을 들고 OBSERVE_A 위치로 가져옵니다")
    move_to(mc, OBSERVE_A, "OBSERVE_A (왼손 관찰)")

    # 2) 책 인식
    print("\n📚 왼손의 책 인식 시도")
    target = detect_book_in_hand(pipeline, align)
    if target is None:
        print("❌ 책 인식 실패 — 왼손 위치/마커 방향 확인")
        return None

    info = target["book"]
    db_category = info["category"]
    category    = CATEGORY_MAP.get(db_category, db_category)
    info["category"] = category

    print(f"\n🎯 인식: ID:{target['id']}  '{info['title']}'  "
          f"[{db_category} → {category}]")

    if dry_run:
        print("  (dry-run: 동작 없이 종료)")
        processed_books.add(target["id"])
        return target

    # 3) 책 잡으러 가기
    manual_wait(manual, "이제 책을 잡으러 갑니다. 왼손은 그대로 들고 있으세요")
    move_to(mc, PICK_A, "PICK_A (왼손 책 위치)")

    # ⚠️ 안전 대기
    safety_countdown(SAFETY_PAUSE_PICK,
                     "그리퍼가 닫힙니다. 손가락 위치 확인! 책 위쪽만 잡고 있으세요")

    gripper_open(mc)
    time.sleep(0.5)
    print("✋ 그리퍼 닫기")
    gripper_close(mc)

    manual_wait(manual, "책을 잡았어요. 왼손에서 책을 놓아주세요. 그리고 오른손을 분류 위치로 옮기세요")

    # 4) 분류 자세로 이동 (OBSERVE_B 경유)
    place_pose = get_place_pose(category)
    if place_pose is None:
        print(f"⚠ 카테고리 '{category}' 자세 미정의 → HOME 복귀")
        move_to(mc, HOME, "HOME (긴급 복귀)")
        log_robot_action("place", "fail",
                         error_message=f"unknown category {category}",
                         parameters={"book_id": target["id"]})
        return None

    move_to(mc, OBSERVE_B, "OBSERVE_B (오른손 영역)")
    move_to(mc, place_pose, f"PLACE_{category}")

    # 5) 놓기 직전 안전 대기
    safety_countdown(SAFETY_PAUSE_PLACE,
                     f"'{category}' 위치에 책을 놓습니다. 오른손이 받을 준비됐는지 확인")

    print("🖐 그리퍼 열기")
    gripper_open(mc)

    manual_wait(manual, "책을 받으셨으면 손을 작업영역 밖으로 빼주세요")

    # 6) HOME 복귀
    move_to(mc, HOME, "HOME")

    # DB 로그
    log_robot_action(
        action="hand_to_hand_sort",
        status="success",
        parameters={
            "book_id":  target["id"],
            "title":    info["title"],
            "category": category,
        },
    )

    processed_books.add(target["id"])
    return target


# ───────────────────── main ─────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",    action="store_true", help="여러 권 반복")
    parser.add_argument("--manual", action="store_true",
                        help="각 단계 Enter로 진행 (안전 권장)")
    parser.add_argument("--dry",    action="store_true", help="동작 없이 검출만")
    args = parser.parse_args()

    if not POSES_OK:
        print("자세 정의가 없어 실행할 수 없습니다.")
        print("  먼저: python tool_teach_poses_hand.py")
        sys.exit(1)

    print("=" * 55)
    print("🤖 도서관 책 분류 로봇 v11 — 손-손 시뮬레이션")
    print("=" * 55)
    print("⚠️  안전 안내:")
    print("   - 책 위쪽만 잡고, 그리퍼는 책 아래쪽에 닿게")
    print("   - 그리퍼 닫히기 직전 카운트다운 시 손가락 점검")
    print("   - 위험하면 즉시 Ctrl+C")
    if args.manual:
        print("   - 수동 모드: 각 단계 Enter로 진행")
    print("=" * 55)
    input("준비됐으면 Enter로 시작...")

    mc = MyCobot(PORT, BAUD)
    mc.thread_lock = True
    mc.power_on()
    time.sleep(1)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    print("📷 RealSense 시작")
    time.sleep(1)

    try:
        move_to(mc, HOME, "HOME (시작)")

        cycle = 0
        while True:
            cycle += 1
            print(f"\n━━━━━━ 사이클 {cycle} ━━━━━━")
            result = process_one_book(mc, pipeline, align,
                                      manual=args.manual,
                                      dry_run=args.dry)
            if result is None:
                break
            if not args.all:
                break
            manual_wait(args.manual or True,
                        "다음 책으로 진행하려면 Enter (중단은 Ctrl+C)")

        move_to(mc, HOME, "HOME (종료)")
        print(f"\n✅ 완료. 처리: {len(processed_books)}권")

    except KeyboardInterrupt:
        print("\n⚠ 사용자 중단")
        try:
            gripper_open(mc)            # 그리퍼 열어 책 떨어뜨리지 않게
            move_to(mc, HOME, "HOME (중단 복귀)")
        except Exception:
            pass
    finally:
        pipeline.stop()
        print("📷 RealSense 종료")


if __name__ == "__main__":
    main()
