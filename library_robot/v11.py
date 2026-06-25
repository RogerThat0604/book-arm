#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v11.py — 도서관 책 분류 로봇 본 파이프라인

시나리오:
  책장 A(섞인 책) → ArUco 인식 + DB 카테고리 조회
                → 가까운 책 잡고 빼기
                → 책장 B의 카테고리 칸에 꽂기
                → 반복

자세 정의는 shelf_poses.py 에서 import.
tool_teach_poses.py 를 먼저 실행해 모든 자세를 가르치고 v11 실행.

실행:
  python v11.py            # 한 사이클 (책 하나 처리)
  python v11.py --all      # 책 안 보일 때까지 반복
  python v11.py --dry      # 동작 없이 검출만 (디버그)
"""

import sys
import time
import argparse
import numpy as np
import pyrealsense2 as rs
from pymycobot import MyCobot280 as MyCobot

from aruco_lib import stable_detect, best_book, FRAME_W, FRAME_H
from books_db import lookup_book

# ─── 자세 import (티칭 후 자동 생성됨) ───
try:
    from shelf_poses import (
        HOME,
        OBSERVE_A,
        OBSERVE_B,
        BASKET_과학,
        BASKET_문학,
        BASKET_역사,
    )
    POSES_OK = True
except ImportError:
    print("⚠ shelf_poses.py 가 없습니다. 먼저 tool_teach_poses.py 로 자세를 가르치세요.")
    POSES_OK = False
    # 임시값 (절대 실행되지 않게 안전장치)
    HOME = OBSERVE_A = OBSERVE_B = None
    BASKET_과학 = BASKET_문학 = BASKET_역사 = None

# ─── 로봇/시스템 설정 ───
PORT = "/dev/ttyJETCOBOT"
BAUD = 1000000

# 동작 파라미터
SPEED_TRAVEL    = 30          # 자세 간 이동 속도
SPEED_PICK      = 15          # 잡기 동작 속도(천천히)
SPEED_PLACE     = 20          # 꽂기 동작 속도
WAIT_AFTER_MOVE = 3.0         # 자세 이동 후 대기(s)
WAIT_GRIPPER    = 1.5         # 그리퍼 동작 대기

# 그리퍼 (스티로폼 책 보호용)
GRIPPER_FORCE_CLOSE = 40      # 잡을 때 힘 (스티로폼이라 50→40)
GRIPPER_FORCE_OPEN  = 50      # 풀 때

# 책 빼기/꽂기 거리 (mm)
PULL_OUT_MM   = 60            # 책 잡고 뒤로 빼는 거리
PUSH_IN_MM    = 50            # 책장에 밀어넣는 거리
APPROACH_MM   = 40            # 책 앞으로 접근하는 거리

# 카테고리 → 자세 매핑
CATEGORY_POSES = {
    "과학": "BASKET_과학",
    "문학": "BASKET_문학",
    "역사": "BASKET_역사",
}

# 디버그 카운터
processed_books = set()


# ───────────────────── 헬퍼 ─────────────────────
def get_basket_pose(category):
    """카테고리 이름 → 자세값"""
    mapping = {
        "과학": BASKET_과학,
        "문학": BASKET_문학,
        "역사": BASKET_역사,
    }
    return mapping.get(category)


def move_to(mc, pose, label="", speed=SPEED_TRAVEL, wait=WAIT_AFTER_MOVE):
    """안전한 자세 이동 + 로그"""
    if pose is None:
        print(f"⚠ 자세 '{label}' 가 None — 이동 취소")
        return False
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


def move_relative(mc, dx=0, dy=0, dz=0, speed=SPEED_PICK):
    """현재 좌표에서 상대 이동 (mm)."""
    coords = mc.get_coords()
    if not coords or len(coords) < 6:
        print("  ⚠ 좌표 읽기 실패")
        return False
    x, y, z, rx, ry, rz = coords
    target = [x + dx, y + dy, z + dz, rx, ry, rz]
    print(f"  상대이동 Δ=({dx:+.0f},{dy:+.0f},{dz:+.0f}) → ({target[0]:.0f},{target[1]:.0f},{target[2]:.0f})")
    mc.send_coords(target, speed, 1)
    time.sleep(2)
    return True


# ───────────────────── 검출 ─────────────────────
def detect_books_at_shelf_a(pipeline, align):
    """
    OBSERVE_A 자세에서 책장 A의 모든 마커 검출.
    반환: 가까운 순(큰 순) 정렬된 책 리스트.
    """
    print("📷 책장 A 스캔 중...")
    detections = stable_detect(pipeline, align, samples=20, min_hits=5)
    valid = [d for d in detections if d.get("book")]

    # 이미 처리한 책 제외
    valid = [d for d in valid if d["id"] not in processed_books]

    if not valid:
        return []

    print(f"  검출된 책 {len(valid)}권:")
    for d in valid:
        b = d["book"]
        print(f"    ID:{d['id']:3d}  {b['title']:15s} [{b['category']}]  "
              f"위치({d['cx']:.0f},{d['cy']:.0f}) {d['size']:.0f}px")
    return valid


# ───────────────────── 동작 ─────────────────────
def pick_book_from_shelf_a(mc):
    """
    책장 A 정면에서 책 한 권 빼기.
    전제: 이미 책 정면 자세(예: OBSERVE_A의 정렬 후 자세).
    동작: 그리퍼 열기 → 앞으로 접근 → 닫기 → 뒤로 빼기
    """
    print("📥 책 빼기 시작")
    gripper_open(mc)

    # 책 쪽으로 접근 (좌표 기반)
    # ⚠ 방향(dx/dy/dz)은 책장 A 배치에 따라 다름. 티칭 후 실측으로 조정.
    print("  앞으로 접근")
    if not move_relative(mc, dx=APPROACH_MM):
        return False

    # 잡기
    gripper_close(mc)

    # 빼내기
    print("  뒤로 빼기")
    if not move_relative(mc, dx=-PULL_OUT_MM):
        return False

    print("  ✅ 책 확보")
    return True


def place_book_to_basket(mc, category):
    """
    잡은 책을 책장 B의 해당 카테고리 칸에 꽂기.
    """
    basket = get_basket_pose(category)
    if basket is None:
        print(f"⚠ 카테고리 '{category}' 자세가 정의되지 않음")
        return False

    print(f"📤 {category} 칸으로 운반")
    # 안전 경유점: HOME 거치면 가운데에서 다시 방향 잡기 좋음
    move_to(mc, HOME, "HOME (경유)")
    move_to(mc, basket, f"BASKET_{category}")

    # 꽂기: 앞으로 밀어넣고 그리퍼 열기
    print("  안쪽으로 밀어넣기")
    move_relative(mc, dx=PUSH_IN_MM, speed=SPEED_PLACE)
    gripper_open(mc)

    # 뒤로 빠지기
    print("  뒤로 빠지기")
    move_relative(mc, dx=-PUSH_IN_MM, speed=SPEED_PLACE)

    print(f"  ✅ {category} 분류 완료")
    return True


def process_one_book(mc, pipeline, align, dry_run=False):
    """한 권 처리 사이클: 검출 → 잡기 → 분류 → HOME"""
    # 1) 책장 A 보러 가기
    move_to(mc, OBSERVE_A, "OBSERVE_A")

    # 2) 검출
    books = detect_books_at_shelf_a(pipeline, align)
    if not books:
        print("📚 책장 A에 처리할 책이 없습니다 — 사이클 종료")
        return None

    # 3) 가장 가까운(큰) 책 선택
    target = books[0]
    info = target["book"]
    print(f"\n🎯 선택: ID:{target['id']} '{info['title']}' [{info['category']}]")

    if dry_run:
        print("  (dry-run 모드: 동작 없이 종료)")
        processed_books.add(target["id"])
        return target

    # 4) 잡기
    if not pick_book_from_shelf_a(mc):
        print("⚠ 잡기 실패 — HOME 복귀")
        move_to(mc, HOME, "HOME")
        return None

    # 5) 분류
    if not place_book_to_basket(mc, info["category"]):
        print("⚠ 분류 실패 — HOME 복귀")
        move_to(mc, HOME, "HOME")
        return None

    # 6) HOME 복귀
    move_to(mc, HOME, "HOME")

    processed_books.add(target["id"])
    return target


# ───────────────────── 메인 ─────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="책 없을 때까지 반복")
    parser.add_argument("--dry", action="store_true", help="동작 없이 검출만")
    args = parser.parse_args()

    if not POSES_OK:
        print("자세 정의가 없어 실행할 수 없습니다.")
        print("  먼저: python tool_teach_poses.py")
        sys.exit(1)

    print("=" * 50)
    print("🤖 도서관 책 분류 로봇 v11")
    print("=" * 50)

    # 로봇 연결
    mc = MyCobot(PORT, BAUD)
    mc.thread_lock = True
    mc.power_on()
    time.sleep(1)

    # 카메라 시작
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    print("📷 RealSense 시작 (color + depth)")
    time.sleep(1)

    try:
        # 시작 자세
        move_to(mc, HOME, "HOME (시작)")

        cycle = 0
        while True:
            cycle += 1
            print(f"\n━━━━━━ 사이클 {cycle} ━━━━━━")
            result = process_one_book(mc, pipeline, align, dry_run=args.dry)

            if result is None:
                break
            if not args.all:
                break

        # 종료: HOME 복귀
        move_to(mc, HOME, "HOME (종료)")
        print(f"\n✅ 완료. 처리된 책: {len(processed_books)}권")

    except KeyboardInterrupt:
        print("\n⚠ 사용자 중단")
        move_to(mc, HOME, "HOME (중단 복귀)")
    finally:
        pipeline.stop()
        print("📷 RealSense 종료")


if __name__ == "__main__":
    main()
