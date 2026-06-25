#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
books_db_init.py — books.csv → books.db + ArUco 마커 PNG 일괄 생성

사용:
  1) books.csv 를 작성 (양식: marker_id,title,category,author,isbn)
  2) python books_db_init.py
  3) 결과:
     - books.db        (SQLite DB 파일)
     - markers/aruco_001.png, aruco_002.png, ...  (인쇄용 마커 이미지)

마커 인쇄 가이드:
  - 한 변 5cm 이상 권장 (책등 너비에 맞춰)
  - 흰 여백 포함해 잘라 책등에 붙이기
"""

import csv
import os
import sqlite3
import sys

import cv2
import numpy as np

try:
    from hangul_draw import put_hangul
    HANGUL_OK = True
except ImportError:
    HANGUL_OK = False

CSV_PATH    = "books.csv"
DB_PATH     = "books.db"
MARKER_DIR  = "markers"
DICT_NAME   = cv2.aruco.DICT_4X4_50    # 최대 ID 50, 4x4 격자(작아도 잘 인식)
PIXELS      = 600                       # 마커 한 변 픽셀(인쇄 품질용)
QUIET_BORDER = 80                       # 흰 여백 픽셀
LABEL_H     = 60                        # 아래 라벨 영역 높이


def ensure_csv():
    if os.path.exists(CSV_PATH):
        return
    # 예시 CSV 생성
    print(f"{CSV_PATH} 가 없어 예시를 생성합니다. 수정 후 다시 실행하세요.")
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["marker_id", "title", "category", "author", "isbn"])
        w.writerow([1, "코스모스",       "과학", "칼 세이건",   "9788983711892"])
        w.writerow([2, "이기적 유전자",   "과학", "리처드 도킨스","9788932473901"])
        w.writerow([3, "데미안",         "문학", "헤르만 헤세", "9788932917245"])
        w.writerow([4, "노인과 바다",     "문학", "헤밍웨이",   "9788937460029"])
        w.writerow([5, "조선왕조실록",    "역사", "",          ""])
        w.writerow([6, "사피엔스",       "역사", "유발 하라리", "9788934972464"])
    print(f"예시 {CSV_PATH} 생성됨. 책 정보 입력 후 다시 실행하세요.")
    sys.exit(0)


def init_db(rows):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS books (
            marker_id  INTEGER PRIMARY KEY,
            title      TEXT NOT NULL,
            category   TEXT NOT NULL,
            author     TEXT,
            isbn       TEXT
        )
    """)
    cur.execute("DELETE FROM books")
    for r in rows:
        cur.execute(
            "INSERT INTO books(marker_id,title,category,author,isbn) VALUES (?,?,?,?,?)",
            (int(r["marker_id"]), r["title"], r["category"],
             r.get("author") or None, r.get("isbn") or None)
        )
    conn.commit()
    conn.close()
    print(f"✅ DB 작성 완료: {DB_PATH}  ({len(rows)}권)")


def make_marker_png(marker_id, title, category):
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_NAME)
    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, PIXELS)

    # 흰 여백 + 아래 라벨
    h, w = img.shape
    canvas = np.ones((h + QUIET_BORDER * 2 + LABEL_H,
                      w + QUIET_BORDER * 2), dtype=np.uint8) * 255
    canvas[QUIET_BORDER:QUIET_BORDER + h,
           QUIET_BORDER:QUIET_BORDER + w] = img

    # 라벨: 한글 그리기 가능하면 책 제목·카테고리까지, 아니면 영문 fallback
    if HANGUL_OK:
        bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        label1 = f"ID:{marker_id:03d}  [{category}]"
        label2 = f"{title}"
        y_label = QUIET_BORDER + h + 8
        bgr = put_hangul(bgr, label1, (QUIET_BORDER, y_label),
                         size=28, color=(0, 0, 0))
        bgr = put_hangul(bgr, label2, (QUIET_BORDER, y_label + 30),
                         size=22, color=(0, 0, 0))
        canvas = bgr
    else:
        cv2.putText(canvas, f"ID:{marker_id:03d}",
                    (QUIET_BORDER, QUIET_BORDER + h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)

    os.makedirs(MARKER_DIR, exist_ok=True)
    path = os.path.join(MARKER_DIR, f"aruco_{marker_id:03d}.png")
    cv2.imwrite(path, canvas)
    return path


def main():
    ensure_csv()

    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("CSV에 데이터가 없습니다."); return

    init_db(rows)

    print("\nArUco 마커 PNG 생성:")
    for r in rows:
        p = make_marker_png(int(r["marker_id"]), r["title"], r["category"])
        print(f"  {p}   ← '{r['title']}' [{r['category']}]")
    print(f"\n총 {len(rows)}개 마커가 '{MARKER_DIR}/' 폴더에 생성됨.")
    print("인쇄해서 책등에 붙이세요. 한 변 ≥5cm 권장.")


if __name__ == "__main__":
    main()
