#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
books_db.py — MariaDB 버전 (기존 SQLite 인터페이스 동일)

v11, aruco_lib 등 기존 코드는 수정 없이 그대로 사용 가능.
접속 정보만 아래 DB_CONFIG에 입력하면 됨.

의존성: pip install pymysql
"""

import json
import pymysql
import pymysql.cursors

# ─────────────────────────────────────────
# DB 접속 정보 (팀 공유 서버로 교체)
# ─────────────────────────────────────────
DB_CONFIG = {
    "host":     "192.168.0.9",       # 서버 IP 또는 도메인
    "port":     3306,
    "user":     "labi_user",      # DB 계정
    "password": "106a1752c19b1f58429b7a6c131dfedb",  # 비밀번호
    "database": "labi",      # DB 이름
    "charset":  "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}


def _connect():
    """DB 연결 반환. 실패 시 None."""
    try:
        return pymysql.connect(**DB_CONFIG)
    except pymysql.Error as e:
        print(f"DB 연결 실패: {e}")
        return None


# ─────────────────────────────────────────
# 기존 인터페이스 (v11·aruco_lib 호환)
# ─────────────────────────────────────────
def lookup_book(marker_id):
    """
    ArUco 마커 ID로 책 정보 조회.
    반환: dict 또는 None
    """
    conn = _connect()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM cb_books WHERE id = %s",
                (int(marker_id),)
            )
            row = cur.fetchone()
        if not row:
            return None
        row["marker_id"] = row["id"]
        row["title"]     = row.get("title_kr") or row.get("title_en") or ""
        return row
    except pymysql.Error as e:
        print(f"DB 조회 오류: {e}")
        return None
    finally:
        conn.close()


def list_all_books():
    """등록된 책 전체 목록."""
    conn = _connect()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM cb_books ORDER BY id")
            rows = cur.fetchall()
        for row in rows:
            row["marker_id"] = row["id"]
            row["title"]     = row.get("title_kr") or row.get("title_en") or ""
        return rows
    except pymysql.Error as e:
        print(f"DB 조회 오류: {e}")
        return []
    finally:
        conn.close()


# ─────────────────────────────────────────
# v11 전용: 작업 큐 + 로그
# ─────────────────────────────────────────
def get_pending_task():
    """cb_robot_tasks에서 가장 오래된 pending 작업 하나."""
    conn = _connect()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT t.*, b.title_kr AS title, b.category
                FROM cb_robot_tasks t
                LEFT JOIN cb_books b ON b.id = t.book_id
                WHERE t.status = 'pending'
                ORDER BY t.created_at ASC
                LIMIT 1
            """)
            return cur.fetchone()
    except pymysql.Error as e:
        print(f"DB 조회 오류: {e}")
        return None
    finally:
        conn.close()


def update_task_status(task_id, status):
    """cb_robot_tasks 상태 업데이트: pending/running/done/fail"""
    conn = _connect()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE cb_robot_tasks
                SET status = %s, updated_at = NOW()
                WHERE id = %s
            """, (status, task_id))
        conn.commit()
        return True
    except pymysql.Error as e:
        print(f"DB 업데이트 오류: {e}")
        return False
    finally:
        conn.close()


def log_robot_action(action, status, robot_type="mycobot280",
                     parameters=None, error_message=None, user_message=None):
    """cb_robot_control_logs에 동작 기록."""
    conn = _connect()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO cb_robot_control_logs
                    (user_message, robot_type, action, parameters,
                     status, error_message)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                user_message, robot_type, action,
                json.dumps(parameters, ensure_ascii=False) if parameters else None,
                status, error_message,
            ))
        conn.commit()
        return True
    except pymysql.Error as e:
        print(f"DB 로그 오류: {e}")
        return False
    finally:
        conn.close()


def update_book_location(book_id, zone, shelf, in_stock=1):
    """책 꽂기 완료 후 위치 업데이트."""
    conn = _connect()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE cb_books
                SET zone = %s, shelf = %s, in_stock = %s
                WHERE id = %s
            """, (zone, shelf, in_stock, book_id))
        conn.commit()
        return True
    except pymysql.Error as e:
        print(f"DB 업데이트 오류: {e}")
        return False
    finally:
        conn.close()


def test_connection():
    """DB 연결 테스트."""
    conn = _connect()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        print("✅ DB 연결 성공!")
        return True
    finally:
        conn.close()


if __name__ == "__main__":
    print("DB 연결 테스트...")
    if not test_connection():
        print("❌ 연결 실패. DB_CONFIG 확인하세요.")
        raise SystemExit(1)

    books = list_all_books()
    if not books:
        print("등록된 책 없음. schema.sql 먼저 실행하세요.")
    else:
        print(f"\n📚 등록된 책 {len(books)}권:")
        for b in books:
            print(f"  [{b['marker_id']:3d}] "
                  f"{b['title']:20s} "
                  f"[{b['category']}] "
                  f"→ {b.get('zone','?')}-{b.get('shelf','?')}")
