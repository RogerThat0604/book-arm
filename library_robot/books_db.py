#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
books_db.py — books.db 조회 헬퍼

사용 (다른 스크립트에서):
  from books_db import lookup_book
  info = lookup_book(7)
  # → {"marker_id": 7, "title": "...", "category": "역사", ...} or None
"""

import sqlite3

DB_PATH = "books.db"


def lookup_book(marker_id, db_path=DB_PATH):
    """ArUco ID로 책 정보 조회. 없으면 None."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM books WHERE marker_id = ?", (int(marker_id),)
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    except sqlite3.Error as e:
        print(f"DB 오류: {e}")
        return None


def list_all_books(db_path=DB_PATH):
    """등록된 책 전체 목록(디버그용)."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM books ORDER BY marker_id"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except sqlite3.Error as e:
        print(f"DB 오류: {e}")
        return []


if __name__ == "__main__":
    # 단독 실행 시 DB 내용 확인
    books = list_all_books()
    if not books:
        print("등록된 책 없음. books_db_init.py 먼저 실행하세요.")
    else:
        print(f"등록된 책 {len(books)}권:")
        for b in books:
            print(f"  [{b['marker_id']:3d}] {b['title']:20s} → {b['category']}")
