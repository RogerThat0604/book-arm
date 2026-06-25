#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hangul_draw.py — OpenCV 영상에 한글 그리기 헬퍼

OpenCV의 putText()는 한글 미지원이므로, PIL로 잠깐 변환해 그린다.
나눔고딕 폰트를 우선 사용, 없으면 시스템 한글 폰트 탐색.

사용:
  from hangul_draw import put_hangul
  frame = put_hangul(frame, "안녕 한글", (x, y), size=20, color=(0,255,0))
"""

import os
import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# 폰트 후보: 나눔고딕 우선
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def _find_font():
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    # 시스템 전체에서 한글 폰트 검색
    for pattern in [
        "/usr/share/fonts/**/Nanum*.ttf",
        "/usr/share/fonts/**/NotoSansCJK*.ttc",
        "/usr/share/fonts/**/*Hangul*.ttf",
    ]:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None


_FONT_PATH = _find_font()
_FONT_CACHE = {}


def _get_font(size):
    if _FONT_PATH is None:
        return None
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.truetype(_FONT_PATH, size)
    return _FONT_CACHE[size]


def put_hangul(frame, text, pos, size=20, color=(0, 255, 0), bg=None):
    """
    OpenCV BGR 영상에 한글 텍스트 그리기.
      pos: (x, y) — 텍스트 좌상단 좌표
      color: (B, G, R)
      bg:   (B, G, R) or None — 배경색 박스(가독성↑)
    """
    font = _get_font(size)
    if font is None:
        # 폰트 없으면 영문으로 fallback
        cv2.putText(frame, text.encode('ascii', 'replace').decode(), pos,
                    cv2.FONT_HERSHEY_SIMPLEX, size / 30, color, 2)
        return frame

    # OpenCV BGR → PIL RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    # 배경 박스(선택)
    if bg is not None:
        try:
            bbox = draw.textbbox(pos, text, font=font)
            pad = 3
            draw.rectangle(
                [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                fill=(bg[2], bg[1], bg[0])  # BGR → RGB
            )
        except AttributeError:
            pass

    # 텍스트 (PIL은 RGB)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    print(f"감지된 한글 폰트: {_FONT_PATH or '없음 (sudo apt install fonts-nanum 필요)'}")
    # 테스트
    img = np.zeros((200, 600, 3), dtype=np.uint8)
    img = put_hangul(img, "한글 테스트: 과학·문학·역사", (10, 50), size=30, color=(0, 255, 0))
    img = put_hangul(img, "ID:001 [과학] 코스모스", (10, 120), size=24, color=(255, 255, 0), bg=(40, 40, 40))
    cv2.imwrite("hangul_test.png", img)
    print("테스트 이미지 저장: hangul_test.png")
