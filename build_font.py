#!/usr/bin/env python3
"""손글씨 템플릿 스캔본으로 폰트(TTF)를 만듭니다.

    python build_font.py scans/*.jpg --name "MyHand" --name-ko "내손글씨"

결과물: out/<이름>.ttf, out/<이름>.woff2(가능한 경우), out/preview.png,
        out/cells/ (칸별로 인식된 이미지 — 확인/수정용)
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from handfont.build import build_font
from handfont.scan import load_overrides, scan_files
from handfont.vectorize import trace

PREVIEW_TEXT = [
    "안녕하세요. 오늘 날씨가 정말 좋네요!",
    "다람쥐 헌 쳇바퀴에 타고파",
    "키스의 고유조건은 입술끼리 만나야 하고 특별한 기술은 필요치 않다.",
    "The quick brown fox jumps over the lazy dog.",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
    "0123456789 !?@#$%&*()[]{}<>+-=/~",
]


def _trace_item(item):
    name, ink = item
    return name, trace(ink)


def render_preview(font_path, out_path, lines=PREVIEW_TEXT):
    from PIL import Image, ImageDraw, ImageFont
    size, pad = 64, 40
    font = ImageFont.truetype(font_path, size)
    width = max(int(font.getlength(t)) for t in lines) + 2 * pad
    height = int(len(lines) * size * 1.4) + 2 * pad
    img = Image.new("L", (width, height), 255)
    d = ImageDraw.Draw(img)
    for i, t in enumerate(lines):
        d.text((pad, pad + i * size * 1.4), t, font=font, fill=0)
    img.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scans", nargs="+", help="스캔/사진 파일 (jpg, png, pdf)")
    ap.add_argument("-o", "--out", default="out", help="출력 폴더 (기본: out)")
    ap.add_argument("--name", default="MyHandwriting", help="폰트 이름(영문)")
    ap.add_argument("--name-ko", help="폰트 이름(한글)")
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="잉크 판정 기준 0~1 (연한 펜/연필이면 0.65 정도로 올리기)")
    ap.add_argument("--weight", type=int, default=0,
                    help="획 굵기 보정(px). 양수=굵게, 음수=가늘게")
    ap.add_argument("--hangul-width", type=int, default=1000, help="한글 글자 폭 (기본 1000)")
    ap.add_argument("--space", type=int, default=280, help="띄어쓰기 폭 (기본 280)")
    ap.add_argument("--overrides", help="수정용 이미지 폴더(<글리프이름>.png 으로 개별 칸 교체)")
    ap.add_argument("--hangul-mode", choices=["auto", "jamo", "syllable"], default="auto",
                    help="한글 칸에 자모만 썼는지(jamo), 글자 전체를 썼는지(syllable). 기본은 자동 판정")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="병렬 처리 개수")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cells_dir = os.path.join(args.out, "cells")
    t0 = time.time()

    print("1) 스캔 이미지 인식")
    cells, _, presets = scan_files(args.scans, args.threshold, args.weight, debug_dir=cells_dir,
                                   hangul_mode=args.hangul_mode)
    overrides = load_overrides(args.overrides, args.threshold)
    if overrides:
        print(f"  수정 이미지 {len(overrides)}개 적용")
        cells.update(overrides)
    if not cells:
        sys.exit("인식된 글자가 없습니다. 템플릿 전체(모서리 검은 사각형 포함)가 보이게 찍어주세요.")

    print(f"2) 윤곽선 변환 ({len(cells)}자)")
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        outlines = dict(ex.map(_trace_item, cells.items(), chunksize=4))

    print("3) 폰트 생성")
    ttf = os.path.join(args.out, f"{args.name}.ttf")
    ttf, woff2 = build_font(outlines, ttf, family=args.name, family_ko=args.name_ko,
                            hangul_advance=args.hangul_width, space_width=args.space)
    preview = render_preview(ttf, os.path.join(args.out, "preview.png"))

    print(f"\n완료 ({time.time() - t0:.0f}초)")
    for p in (ttf, woff2, preview):
        if p:
            print("  " + p)
    print(f"  {cells_dir}/  ← 칸별 인식 결과")


if __name__ == "__main__":
    main()
