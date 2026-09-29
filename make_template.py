#!/usr/bin/env python3
"""인쇄용 손글씨 템플릿을 만듭니다.

    python make_template.py                 # full + quick 모두 생성
    python make_template.py --preset quick  # 간단 버전만
"""

import argparse

from handfont.charset import PRESETS
from handfont.template import make_templates


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS) + ["all"], default="all",
                    help="full: 한글 183칸(권장) / quick: 한글 105칸")
    ap.add_argument("-o", "--out", default="templates", help="출력 폴더")
    ap.add_argument("--font", help="템플릿 라벨용 한글 폰트 경로(자동 탐색)")
    ap.add_argument("--no-png", action="store_true", help="PDF만 생성")
    args = ap.parse_args()

    presets = list(PRESETS) if args.preset == "all" else [args.preset]
    for p in presets:
        for path in make_templates(p, args.out, args.font, png=not args.no_png):
            print(path)


if __name__ == "__main__":
    main()
