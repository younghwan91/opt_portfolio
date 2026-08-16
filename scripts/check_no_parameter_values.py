#!/usr/bin/env python3
"""
채택 파라미터의 **값**이 산문에 적혔는지 검사한다.

왜 필요한가: `configs/` 에서 설정 파일을 걷어내도, 같은 값을 문장으로 말하는
산출물이 남는다. 실제로 두 번 놓쳤다 —

- `reports/2026-08-13-…html` 이 시총 구간과 보유 종목 수를 서술하고 있었다
- `07-experiment-log.md` §6 이 같은 구간을 적고 있었다. 바로 위 §3.4 는
  "구체적 밴드 값은 적지 않는다" 고 선언한 상태였다

둘 다 확장자 기준 점검(`*.json` 을 훑는 방식)이 놓친 자리다. **감출 대상은
파일 형식이 아니라 정보다.**

## 값이 아니라 모양을 본다

패턴에 실제 밴드 숫자를 적으면 **이 파일 자체가 유출이 된다.** 그래서 값을
모르는 채로 잡히는 표현 형태만 검사한다. 오탐이 나면 그 문장을 고치는 편이
패턴에 예외를 다는 것보다 낫다 — 예외는 다음 유출이 통과하는 구멍이 된다.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: (설명, 정규식). 값을 담지 않는다.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # "시총 하위 20%" — 유니버스를 백분위로 특정하는 서술
    ("시총 백분위로 유니버스를 특정", re.compile(r"시총\s*하위\s*\d+\s*%")),
    ("market-cap percentile disclosed", re.compile(r"bottom\s+\d+\s*%\s+by\s+market\s+cap", re.I)),
    # "$27M~$72M" / "$5M–$80M" — 시총 구간을 달러로 특정하는 서술
    (
        "시총 구간을 달러로 특정",
        re.compile(r"\$\s?\d+(?:\.\d+)?\s*[MB]\s*[~\-–—]\s*\$?\s?\d+(?:\.\d+)?\s*[MB]"),
    ),
    # "20종목을 (분기마다) 담고" — 보유 종목 수 특정.
    # 조사 `을/씩/만` 을 요구해 데이터 행수 서술("21,962종목")과 구분하고,
    # 사이에 부사구가 끼어도 잡히게 최대 세 어절까지 건너뛴다. 처음 쓴 패턴은
    # 붙어 있는 경우만 잡아서 실제 유출 문장을 통과시켰다.
    (
        "보유 종목 수 특정",
        re.compile(r"\d+\s*종목(?:을|씩|만)\s*(?:\S+\s+){0,3}?(?:담|보유|골|사)"),
    ),
    ("holdings count disclosed", re.compile(r"\bholds?\s+\d+\s+stocks?\b", re.I)),
]

#: 검사에서 제외할 경로. 스키마·기본값 문서는 값이 아니라 **필드의 존재**를
#: 설명하므로 대상이 아니다.
EXEMPT = (
    # 필터 스키마 — 값이 아니라 **필드의 의미**를 설명한다. 예로 든 숫자를
    # 지우면 파라미터가 무슨 뜻인지 읽을 수 없다. 채택 설정은 여기 예시와
    # 다른 방식(절대 밴드)을 쓰므로 예시가 채택값을 알려주지도 않는다.
    "docs/factor-system/02-universe-spec.md",
    "src/opt_portfolio/factor/universe/filters.py",
    "configs/strategy.json",  # 합성 예제 (중대형주 밴드)
    "scripts/check_no_parameter_values.py",  # 이 파일
)

#: 공개하기로 판단한 값과 그 근거.
#:
#: 검사기는 "채택 밴드"와 "기각된 밴드"를 구분할 수 없다 — 둘 다 같은 모양이다.
#: 구분은 사람이 하고, 그 판단을 여기에 **근거와 함께** 남긴다. 목록에 값을
#: 추가하는 것은 방어를 한 칸 여는 행위이므로 이유 없이는 하지 않는다
#: (CLAUDE.md §1-b — 방어 장치를 끄면 그 이유를 남긴다).
ALLOWED: dict[str, str] = {
    "$300M~5B": "기각된 중형 유니버스 (07-experiment-log §3.3, Sharpe 0.33/0.18)",
    "$100M~1B": "기각된 마이크로캡 유니버스 (§3.3, Sharpe 0.19)",
    "$100M~2B": "기각된 소형 유니버스 (§3.3, Sharpe 0.07)",
}


def scan(path: Path) -> list[str]:
    """파일에서 걸린 항목을 사람이 읽을 수 있는 줄로 돌려준다."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # 읽을 수 없으면 조용히 넘기지 않는다
        return [f"{path}: 읽을 수 없음 ({exc})"]

    # HTML 은 태그를 걷어내고 본문만 본다 — 유출은 산문에서 일어난다
    if path.suffix.lower() in (".html", ".htm"):
        text = re.sub(r"<[^>]*>", " ", text)

    hits = []
    for label, pattern in PATTERNS:
        for m in pattern.finditer(text):
            found = m.group(0).strip()
            if any(a in found for a in ALLOWED):
                continue
            line = text.count("\n", 0, m.start()) + 1
            hits.append(f"{path}:{line}: {label} — {found!r}")
    return hits


def main(argv: list[str]) -> int:
    hits: list[str] = []
    for name in argv:
        if name in EXEMPT:
            continue
        hits.extend(scan(Path(name)))

    if not hits:
        return 0

    print("채택 파라미터의 값이 산문에 적혀 있다. 이 저장소는 공개다.\n")
    for h in hits:
        print(f"  {h}")
    print(
        "\n성과·방법론은 공개하되 파라미터의 값은 가린다 (`configs/README.md`).\n"
        "값을 지우고 서술로 바꾸거나, 정말 공개할 값이면 EXEMPT 에 근거와 함께 추가한다."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
