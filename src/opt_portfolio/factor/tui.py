"""
팩터 엔진 터미널 UI — 운용을 위한 화면.

`opt-factor` 서브커맨드는 재현 가능한 파이프라인용이고, 이 화면은 **매달
같은 일을 반복하는 사람**을 위한 것이다. 무엇을 살지 보고, 지금 보유와
비교하고, 성과를 확인하는 세 가지만 한다.

설계 원칙:
- 계산은 전부 기존 함수를 호출한다. 화면이 자체 로직을 갖는 순간
  백테스트와 실전이 갈라진다.
- 되돌릴 수 없는 동작은 없다. 주문을 내지 않고 **목록만 보여준다.**
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


@dataclass(frozen=True)
class Session:
    """한 화면 세션이 붙들고 있는 것 — 스토어·전략·유니버스."""

    store: str
    config: str
    tickers_file: str | None = None


def _rule(char: str = "─", width: int = 72) -> str:
    return char * width


def _header() -> None:
    print(f"\n{_rule('━')}")
    print("  팩터 엔진 — 운용 화면")
    print(_rule("━"))


def _list_configs() -> list[Path]:
    return sorted(CONFIG_DIR.glob("strategy*.json")) if CONFIG_DIR.exists() else []


def _describe(path: Path) -> str:
    """전략 파일 한 줄 요약 — 팩터 수·종목 수·리밸런싱·타이밍."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return "(읽을 수 없음)"
    bt = raw.get("backtest", {})
    timing = raw.get("timing_ma_days")
    return (
        f"팩터 {len(raw.get('factors', []))}개 · "
        f"{bt.get('n_stocks', '?')}종목 · "
        f"{bt.get('rebalance', '?')} · "
        f"{bt.get('weighting', '?')}" + (f" · 타이밍 {timing}일" if timing else "")
    )


def choose_strategy(current: str | None) -> str | None:
    configs = _list_configs()
    if not configs:
        print(f"\n{CONFIG_DIR} 에 strategy*.json 이 없습니다.")
        return current
    print(f"\n{_rule()}")
    for i, path in enumerate(configs, 1):
        mark = "→" if current and Path(current).name == path.name else " "
        print(f" {mark} {i:2d}. {path.stem:<28} {_describe(path)}")
    print(_rule())
    raw = input("번호 (엔터=유지): ").strip()
    if not raw:
        return current
    if raw.isdigit() and 1 <= int(raw) <= len(configs):
        return str(configs[int(raw) - 1])
    print("잘못된 번호입니다.")
    return current


def show_holdings(session: Session, current_csv: str | None = None) -> None:
    """오늘 매수 목록 — cmd_holdings 와 같은 경로를 탄다."""
    import argparse

    from opt_portfolio.factor.cli import cmd_holdings

    args = argparse.Namespace(
        store=session.store,
        config=session.config,
        tickers_file=session.tickers_file,
        as_of=None,
        current=current_csv,
        start=None,
        end=None,
        out=None,
    )
    print("\n계산 중… (패널 구성에 시간이 걸립니다)\n")
    cmd_holdings(args)


def show_backtest(session: Session) -> None:
    import argparse

    from opt_portfolio.factor.cli import cmd_backtest

    args = argparse.Namespace(
        store=session.store,
        config=session.config,
        tickers_file=session.tickers_file,
        start=None,
        end=None,
        out=None,
    )
    print("\n백테스트 실행 중…\n")
    cmd_backtest(args)


def show_store(session: Session) -> None:
    # PITStore 는 없는 경로를 새로 만든다. 빈 스토어를 조용히 만들어
    # "데이터가 없다"를 "데이터가 비었다"로 바꿔 보여주면 안 되므로,
    # CLI 와 같은 가드(_open_existing)를 쓴다.
    from opt_portfolio.factor.cli import _open_existing

    with _open_existing(session.store) as store:
        print()
        print(store.coverage().to_string(index=False))


MENU = """
  1. 오늘 매수 목록
  2. 리밸런싱 계획 (현재 보유 CSV 필요)
  3. 백테스트 (참고용)
  4. 데이터 상태
  5. 전략 바꾸기
  0. 종료
"""


def run(session: Session) -> int:
    _header()
    print(f"  스토어  {session.store}")
    print(f"  전략    {Path(session.config).stem}")
    if session.tickers_file:
        print(f"  유니버스 {session.tickers_file}")

    while True:
        print(MENU)
        try:
            choice = input("선택: ").strip()
            if choice == "1":
                show_holdings(session)
            elif choice == "2":
                path = input("현재 보유 CSV 경로 (ticker,weight): ").strip()
                if path:
                    show_holdings(session, current_csv=path)
            elif choice == "3":
                show_backtest(session)
            elif choice == "4":
                show_store(session)
            elif choice == "5":
                picked = choose_strategy(session.config)
                if picked:
                    session = Session(session.store, picked, session.tickers_file)
                    print(f"전략 변경: {Path(picked).stem}")
            elif choice in ("0", "q"):
                print("\n종료합니다.\n")
                return 0
            else:
                print("메뉴에 없는 번호입니다.")
        except (KeyboardInterrupt, EOFError):
            print("\n\n중단되었습니다.\n")
            return 130
        except SystemExit as exc:  # cmd_* 가 안내 메시지와 함께 종료를 요청
            print(f"\n{exc}\n")
        except Exception as exc:  # noqa: BLE001 — 화면이 한 번의 실패로 죽지 않는다
            print(f"\n오류: {type(exc).__name__}: {exc}\n")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="opt-factor-tui", description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tickers-file", default=None)
    args = parser.parse_args(argv)

    if not Path(args.store).exists():
        raise SystemExit(f"스토어가 없습니다: {args.store}")
    return run(Session(args.store, args.config, args.tickers_file))


if __name__ == "__main__":
    raise SystemExit(main())
