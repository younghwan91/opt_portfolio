"""
opt-factor CLI — 운영 진입점

    opt-factor factors [--category quality]        # 팩터 카탈로그
    opt-factor status --store data.duckdb          # 스토어 커버리지
    opt-factor ingest --store data.duckdb --provider sharadar --tables sf1,sep,tickers
    opt-factor ingest --store data.duckdb --provider csv --csv PATH --kind fundamentals
    opt-factor validate --store data.duckdb --config strategy.json   # IC 리포트
    opt-factor backtest --store data.duckdb --config strategy.json
    opt-factor optimize --store data.duckdb --config strategy.json --space space.json

strategy.json 예시:
    {
      "factors": ["PER_TTM", "GP_A", "MOM_12_1"],
      "universe": {"exclude_financials": true, "min_price_usd": 5.0},
      "backtest": {"n_stocks": 20, "rebalance": "ME", "weighting": "equal"},
      "timing_ma_days": 200,
      "subscribed": ["SF1", "SEP"]
    }

space.json 예시 (walk-forward PO 탐색 공간):
    {
      "n_stocks": ["int", 10, 50],
      "weighting": ["cat", ["equal", "hrp", "black_litterman"]],
      "view_confidence": ["float", 0.05, 2.0]
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

import pandas as pd

from opt_portfolio.factor.backtest.costs import CostModel
from opt_portfolio.factor.backtest.engine import BacktestConfig
from opt_portfolio.factor.data.store import PITStore
from opt_portfolio.factor.dsl.registry import REGISTRY
from opt_portfolio.factor.pipeline import FactorPipeline, StrategyConfig
from opt_portfolio.factor.universe.filters import UniverseConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ 설정 로딩


def _build_dataclass(cls: type, payload: dict) -> object:
    """dict → dataclass. 모르는 키는 에러 (조용한 오타 무시 방지)."""
    names = {f.name for f in dc_fields(cls)}
    unknown = set(payload) - names
    if unknown:
        raise SystemExit(f"{cls.__name__} 에 없는 설정 키: {sorted(unknown)}")
    coerced = {k: tuple(v) if isinstance(v, list) else v for k, v in payload.items()}
    return cls(**coerced)


def load_strategy(path: str | Path) -> StrategyConfig:
    raw = json.loads(Path(path).read_text())
    universe = _build_dataclass(UniverseConfig, raw.pop("universe", {}))
    bt_raw = raw.pop("backtest", {})
    cost = _build_dataclass(CostModel, bt_raw.pop("cost", {}))
    backtest = _build_dataclass(BacktestConfig, {**bt_raw, "cost": cost})
    raw.setdefault("factors", [])
    return _build_dataclass(  # type: ignore[return-value]
        StrategyConfig, {**raw, "universe": universe, "backtest": backtest}
    )


def load_space(path: str | Path) -> dict:
    raw = json.loads(Path(path).read_text())
    return {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}


# ------------------------------------------------------------------ 서브커맨드


def cmd_factors(args: argparse.Namespace) -> int:
    import opt_portfolio.factor.library  # noqa: F401  (레지스트리 등록)

    specs = REGISTRY.by_category(args.category) if args.category else REGISTRY.all()
    rows = [
        {
            "name": s.name,
            "category": s.category,
            "label": s.label,
            "requires": "+".join(sorted(s.requires)),
            "auto": "✓" if s.derived_from else "",
        }
        for s in specs
    ]
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    print(f"\n총 {len(frame)}개 팩터")
    return 0


def _open_existing(path: str) -> PITStore:
    """읽기 커맨드용 — 없는 스토어는 트레이스백 대신 안내 메시지."""
    if path != ":memory:" and not Path(path).exists():
        raise SystemExit(
            f"스토어가 없습니다: {path}\n먼저 `opt-factor ingest` 로 데이터를 적재하세요."
        )
    return PITStore(path)


def cmd_status(args: argparse.Namespace) -> int:
    with _open_existing(args.store) as store:
        print(store.coverage().to_string(index=False))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    with PITStore(args.store) as store:
        if args.provider == "csv":
            return _ingest_csv(store, args)
        if args.provider == "sharadar":
            return _ingest_sharadar(store, args)
        raise SystemExit(f"알 수 없는 프로바이더: {args.provider}")


def _ingest_csv(store: PITStore, args: argparse.Namespace) -> int:
    from opt_portfolio.factor.data.sharadar import SharadarProvider

    if not args.csv or not args.kind:
        raise SystemExit("csv 프로바이더에는 --csv PATH 와 --kind 가 필요합니다")
    provider = SharadarProvider(api_key="unused")
    upsert = {
        "fundamentals": store.upsert_fundamentals,
        "prices": store.upsert_prices,
        "institutions": store.upsert_institutions,
        "insiders": store.upsert_insiders,
        "tickers": store.upsert_tickers,
    }[args.kind]
    total = 0
    if args.kind == "tickers":
        total += upsert(pd.read_csv(args.csv))
    else:
        for chunk in provider.load_csv(args.csv, args.kind):
            total += upsert(chunk)
    print(f"{args.kind}: {total}행 적재")
    return 0


def _read_ticker_file(path: Path) -> list[str]:
    """
    유니버스 파일 → 티커 목록.

    TICKERS 벌크 CSV(`ticker` 컬럼) 또는 줄/쉼표 구분 텍스트를 받는다.
    18,000종목을 명령줄 인자로 넘기는 것은 현실적이지 않으므로 이 경로가
    풀 유니버스 적재의 정규 입력이다.
    """
    if not path.exists():
        raise SystemExit(f"유니버스 파일이 없습니다: {path}")
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if "ticker" not in frame.columns:
            raise SystemExit(f"{path} 에 'ticker' 컬럼이 없습니다: {list(frame.columns)[:8]}")
        names = frame["ticker"].dropna().astype(str)
    else:
        names = pd.Series(re.split(r"[,\s]+", path.read_text()))
    out = sorted({t.strip().upper() for t in names if t and t.strip()})
    if not out:
        raise SystemExit(f"유니버스 파일이 비었습니다: {path}")
    return out


def _resolve_universe(store: PITStore, args: argparse.Namespace) -> list[str] | None:
    """명시 유니버스를 우선 쓰고, 없으면 None (= 자동 탐색으로 폴백)."""
    if args.tickers:
        return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.tickers_file:
        tickers = _read_ticker_file(Path(args.tickers_file))
        print(f"유니버스 파일: {len(tickers)}종목 ({args.tickers_file})")
        return tickers
    if args.universe == "store":
        known = store.known_tickers()
        if not known:
            raise SystemExit(
                "스토어에 티커가 없습니다. 먼저 TICKERS 를 적재하세요:\n"
                "  opt-factor ingest --store ... --provider csv --kind tickers --csv tickers.csv"
            )
        print(f"스토어 유니버스: {len(known)}종목")
        return known
    return None


def _ingest_sharadar(store: PITStore, args: argparse.Namespace) -> int:
    from opt_portfolio.factor.data.sharadar import SharadarProvider

    provider = SharadarProvider(api=args.api, chunk_size=args.chunk)
    tickers = _resolve_universe(store, args)
    if tickers is None and args.api == "direct":
        # 직판은 무필터 대량조회 시 '최신 N행'만 돌려주므로 유니버스를 먼저
        # 확정하고 티커 청크로 받는다 — 조용한 절단을 구조적으로 차단.
        tickers = provider.accessible_tickers()
        print(f"접근 가능 유니버스: {len(tickers)}종목")
        print(
            "⚠️  이 유니버스는 최근 분기 재무가 있는 종목만이다 — "
            "상장폐지 종목은 원리적으로 빠진다.\n"
            "    유료 플랜의 폐지 종목까지 적재하려면 TICKERS 벌크 CSV 로 "
            "유니버스를 먼저 확정하고\n"
            "    `--tickers` 로 명시하라 (`ingest --provider csv --kind tickers`)."
        )
    if not provider.api_key:
        raise SystemExit("API 키가 없습니다. NASDAQ_DATA_LINK_API_KEY 환경변수를 설정하세요.")
    # daily 가 빠지면 prices.mcap 이 통째로 비고, 시총 유니버스 필터·EV 팩터·
    # Black-Litterman 시장가중이 조용히 죽는다 (mcap/ev 의 출처가 DAILY 다).
    # tickers 는 마지막 — 적재된 종목 기준으로 메타를 청크 조회한다.
    tables = args.tables.split(",") if args.tables else ["sf1", "sep", "daily", "tickers"]
    for table in tables:
        total = 0
        if table == "sf1":
            for chunk in provider.fundamentals(since=args.since, tickers=tickers):
                total += store.upsert_fundamentals(chunk)
        elif table == "sep":
            for chunk in provider.prices(since=args.since, tickers=tickers):
                total += store.upsert_prices(chunk)
        elif table == "daily":
            for chunk in provider.daily_metrics(since=args.since, tickers=tickers):
                total += store.upsert_prices(chunk)
        elif table == "sf3":
            for chunk in provider.institutions(since=args.since, tickers=tickers):
                total += store.upsert_institutions(chunk)
        elif table == "sf2":
            for chunk in provider.insiders(since=args.since, tickers=tickers):
                total += store.upsert_insiders(chunk)
        elif table == "tickers":
            # 메타는 티커 목록을 명시해 청크로 받는다 — 무필터 조회는
            # limit(10,000)에서 잘려 정작 필요한 종목이 빠질 수 있다.
            wanted = tickers or store.known_tickers()
            if wanted:
                for i in range(0, len(wanted), 200):
                    total += store.upsert_tickers(provider.tickers(tickers=wanted[i : i + 200]))
            else:
                total += store.upsert_tickers(provider.tickers())
        else:
            raise SystemExit(f"알 수 없는 테이블: {table}")
        print(f"{table}: {total}행 적재")
    return 0


def _pipeline(args: argparse.Namespace) -> tuple[FactorPipeline, StrategyConfig]:
    import opt_portfolio.factor.library  # noqa: F401

    config = load_strategy(args.config)
    with _open_existing(args.store) as store:
        ctx = store.build_context(start=args.start, end=args.end, benchmark=config.benchmark)
    return FactorPipeline(ctx), config


def cmd_validate(args: argparse.Namespace) -> int:
    from opt_portfolio.factor.research.ic import forward_returns, rank_ic, summarize_ic
    from opt_portfolio.factor.research.quantiles import analyze_quantiles

    pipeline, config = _pipeline(args)
    dates = pipeline.signal_dates(config.signal_freq)
    fwd = forward_returns(pipeline.close, horizon=21).reindex(dates)

    rows = []
    for spec in config.resolved_factors():
        panel = pipeline.factor_panel(spec, dates)
        ic = summarize_ic(rank_ic(panel, fwd), horizon=1)  # 월간 샘플 — 겹침 없음
        quant = analyze_quantiles(panel, fwd, n_quantiles=5)
        rows.append(
            {
                "factor": spec.name,
                "ic_mean": round(ic.mean, 4),
                "ic_ir": round(ic.ir, 3),
                "t": round(ic.t_stat, 2),
                "spread": round(quant.spread, 4),
                "mono": round(quant.monotonicity, 2),
                "turnover": round(quant.top_turnover, 3),
            }
        )
    report = pd.DataFrame(rows).sort_values("t", ascending=False)
    print(report.to_string(index=False))
    if args.out:
        report.to_json(args.out, orient="records", indent=2)
        print(f"\n저장: {args.out}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    pipeline, config = _pipeline(args)
    result = pipeline.run(config, start=args.start, end=args.end)
    stats = result.stats()
    print(pd.Series(stats).to_string())
    print(
        "\n⚠️  단일 백테스트는 공식 성과가 아닙니다. "
        "전략 승인은 `opt-factor optimize` (walk-forward + DSR) 로 하세요."
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """백테스트 + IC 검증을 실행해 자기완결 HTML 티어시트를 만든다."""
    from opt_portfolio.factor.report import render_tearsheet
    from opt_portfolio.factor.research.ic import forward_returns, rank_ic, summarize_ic

    pipeline, config = _pipeline(args)
    result = pipeline.run(config, start=args.start, end=args.end)

    dates = pipeline.signal_dates(config.signal_freq)
    fwd = forward_returns(pipeline.close, horizon=21).reindex(dates)
    ic_rows = []
    for spec in config.resolved_factors():
        ic = summarize_ic(rank_ic(pipeline.factor_panel(spec, dates), fwd), horizon=1)
        ic_rows.append(
            {
                "팩터": spec.label,
                "IC": f"{ic.mean:.3f}",
                "IC-IR": f"{ic.ir:.2f}",
                "t": f"{ic.t_stat:.2f}",
            }
        )
    ic_table = pd.DataFrame(ic_rows).sort_values("t", ascending=False)

    with _open_existing(args.store) as store:
        coverage = store.coverage()

    universe_mask = pipeline.universe(config.universe)
    universe_avg = float(universe_mask.sum(axis=1).mean())

    html = render_tearsheet(
        result,
        title=args.title,
        config_summary={
            "종목": str(config.backtest.n_stocks),
            "리밸런싱": config.backtest.rebalance,
            "비중": config.backtest.weighting,
            "비용": f"{config.backtest.cost.linear_rate * 1e4:.0f}bp",
            "팩터": str(len(config.factors)),
        },
        ic_table=ic_table,
        coverage=coverage,
        universe_avg=universe_avg,
    )
    out = Path(args.out)
    out.write_text(html)
    print(f"티어시트 저장: {out}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    from opt_portfolio.factor.optimize.walkforward import run_walk_forward

    pipeline, config = _pipeline(args)
    space = load_space(args.space)
    result = run_walk_forward(
        pipeline.evaluator(config),
        space,
        pd.DatetimeIndex(pipeline.close.index),
        method=args.method,
        n_trials_per_fold=args.trials,
        min_train_years=args.min_train_years,
        embargo_days=args.embargo,
    )
    print(f"폴드 수:          {len(result.folds)}")
    print(f"총 시도:          {result.n_trials_total}")
    print(f"OOS Sharpe:       {result.sharpe():.3f}")
    dsr = result.deflated_sharpe()
    print(f"Deflated Sharpe:  {dsr:.3f}  ({'유의' if dsr >= 0.95 else '우연과 구분 불가'})")
    print("\n폴드별 선택 파라미터:")
    print(result.param_stability().to_string())
    if args.out:
        result.oos_returns.to_json(args.out)
        print(f"\nOOS 수익률 저장: {args.out}")
    return 0


# ------------------------------------------------------------------ 엔트리


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="opt-factor", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("factors", help="팩터 카탈로그")
    p.add_argument("--category", default=None)
    p.set_defaults(fn=cmd_factors)

    p = sub.add_parser("status", help="스토어 커버리지")
    p.add_argument("--store", required=True)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("ingest", help="데이터 적재")
    p.add_argument("--store", required=True)
    p.add_argument("--provider", choices=["sharadar", "csv"], required=True)
    p.add_argument("--tables", default=None, help="sf1,sep,daily,sf2,sf3,tickers")
    p.add_argument(
        "--api",
        default="direct",
        choices=["direct", "ndl"],
        help="sharadar 직판(기본) 또는 Nasdaq Data Link 폴백",
    )
    p.add_argument("--since", default=None)
    p.add_argument("--tickers", default=None, help="쉼표 구분 종목 제한 (파일럿용)")
    p.add_argument(
        "--tickers-file",
        default=None,
        help="유니버스 파일 (TICKERS 벌크 CSV 또는 줄/쉼표 구분 텍스트) — 풀 유니버스 적재용",
    )
    p.add_argument(
        "--universe",
        default="auto",
        choices=["auto", "store"],
        help="auto=최근 분기 재무로 탐색(폐지 종목 누락), store=스토어 tickers 테이블 사용",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="티커 청크 크기 덮어쓰기 — 기본값은 5년 기준이라 풀 히스토리는 줄여야 한다",
    )
    p.add_argument("--csv", default=None)
    p.add_argument(
        "--kind",
        default=None,
        choices=["fundamentals", "prices", "institutions", "insiders", "tickers"],
    )
    p.set_defaults(fn=cmd_ingest)

    for name, fn in [
        ("validate", cmd_validate),
        ("backtest", cmd_backtest),
    ]:
        p = sub.add_parser(name)
        p.add_argument("--store", required=True)
        p.add_argument("--config", required=True)
        p.add_argument("--start", default=None)
        p.add_argument("--end", default=None)
        p.add_argument("--out", default=None)
        p.set_defaults(fn=fn)

    p = sub.add_parser("report", help="HTML 티어시트 생성")
    p.add_argument("--store", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--out", default="tearsheet.html")
    p.add_argument("--title", default="Factor Strategy Tearsheet")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("optimize", help="walk-forward PO — 공식 성과 경로")
    p.add_argument("--store", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--space", required=True)
    p.add_argument("--method", default="bayesian", choices=["bayesian", "random", "grid"])
    p.add_argument("--trials", type=int, default=24)
    p.add_argument("--min-train-years", type=float, default=5.0)
    p.add_argument("--embargo", type=int, default=21)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_optimize)

    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
