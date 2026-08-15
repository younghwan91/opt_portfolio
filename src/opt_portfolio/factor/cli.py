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
        "actions": store.upsert_actions,
        "sp500": store.upsert_sp500,
        "daily": store.upsert_prices,
    }[args.kind]
    # 벌크 CSV 는 전체 유니버스(~18,000종목)다. 유니버스를 지정하면 적재
    # 단계에서 걸러 스토어와 적재 시간을 필요한 만큼만 쓴다.
    wanted = _resolve_universe(store, args)
    keep = set(wanted) if wanted else None
    total = skipped = 0
    for chunk in provider.load_csv(args.csv, args.kind):
        if keep is not None and "ticker" in chunk.columns:
            before = len(chunk)
            chunk = chunk[chunk["ticker"].astype(str).str.upper().isin(keep)]
            skipped += before - len(chunk)
        total += upsert(chunk)
    if skipped:
        print(f"  ↳ 유니버스 밖 {skipped}행 제외")
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
        rescaled = 0
        if table == "sf1":
            for chunk in provider.fundamentals(since=args.since, tickers=tickers):
                total += store.upsert_fundamentals(chunk)
        elif table == "sep":
            for chunk in provider.prices(since=args.since, tickers=tickers):
                # 벤더가 분할·배당으로 과거 수정주가를 다시 계산했는지 겹침
                # 구간에서 확인하고, 그랬다면 저장된 히스토리를 먼저 맞춘다.
                # 이걸 건너뛰면 경계에 분할 배수만큼 가짜 수익률이 남는다.
                rescaled += store.rescale_prices(store.detect_adjustment_factors(chunk))
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
        elif table == "actions":
            for chunk in provider.actions(since=args.since):
                total += store.upsert_actions(chunk)
        elif table == "sp500":
            for chunk in provider.sp500(since=args.since):
                total += store.upsert_sp500(chunk)
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
        if rescaled:
            print(f"  ↳ 수정주가 소급 재조정으로 기존 {rescaled}행을 함께 맞췄습니다")
    return 0


def _pipeline(args: argparse.Namespace) -> tuple[FactorPipeline, StrategyConfig]:
    import opt_portfolio.factor.library  # noqa: F401

    config = load_strategy(args.config)
    # 후보 유니버스를 미리 좁히지 않으면 패널이 (전 종목 × 전 거래일) 로
    # 만들어진다. 전체 미장이면 22,000종목 × 7,000일 = 1.5억 셀 × 팩터 수라
    # 15GB 머신에서는 OOM 으로 죽는다 (2026-08-12 실측). 유니버스 필터는
    # 어차피 밴드 밖 종목을 버리므로, 로딩 단계에서 거르는 게 맞다.
    tickers = _read_ticker_file(Path(args.tickers_file)) if args.tickers_file else None
    if tickers:
        print(f"후보 유니버스: {len(tickers)}종목 ({args.tickers_file})")
        # 벤치마크는 유니버스 밖이라도 반드시 싣는다 — 마켓타이밍·베타 팩터가
        # 이걸 쓴다. 유니버스 파일에 SPY 를 넣어야 한다는 것을 사용자가
        # 기억하게 만드는 설계는 틀렸다.
        if config.benchmark and config.benchmark not in tickers:
            tickers = [*tickers, config.benchmark]
    with _open_existing(args.store) as store:
        ctx = store.build_context(
            start=args.start,
            end=args.end,
            tickers=tickers,
            benchmark=config.benchmark,
            # 팩터·엔진이 실제로 쓰는 넷만 싣는다. open/high/low/dividends/ev 는
            # 어디서도 참조되지 않는데 패널 하나가 수백 MB 라 GB 단위가 낭비된다
            # (EV 팩터도 벤더 ev 대신 mcap+debt-cashneq 로 직접 계산한다).
            price_fields=("close", "closeunadj", "volume", "mcap"),
        )
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


def cmd_holdings(args: argparse.Namespace) -> int:
    """오늘 무엇을 사는가 — 연구 결과를 실제 주문으로 옮기는 지점."""
    from opt_portfolio.factor.holdings import current_holdings, rebalance_plan

    pipeline, config = _pipeline(args)
    scores = (
        pipeline.regime_scores(config) if config.regime_conditional else pipeline.scores(config)
    )
    bt = config.backtest
    mcap = pipeline.ctx.daily.get("mcap")
    held = current_holdings(
        scores,
        pipeline.close,
        n_stocks=bt.n_stocks,
        weighting=bt.weighting,
        max_weight=bt.max_weight,
        universe=pipeline.universe(config.universe),
        market_caps=mcap.iloc[-1] if mcap is not None else None,
        cov_window=bt.cov_window,
        as_of=args.as_of,
    )
    if held.empty:
        raise SystemExit("선정된 종목이 없습니다 — 유니버스 필터가 너무 좁거나 데이터가 없습니다.")

    meta = pipeline.ctx.meta
    for col in ("name", "sector"):
        series = meta.get(col) if hasattr(meta, "get") else None
        if series is not None and hasattr(series, "reindex"):
            held[col] = series.reindex(held.index)

    exposure = pipeline.exposure(config)
    if exposure is not None:
        level = float(exposure.iloc[-1])
        state = "투자" if level > 0.5 else ("현금" if level < 0.5 else "부분")
        print(f"마켓타이밍({config.timing_ma_days}일): 익스포저 {level:.0%} → {state}\n")

    shown = held.copy()
    shown["weight"] = (shown["weight"] * 100).round(2)
    print(shown.to_string())
    print(f"\n총 {len(held)}종목 · 비중 합 {held['weight'].sum():.4f}")

    if args.current:
        current = pd.read_csv(args.current)
        if not {"ticker", "weight"} <= set(current.columns):
            raise SystemExit(f"{args.current} 에 ticker,weight 컬럼이 필요합니다")
        cur = current.set_index("ticker")["weight"].astype(float)
        if cur.sum() > 1.5:  # 퍼센트로 준 경우
            cur = cur / 100.0
        plan = rebalance_plan(held["weight"], cur)
        print("\n=== 리밸런싱 계획 ===")
        print(plan[plan["diff"].abs() > 1e-6].to_string())
        print(f"\n편도 회전율: {plan['diff'].abs().sum() / 2:.1%}")

    if args.out:
        held.to_csv(args.out)
        print(f"\n저장: {args.out}")
    return 0


def cmd_pbo(args: argparse.Namespace) -> int:
    """
    PBO (CSCV) — `05-math-spec.md` §5 의 승인 관문 ④.

    이 관문은 문서에 규정돼 있으면서 **한 번도 실행된 적이 없었다**
    (2026-08-15 전수 검토에서 발견). `probability_of_backtest_overfitting` 의
    독스트링은 "optimize 레이어가 자동으로 쌓아준다"고 적혀 있었지만,
    `SearchResult.trials` 는 목적함수 **값만** 저장하고 수익률 시계열은 버린다.

    PBO 는 walk-forward 가 아니라 **탐색 공간 전체**를 요구한다 — 각 파라미터
    조합을 전 구간에 돌려 (기간 × 조합) 행렬을 만들고, 블록을 IS/OOS 로 갈라
    "IS 최적이 OOS 에서도 상위인가"를 센다. 그래서 별도 커맨드다.
    """
    from opt_portfolio.factor.optimize.search import grid_params
    from opt_portfolio.factor.research.overfitting import probability_of_backtest_overfitting

    pipeline, config = _pipeline(args)
    space = load_space(args.space)
    evaluate = pipeline.evaluator(config)
    calendar = pd.DatetimeIndex(pipeline.close.index)
    start, end = calendar[0], calendar[-1]

    candidates = grid_params(space, args.grid_steps)
    print(f"탐색 조합 {len(candidates)}개 — 전 구간 {start.date()} ~ {end.date()}")
    columns = {}
    for i, params in enumerate(candidates, 1):
        label = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
        columns[label] = evaluate(params, start, end)
        print(f"  [{i}/{len(candidates)}] {label}")

    trial_returns = pd.DataFrame(columns)
    result = probability_of_backtest_overfitting(trial_returns, n_blocks=args.blocks)
    verdict = "통과" if result.pbo < 0.3 else "실패 — 탐색 공간을 줄여야 한다"
    print(f"\nPBO: {result.pbo:.3f}  (관문 < 0.3 → {verdict})")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    from opt_portfolio.factor.optimize.walkforward import OBJECTIVES, run_walk_forward

    pipeline, config = _pipeline(args)
    space = load_space(args.space)
    result = run_walk_forward(
        pipeline.evaluator(config),
        space,
        pd.DatetimeIndex(pipeline.close.index),
        objective=OBJECTIVES[args.objective],
        method=args.method,
        n_trials_per_fold=args.trials,
        min_train_years=args.min_train_years,
        embargo_days=args.embargo,
        train_window_years=args.train_window,
        ensemble_k=args.ensemble,
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
    p.add_argument("--tables", default=None, help="sf1,sep,daily,sf2,sf3,tickers,actions,sp500")
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
        choices=[
            "fundamentals",
            "prices",
            "institutions",
            "insiders",
            "tickers",
            "actions",
            "sp500",
            "daily",
        ],
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
        p.add_argument("--tickers-file", default=None, help="후보 유니버스 — 패널 크기를 줄인다")
        p.set_defaults(fn=fn)

    p = sub.add_parser("holdings", help="오늘 매수할 종목·비중")
    p.add_argument("--store", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--tickers-file", default=None, help="후보 유니버스 — 패널 크기를 줄인다")
    p.add_argument("--as-of", default=None, help="기준일 (기본: 최신 신호일)")
    p.add_argument("--current", default=None, help="현재 보유 CSV (ticker,weight) → 매매 계획")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_holdings)

    p = sub.add_parser("report", help="HTML 티어시트 생성")
    p.add_argument("--store", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--out", default="tearsheet.html")
    p.add_argument("--title", default="Factor Strategy Tearsheet")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("pbo", help="PBO (CSCV) — 승인 관문 ④")
    p.add_argument("--store", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--space", required=True)
    p.add_argument("--tickers-file", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--grid-steps", type=int, default=4, help="연속 축을 몇 단계로 이산화하는가")
    p.add_argument("--blocks", type=int, default=10, help="CSCV 블록 수 (짝수)")
    p.set_defaults(fn=cmd_pbo)

    p = sub.add_parser("optimize", help="walk-forward PO — 공식 성과 경로")
    p.add_argument("--store", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--space", required=True)
    p.add_argument("--method", default="bayesian", choices=["bayesian", "random", "grid"])
    p.add_argument("--trials", type=int, default=24)
    p.add_argument("--min-train-years", type=float, default=5.0)
    p.add_argument("--embargo", type=int, default=21)
    p.add_argument(
        "--train-window",
        type=float,
        default=None,
        help="롤링 윈도 학습 — 직전 N년만 학습한다 (미지정 시 확장 윈도)",
    )
    p.add_argument("--tickers-file", default=None, help="후보 유니버스 — 패널 크기를 줄인다")
    p.add_argument("--objective", default="sharpe", choices=["sharpe", "calmar"])
    p.add_argument(
        "--ensemble",
        type=int,
        default=1,
        help="상위 k개 파라미터의 검증 수익률을 평균 (1 = 최적값 하나)",
    )
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_optimize)

    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
