"""
Command Line Interface for Portfolio Management

Provides a professional CLI experience for the portfolio management system.
"""

import sys
from datetime import date

from ..analysis.backtest import BacktestEngine
from ..config import ASSETS, StrategyType
from ..core.cache import get_cache
from ..core.portfolio import Portfolio
from ..strategies.momentum import MomentumAnalyzer
from ..strategies.vaa import VAAStrategy


def print_header():
    """Print application header."""
    print("\n" + "=" * 60)
    print("🚀 최적 포트폴리오 관리 시스템")
    print("=" * 60)
    print("고급 OU 예측 기반 VAA 전략\n")


def print_menu():
    """Print main menu."""
    print("\n원하는 메뉴를 선택하세요:")
    print("1. 📊 VAA 분석 실행")
    print("2. 💼 포트폴리오 관리")
    print("3. 📈 전략 백테스트")
    print("4. 📉 모멘텀 히스토리 그래프")
    print("5. 💾 캐시 관리")
    print("6. ❌ 종료")
    return input("\n번호 입력 (1-6): ").strip()


def run_vaa_analysis():
    """Run VAA analysis interactively."""
    print("\n📊 VAA ETF 선택 분석")
    print("-" * 40)

    # 전략 선택
    print("\n전략을 선택하세요:")
    print("1. 현재 모멘텀(VAA)")
    print("2. 1개월 예측(OU)")
    print("3. 3개월 예측")
    print("4. 6개월 예측")
    print("5. 모멘텀 변화(Δ)")

    strategy_choice = input("번호 입력 (1-5, 기본 1): ").strip() or "1"

    strategy_map = {
        "1": StrategyType.CURRENT,
        "2": StrategyType.FORECAST_1M,
        "3": StrategyType.FORECAST_3M,
        "4": StrategyType.FORECAST_6M,
        "5": StrategyType.DELTA,
    }

    strategy = strategy_map.get(strategy_choice, StrategyType.CURRENT)

    try:
        vaa = VAAStrategy(use_forecasting=True)
        result = vaa.select(date.today(), strategy)

        print("\n" + "=" * 50)
        print(f"🎯 선택된 ETF: {result.selected_etf}")
        print("=" * 50)

        return result.selected_etf

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def run_portfolio_management(selected_etf: str | None = None):
    """Run portfolio management interactively."""
    print("\n💼 포트폴리오 관리")
    print("-" * 40)

    if not selected_etf:
        print("⚠️ 선택된 ETF가 없습니다. 먼저 VAA 분석을 실행합니다...")
        selected_etf = run_vaa_analysis()
        if not selected_etf:
            return

    print(f"\n🎯 VAA 선택 ETF: {selected_etf} (목표 50%)")
    print("\n현재 보유 종목을 입력하세요:")

    holdings = {}

    # Selected ETF
    while True:
        try:
            shares = int(input(f"  {selected_etf} shares: "))
            if shares >= 0:
                holdings[selected_etf] = shares
                break
            print("  Please enter a non-negative number.")
        except ValueError:
            print("  Please enter a valid number.")

    # Core holdings
    print("\n코어 종목 (각 12.5%):")
    for etf in ["SPY", "TLT", "GLD", "BIL"]:
        if etf == selected_etf:
            continue
        while True:
            try:
                shares = int(input(f"  {etf} shares: "))
                if shares >= 0:
                    holdings[etf] = shares
                    break
                print("  Please enter a non-negative number.")
            except ValueError:
                print("  Please enter a valid number.")

    # Additional cash
    while True:
        try:
            cash = float(input("\n추가 투자금 입력 ($): "))
            if cash >= 0:
                break
            print("0 이상의 숫자를 입력하세요.")
        except ValueError:
            print("유효한 숫자를 입력하세요.")

    # Create portfolio and calculate rebalancing
    portfolio = Portfolio.from_dict(holdings)
    portfolio.update_prices()

    print("\n📊 현재 포트폴리오:")
    print("-" * 40)
    allocation = portfolio.get_allocation()
    if not allocation.empty:
        print(allocation.to_string())
        print(f"\n💰 총 포트폴리오 가치: ${portfolio.total_value:,.2f}")

    # Calculate rebalancing
    print("\n⚖️ 리밸런싱 추천:")
    print("-" * 40)

    recommendations = portfolio.calculate_rebalance(selected_etf, cash)

    if "error" in recommendations:
        print(f"❌ 오류: {recommendations['error']}")
        return

    # Print transactions
    if recommendations["transactions"]:
        print("\n📋 필요 거래 내역:")
        for ticker, trans in recommendations["transactions"].items():
            action = trans["action"]
            shares = trans["shares"]
            price = trans["price"]
            value = trans.get("cost", trans.get("proceeds", 0))

            symbol = "🔴 매도" if action == "SELL" else "🟢 매수"
            print(f"  {symbol} {ticker}: {shares}주 @ ${price:.2f} = ${value:,.2f}")
    else:
        print("✅ 포트폴리오가 최적화되어 거래가 필요 없습니다.")

    # Allocation summary
    print("\n🎯 최종 배분:")
    print("-" * 50)
    print(f"{'ETF':<8} {'주식수':<8} {'목표 %':<10} {'실제 %':<10} {'오차':<10}")
    print("-" * 50)

    for ticker, error in recommendations["allocation_errors"].items():
        shares = recommendations["optimized_portfolio"].get(ticker, 0)
        print(
            f"{ticker:<8} {shares:<8} {error['target_percentage']:<10.1f} "
            f"{error['actual_percentage']:<10.1f} {error['percentage_error']:+.1f}%"
        )

    print(f"\n💰 최종 포트폴리오 가치: ${recommendations['final_portfolio_value']:,.2f}")
    print(f"💵 남은 현금: ${recommendations['remaining_cash']:,.2f}")


def run_backtest():
    """Run strategy backtest."""
    print("\n📈 전략 백테스트")
    print("-" * 40)

    while True:
        try:
            years = int(input("백테스트 기간(년, 기본 15): ") or "15")
            if 1 <= years <= 25:
                break
            print("1~25 사이의 숫자를 입력하세요.")
        except ValueError:
            print("유효한 숫자를 입력하세요.")

    print("\n백테스트 실행 중... 잠시만 기다려주세요.")

    engine = BacktestEngine()
    results = engine.run_vaa_backtest(years=years)

    if results:
        engine.plot_results(results)


def plot_momentum_history():
    """Plot momentum history."""
    print("\n📉 모멘텀 히스토리 그래프")
    print("-" * 40)

    print("1. 공격 자산군 (SPY, EFA, EEM, AGG)")
    print("2. 방어 자산군 (LQD, IEF, SHY)")
    print("3. 직접 입력")

    choice = input("번호 선택 (1-3): ").strip()

    if choice == "1":
        tickers = list(ASSETS.AGGRESSIVE_TICKERS)
        title = "공격 자산군 모멘텀"
    elif choice == "2":
        tickers = list(ASSETS.PROTECTIVE_TICKERS)
        title = "방어 자산군 모멘텀"
    elif choice == "3":
        ticker_str = input("티커 입력(쉼표로 구분): ").strip()
        tickers = [t.strip().upper() for t in ticker_str.split(",")]
        title = "사용자 지정 자산 모멘텀"
    else:
        print("잘못된 선택입니다.")
        return

    while True:
        try:
            years = int(input("조회 기간(년, 기본 2): ") or "2")
            if years > 0:
                break
            print("0보다 큰 숫자를 입력하세요.")
        except ValueError:
            print("유효한 숫자를 입력하세요.")

    from datetime import timedelta

    import matplotlib.pyplot as plt

    analyzer = MomentumAnalyzer()
    end_date = date.today()
    start_date = end_date - timedelta(days=years * 365)

    print(f"\nCalculating momentum from {start_date} to {end_date}...")
    momentum_df = analyzer.calculate_historical_momentum(tickers, start_date, end_date)

    if momentum_df.empty:
        print("No data available to plot.")
        return

    # Plot
    plt.figure(figsize=(12, 6))
    for ticker in momentum_df.columns:
        plt.plot(momentum_df.index, momentum_df[ticker], label=ticker, linewidth=2)

    plt.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Momentum Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def cache_management():
    """Cache management menu."""
    print("\n💾 캐시 관리")
    print("-" * 40)

    cache = get_cache()

    print("1. 캐시 통계 보기")
    print("2. 전체 캐시 삭제")
    print("3. 오래된 캐시 삭제(30일 초과)")
    print("4. 특정 티커 캐시 삭제")
    print("5. 데이터베이스 최적화")
    print("6. 메인 메뉴로 돌아가기")

    choice = input("\n번호 선택 (1-6): ").strip()

    if choice == "1":
        stats = cache.get_cache_stats()
        if stats.empty:
            print("캐시된 데이터가 없습니다.")
        else:
            print("\n📊 캐시 통계:")
            print(stats.to_string())
            print(f"\n총 캐시 티커 수: {len(stats)}")

    elif choice == "2":
        confirm = input("⚠️ 전체 캐시를 삭제할까요? (yes/no): ").strip().lower()
        if confirm == "yes":
            cache.clear_cache()

    elif choice == "3":
        cache.clear_cache(older_than_days=30)

    elif choice == "4":
        tickers = input("티커 입력(쉼표로 구분): ").strip()
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        confirm = input(f"{ticker_list} 캐시를 삭제할까요? (yes/no): ").strip().lower()
        if confirm == "yes":
            cache.clear_cache(tickers=ticker_list)

    elif choice == "5":
        cache.optimize()


def main():
    """Main CLI entry point."""
    print_header()

    selected_etf = None

    while True:
        choice = print_menu()

        if choice == "1":
            selected_etf = run_vaa_analysis()

        elif choice == "2":
            run_portfolio_management(selected_etf)

        elif choice == "3":
            run_backtest()

        elif choice == "4":
            plot_momentum_history()

        elif choice == "5":
            cache_management()

        elif choice == "6":
            print("\n👋 안녕히 가세요!")
            sys.exit(0)

        else:
            print("잘못된 선택입니다. 다시 시도하세요.")


if __name__ == "__main__":
    main()
