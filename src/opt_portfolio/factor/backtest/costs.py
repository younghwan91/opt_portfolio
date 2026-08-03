"""
거래비용 모델

퀀트 관점:
- 수수료(commission)는 협상 가능하지만 슬리피지는 유동성의 함수다.
  소형주 팩터일수록 슬리피지가 알파를 잠식한다 — 기본값 10bp 는
  대형주 기준이며, 유니버스가 작아질수록 올려 잡아야 한다.
- 시장충격(√ 모델)은 거래대금/일평균거래대금(ADV) 비율에 비례한다.
  운용 규모를 명시해야 계산되므로 기본 비활성 — 규모가 정해지면 켠다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """리밸런싱 1회당 거래 비용률 계산기."""

    commission_bps: float = 5.0    # 편도 수수료
    slippage_bps: float = 10.0     # 호가 스프레드 + 체결 지연
    impact_coeff: float = 0.0      # √충격 계수 (0 = 비활성)
    portfolio_value: float = 0.0   # 운용 규모 USD (충격 계산에만 사용)

    @property
    def linear_rate(self) -> float:
        """거래대금 1 단위당 선형 비용률."""
        return (self.commission_bps + self.slippage_bps) / 10_000.0

    def rebalance_cost(
        self,
        traded_weight: pd.Series,
        adv_dollars: pd.Series | None = None,
    ) -> float:
        """
        리밸런싱 비용 (포트폴리오 수익률 차감분).

        Args:
            traded_weight: 종목별 |Δ비중| (매수+매도 모두 양수)
            adv_dollars: 종목별 20일 평균 거래대금. impact_coeff > 0 이고
                portfolio_value > 0 일 때만 √충격이 추가된다.
        """
        total_traded = float(traded_weight.sum())
        cost = total_traded * self.linear_rate

        if self.impact_coeff > 0 and self.portfolio_value > 0 and adv_dollars is not None:
            trade_dollars = traded_weight * self.portfolio_value
            participation = trade_dollars / adv_dollars.reindex(traded_weight.index)
            impact = self.impact_coeff * np.sqrt(participation.clip(lower=0.0))
            cost += float((traded_weight * impact.fillna(0.0)).sum())
        return cost

    @classmethod
    def zero(cls) -> CostModel:
        """마찰 없는 세계 — 비용 민감도 분석의 기준선."""
        return cls(commission_bps=0.0, slippage_bps=0.0)
