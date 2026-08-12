"""
운용 화면 — 계산을 스스로 하지 않고 기존 경로에 위임하는지 본다.

화면이 자체 로직을 갖는 순간 백테스트와 실전이 갈라진다. 그래서 테스트도
"무엇을 계산하는가"가 아니라 **"어디로 위임하는가"**를 검사한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opt_portfolio.factor.tui import Session, _describe, run


class TestStrategySummary:
    def test_describes_key_settings(self, tmp_path: Path) -> None:
        path = tmp_path / "strategy_x.json"
        path.write_text(
            json.dumps(
                {
                    "factors": ["A", "B", "C"],
                    "backtest": {"n_stocks": 20, "rebalance": "QE", "weighting": "equal"},
                    "timing_ma_days": 200,
                }
            )
        )

        text = _describe(path)

        assert "팩터 3개" in text
        assert "20종목" in text
        assert "QE" in text
        assert "타이밍 200일" in text

    def test_unreadable_file_does_not_raise(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json")

        assert "읽을 수 없음" in _describe(path)


class TestMenuLoop:
    def test_quit_exits_cleanly(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr("builtins.input", lambda *_: "0")

        assert run(Session("s.duckdb", "c.json")) == 0
        assert "종료" in capsys.readouterr().out

    def test_unknown_choice_does_not_crash(self, monkeypatch, capsys) -> None:
        answers = iter(["99", "0"])
        monkeypatch.setattr("builtins.input", lambda *_: next(answers))

        assert run(Session("s.duckdb", "c.json")) == 0
        assert "메뉴에 없는" in capsys.readouterr().out

    def test_failure_in_action_keeps_menu_alive(self, monkeypatch, capsys) -> None:
        """한 번의 실패로 화면이 죽으면 매달 쓰는 도구가 될 수 없다."""
        answers = iter(["4", "0"])
        monkeypatch.setattr("builtins.input", lambda *_: next(answers))

        assert run(Session("없는파일.duckdb", "c.json")) == 0
        out = capsys.readouterr().out
        assert "오류" in out or "스토어가 없습니다" in out

    def test_interrupt_returns_130(self, monkeypatch) -> None:
        def boom(*_: object) -> str:
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", boom)

        assert run(Session("s.duckdb", "c.json")) == 130


class TestDelegation:
    def test_holdings_menu_calls_cmd_holdings(self, monkeypatch) -> None:
        """화면은 계산하지 않는다 — cmd_holdings 에 그대로 넘긴다."""
        seen: dict = {}

        def fake(args) -> int:
            seen.update(vars(args))
            return 0

        monkeypatch.setattr("opt_portfolio.factor.cli.cmd_holdings", fake)
        answers = iter(["1", "0"])
        monkeypatch.setattr("builtins.input", lambda *_: next(answers))

        run(Session("my.duckdb", "my.json", "uni.txt"))

        assert seen["store"] == "my.duckdb"
        assert seen["config"] == "my.json"
        assert seen["tickers_file"] == "uni.txt"

    def test_rebalance_menu_passes_current_file(self, monkeypatch) -> None:
        seen: dict = {}
        monkeypatch.setattr(
            "opt_portfolio.factor.cli.cmd_holdings",
            lambda args: (seen.update(vars(args)), 0)[1],
        )
        answers = iter(["2", "보유.csv", "0"])
        monkeypatch.setattr("builtins.input", lambda *_: next(answers))

        run(Session("my.duckdb", "my.json"))

        assert seen["current"] == "보유.csv"


@pytest.mark.parametrize("choice", ["1", "3", "4"])
def test_every_action_is_read_only(choice: str, monkeypatch) -> None:
    """되돌릴 수 없는 동작은 없다 — 어떤 메뉴도 주문을 내지 않는다."""
    answers = iter([choice, "0"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))

    assert run(Session("없는스토어.duckdb", "없는설정.json")) == 0
