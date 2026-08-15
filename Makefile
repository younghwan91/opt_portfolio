.PHONY: install test test-one lint format typecheck run clean help

help:
	@echo "사용 가능한 명령어:"
	@echo "  make install     - 의존성 설치 (dev 포함)"
	@echo "  make test        - 전체 테스트 실행"
	@echo "  make test-one    - 특정 테스트 실행 (예: make test-one T=tests/test_config.py::TestAllocationConfig::test_default_validation)"
	@echo "  make lint        - 코드 스타일 검사 (ruff check + format --check)"
	@echo "  make format      - 코드 자동 포맷 (ruff --fix + format)"
	@echo "  make typecheck   - 타입 검사 (mypy)"
	@echo "  make run         - 인터랙티브 메뉴 실행"
	@echo "  make clean       - 임시 파일 정리"

install:
	uv sync --extra dev

test:
	uv run pytest tests/ -v --cov=src/opt_portfolio --cov-report=term-missing

test-one:
	uv run pytest $(T) -v

lint:
	uv run ruff check src/ tests/ scripts/
	uv run ruff format --check src/ tests/ scripts/

format:
	uv run ruff check --fix src/ tests/ scripts/
	uv run ruff format src/ tests/ scripts/

typecheck:
	uv run mypy src/

run:
	python3 run.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ coverage.xml dist/ build/
	rm -f *.db *.duckdb
