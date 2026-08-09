# opt_portfolio — 작업 규약

미국 주식 팩터 투자 엔진 + 기존 ETF 자산배분 시스템.
**두 서브시스템은 격리되어 있다.** 신규 팩터 코드는 `src/opt_portfolio/factor/` 이하에만 쓴다.
기존 `strategies/`(VAA·OU), `analysis/`, `core/` 는 보존 대상이다.
공유하는 것은 `analysis/metrics.py`, `analysis/risk.py`, `utils/visualization.py` 뿐이다.

## 명령어

```bash
make install     # uv sync --extra dev
make test        # pytest + coverage
make test-one T=tests/factor/test_dsl.py::TestX::test_y
make lint        # ruff check + ruff format --check
make format      # ruff --fix + format
make typecheck   # mypy src/
```

패키지 관리는 **uv** 다 (`uv.lock`). `pip install` 하지 않는다.

팩터 엔진 CLI: `opt-factor {ingest,status,validate,backtest,report,optimize}`.
전체 사용법은 `docs/factor-system/04-data-contract.md` §5 참조.

## 절대 규칙

### 1. 조용한 절단 금지 — 이 프로젝트의 지배적 실패 유형

이 저장소의 실데이터 버그는 **전부** 예외 없이 "성공" 로그와 함께 발생했다.
페이지네이션이 5년 중 2년만 싣고도 정상 종료했고, 티커 청크가 조용히 잘렸다.

> 데이터를 가져오는 코드는, 요청한 것보다 적게 받으면 **반드시 예외를 던지거나
> 최소한 경고를 남긴다.** 조용히 진행하지 않는다.

- 페이지네이션 루프는 종료 조건을 "빈 응답"이 아니라 **기대 범위 도달**로 검사한다.
- 필터·조인·병합 후 행 수가 줄면 그 사실을 로그로 남긴다.
- `limit`(Sharadar 최대 10,000)에 도달한 응답은 **절단 의심 상태**로 다룬다.

### 2. 적재 후에는 반드시 수치로 확인한다

`status` 만으로는 절단을 못 잡는다. S&P500 5년 기준 정상값:

| 검사 | 기대값 |
|---|---|
| 종목당 거래일 수 | **1,237** |
| `prices.mcap` 결측 | **0** |
| 종목당 분기 수 | **19.8** |

검증 SQL은 `04-data-contract.md` §5 에 있다.

### 3. PIT 불변식을 깨지 않는다

- `datekey >= reportperiod` — **`calendardate` 가 아니다.** 비표준 결산월 기업(NKE 5월)은
  `datekey < calendardate` 가 정상이다. `validate_pit_frame()`(`factor/data/provider.py`)이
  스토어 진입 전 마지막 방어선이며, `reportperiod` 가 있으면 엄격 검증하고 없으면
  `calendardate` 대비 92일(한 분기)까지의 조기 공시를 허용한다.
- **최초 공시 우선.** 재공시(정정)는 무시한다. 시장이 처음 본 숫자가 백테스트가 볼 숫자다.
- 혼합 소스 표현식은 **늦은 쪽 공시일**을 따른다 (`Panel.avail` element-wise max).
- 추정치는 소급 금지 — 수집 시작일 이전에는 PIT 가 성립하지 않는다.

상세: `docs/factor-system/04-data-contract.md` §2.

### 4. 벤더 데이터 함정 (실측 검증됨, 추정 금지)

- Sharadar 직판은 **숫자를 문자열로** 준다 → 합산 전 `to_numeric`.
- **DAILY 의 `marketcap`/`ev` 는 백만 달러 단위** (SF1 은 달러). 미환산 시 10⁶배 왜곡.
- `assetsavg`·`equityavg`·`invcapavg` 는 **ARQ 차원에서 전부 null** → `avg_balance()` 로 직접 계산.
- 절단 방향이 ticker 필터 유무로 뒤집힌다 → **항상 티커 청크로 요청**한다.

벤더 동작을 **추측해서 코드를 쓰지 않는다.** 초판 문서가 추정으로 쓰였고 그 차이가 버그 4건을 낳았다.
불확실하면 실계정으로 확인하고 `04-data-contract.md` 에 실측 결과를 남긴다.

### 5. 완료 선언 전 증거를 제시한다

"적재되었습니다" / "동작합니다" 는 근거가 아니다. 실제 실행 출력(행 수, 날짜 범위,
테스트 결과)을 확인한 뒤에 완료라고 말한다.

## 코드 규약

- Python 3.10+, line-length **100**, ruff (lint + format), isort 는 ruff `I` 규칙이 담당.
- mypy `disallow_untyped_defs = true` — **신규 함수에는 타입 힌트가 필수다.**
- 테스트는 `tests/` (기존) / `tests/factor/` (팩터 엔진)로 나눈다.
- 커밋 메시지는 **한국어**, Conventional Commits 형식: `feat(factor): ...`, `fix(factor): ...`.
  본문에는 무엇을 고쳤는지가 아니라 **왜 그게 문제였는지**를 쓴다.

## 문서

`docs/factor-system/` 이 설계의 단일 진실 공급원이다.

| 파일 | 내용 |
|---|---|
| `00-overview.md` | 설계 개요, 데이터 소스 선정 근거 |
| `01-factor-spec.md` | 팩터 정의 |
| `02-universe-spec.md` | 유니버스 필터 |
| `04-data-contract.md` | **스토어 스키마 · PIT 규약 · 벤더 실측 · 운영 절차** |
| `05-math-spec.md` | 가중·백테스트·walk-forward 수식 |
| `06-provider-review.md` | 벤더 12종 비교 |

구현이 문서와 어긋나면 **문서를 고치거나 구현을 고친다.** 방치하지 않는다.

## 알려진 제약

- 현재 구독 티어는 **S&P500 현재 구성종목 500개 / 5년**이며 과거 편출 종목이 없다
  → **생존편향이 존재한다.** 백테스트 수익률을 액면 그대로 신뢰하지 않는다.
- 공식 성과 측정은 `backtest` 가 아니라 **`optimize`(walk-forward)** 다.
  `backtest` 는 참고용이다.
