# 데이터 계약 — 스토어 스키마 · PIT 규약 · 벤더 실측 · 운영 절차

> 이 문서의 벤더 관련 항목은 **2026-08-05 실계정 실측**으로 검증됨.
> 추정으로 쓰인 초판의 내용은 대부분 틀렸고, 그 차이가 버그 4건을 낳았다.

## 1. 스토어 스키마 (DuckDB, 벤더 중립)

| 테이블 | 키 | 내용 | 공시일 |
|---|---|---|---|
| `fundamentals` | (ticker, calendardate, dimension) | SF1 분기 재무 | `datekey` (실제 공시일) |
| `institutions` | (ticker, calendardate) | 13F 티커 집계 | 분기말 + 45일 |
| `insiders` | (ticker, calendardate) | Form 4 분기 집계 | 분기말 + 3일 |
| `estimates` | (ticker, calendardate) | 애널리스트 추정치 | 수집 시점 스냅샷 |
| `prices` | (ticker, date) | 일별 가격·거래량·시총 (SEP + DAILY 병합) | 해당일 |
| `tickers` | (ticker) | 섹터·소재지·유형·이름 | — |
| `actions` | (ticker, date, action) | 분할·배당·상장·폐지·분사 | 해당일 |
| `sp500` | (date, ticker, action) | 지수 편입·편출 이력 (1957~) | 해당일 |

**현재 적재 상태** (2026-08-15 실측):

| 테이블 | 행수 |
|---|---|
| `prices` | 46,331,751 |
| `actions` | 672,349 |
| `fundamentals` | 654,637 |
| `sp500` | 60,172 |
| `tickers` | 44,938 |
| `institutions` · `insiders` · `estimates` | **0** |

**0행인 셋은 의도된 상태가 아니다.** `institutions` 는 벌크 CSV 리더가 실제
헤더(`date`)가 아니라 `calendardate` 를 기대해 한 번도 동작한 적이 없었다
(2026-08-15 발견, 수정됨). `estimates` 는 Sharadar 에 애널리스트 추정치가 없어
비어 있는 것이 정상이다 (`00-overview.md` §2).

빈 테이블이 조용한 오류로 번지지는 않는다 — 전략 설정의 `subscribed` 가
`["SF1","SEP"]` 이면 SF2/SF3 를 요구하는 팩터(`flow_proxy` 5종)는 `requires`
메타로 자동 제외되고 경고가 남는다. **다만 "쓸 수 있다고 문서에 적혀 있는데
실제로는 비어 있는 것"과 "설계상 없는 것"은 구분해서 적어야 한다.**

**13F 와 내부자를 분리한 이유**: 같은 분기라도 공시 지연이 다르다 (+45일 vs +3일).
한 테이블에 datekey 하나로 합치면 둘 중 하나의 가용 시점이 왜곡된다.

**prices 가 두 소스를 병합하는 이유**: SEP 은 가격, DAILY 는 시총/EV 를 같은
(ticker, date) 키에 채운다. `merge_fields=True` 로 COALESCE 업데이트하지 않으면
나중에 적재되는 쪽이 통째로 스킵된다.

## 2. PIT 불변식 (코드가 강제)

1. **`datekey >= reportperiod`** — 위반 행은 `validate_pit_frame()` 이 거부.
   ⚠️ `calendardate` 가 아니라 `reportperiod` 기준이다. `calendardate` 는
   표준화된 달력 분기말이라 비표준 결산월 기업(NKE 5월 결산)은
   `datekey < calendardate` 가 정상이다.
2. **최초 공시 우선** — 같은 (ticker, 분기) 재공시(정정)는 무시된다.
   시장이 처음 본 숫자가 백테스트가 봐야 하는 숫자다.
3. **소스별 공시일 분리** — SF1·SF3(+45일)·SF2(+3일)의 datekey 가 각각 관리되고,
   혼합 소스 표현식은 BinOp 에서 **늦은 쪽 공시일**을 따른다
   (`Panel.avail` element-wise max 전파).
4. **집계의 확정 시점** — 분기 중 도착하는 데이터(Form 4)의 분기 합계는
   분기가 끝나야 확정이다. datekey = 분기말 + 마감유예.
5. **추정치는 소급 금지** — FMP 추정치는 수집 시작일부터만 PIT 성립.

## 3. Sharadar 직판 API 실측 (2026-08-05)

```
https://api.sharadar.com/v1.0/data/{slug}?api_key=..&format=json&limit=..
```

| 항목 | 실측 결과 |
|---|---|
| 테이블 슬러그 | `fundamentals` `sep` `sfp` `daily` `sf2` `sf3` `sf3a` `tickers` `actions` `sp500` `events` `metrics` (전부 200). `prices`·`institutions` 는 403 |
| 날짜 컬럼 | **전 테이블이 `date` 로 통일** — SF1 의 datekey, SF2 의 filingdate, SF3A 의 calendardate 가 전부 `date` 로 온다 (`_DIRECT_RENAME` 에서 복원) |
| 숫자 타입 | **문자열로 옴** — 합산 전 `to_numeric` 필수 |
| DAILY 단위 | **marketcap/ev 가 백만 달러** (SF1 은 달러). 미환산 시 배수가 10⁶배 왜곡 |
| `limit` | 최대 10,000 |
| `from`/`to` | 정상 동작 (양쪽 다) |

### ⚠️ 절단 방향이 ticker 필터 유무로 뒤집힌다

결과가 `limit` 을 넘으면 서버가 일부만 돌려주는데, **어느 쪽 끝을 주는지가 다르다**:

| ticker 필터 | `sort=date.asc` 효과 | 반환 |
|---|---|---|
| **있음** | 선택까지 지배 | **가장 오래된 N행** |
| 없음 | 페이지 내부 정렬만 | **가장 최근 N행** |

→ 어댑터는 **항상 티커 청크로 요청**(필터 있음 상태 강제)하고 `from` 을 올리며
과거→최신으로 마칭한다. 무필터 대량조회 경로는 CLI 에서 제거했다.

청크 크기(`_DIRECT_CHUNK`): SEP·DAILY 5개, SF1·SF3A 100개, SF2 40개, TICKERS 200개.

### 벤더 컬럼 함정

| 컬럼 | 문제 | 대응 |
|---|---|---|
| `assetsavg` `equityavg` `invcapavg` | **ARQ 차원에서 전부 null** | `avg_balance()` 로 (전분기+당분기)/2 직접 계산 |
| `debt`/`debtusd`, `cashneq`/`cashnequsd` | 원시·USD 공존 → 리네임 충돌 | USD 우선, 충돌 원시 컬럼 선제 제거 |
| `close`/`closeadj` | 동일 | `closeadj` 우선 |

## 4. 구독 티어 실측

**현재 구독 (2026-08 기준): Bundle · 풀 히스토리.** 실적재 결과는 21,963종목 /
1997~2026 / 폐지 종목 포함이며, 생존편향은 해소됐다. 아래는 그 전 무료 티어의
실측 기록으로, **왜 유료 전환이 필요했는지의 근거**로 남긴다.

### 4.1 무료 티어의 정체 (역사 기록)

| 항목 | 결과 |
|---|---|
| 유니버스 | **S&P 500 현재 구성종목 500개** (다우 30 아님) |
| 비회원 종목 | PLUG·RIOT·FUBO 등 → 0건 |
| 히스토리 | **5년** (2021-08~) |
| 과거 편출 종목 | ❌ `sp500` 테이블에 `action=current` 만 → **생존편향 존재** |

**가격은 두 축이다** (2026-08-11 공식 문서 확인). 데이터셋 축 —
Fundamentals $19 / Prices $9 / Investors $9 / **Bundle $29**. 히스토리 축 —
`years` 를 5 / 10 / full 로 선택하며 Bundle 기준 $30 / $49 / $69.
히스토리는 나중에 비례정산으로 업그레이드된다. 직판 문서 기준 펀더멘털
풀 히스토리는 **1998년~**, 유니버스는 "nearly 18,000 active and delisted".
티어 선택 근거는 `06-provider-review.md`, 필요 Sharpe 계산은 `05-math-spec.md` 참조.

**유니버스 권한 — 해결 (2026-08-11, 벤더 직접 확인)**

> "Yes, delisted tickers are included in our paid plans - this is an
> important feature for us." — Sharadar, Vince

유료 플랜은 폐지 종목을 포함한다. 무료 티어의 500종목 제한은 무료 티어에만
걸린 것이다. **따라서 Norgate 추가 구독은 불필요하다.**

**⚠️ 그런데 유료 전환이 우리 코드의 두 지점을 깨뜨린다** (2026-08-11 코드 검토):

1. `accessible_tickers()` 는 최근 4개 분기 SF1 으로 유니버스를 찾는다 →
   **폐지 종목이 원리적으로 안 잡힌다.** 이 목록으로 적재하면 폐지 종목을
   돈 주고 받아놓고 버린다. 전체 유니버스는 **TICKERS 벌크 CSV** 로 확정하고
   `--tickers` 로 명시해야 한다.
2. TICKERS 는 날짜 커서가 없어 단일 요청인데, 18,000종목은 페이지
   한도(10,000)를 넘는다 → 무필터 조회는 이제 `TruncatedDataError` 로
   즉시 실패한다 (조용히 잘리지 않는다).

## 4-1. 구독 첫날 실행 순서

풀 히스토리 Bundle 결제 직후, 위에서 아래로 그대로 실행한다.

```bash
export SHARADAR_API_KEY=...

# ① 벤더 계약 검증 — 전사(轉寫)된 가정이 실제 응답과 맞는지 먼저 본다
uv run python scripts/record_sharadar_fixtures.py
uv run pytest tests/factor/test_vendor_contract.py -q     # skip 이 사라져야 정상

# ② 유니버스 확정 — 폐지 종목이 여기서 들어온다.
#    TICKERS 벌크 CSV 를 받아 적재한다 (API 무필터 조회는 10,000 에서 잘린다).
opt-factor ingest --store us.duckdb --provider csv --kind tickers --csv tickers.csv

# ③ 본 적재 — 유니버스를 스토어에서 가져온다 (자동 탐색은 폐지 종목을 놓친다).
#    풀 히스토리는 티커당 행수가 5년의 5~6배라 청크를 줄인다.
opt-factor ingest --store us.duckdb --provider sharadar --universe store \
  --tables sf1,sep,daily,sf3,sf2 --chunk 1

# ④ 수치 검증 — 아래 SQL 로 절단 여부를 확인한다 (status 만으로는 부족)
opt-factor status --store us.duckdb

# ⑤ 공식 성과
opt-factor optimize --store us.duckdb \
  --config configs/strategy.json --space configs/space.json
```

`configs/strategy.json` 과 `configs/space.json` 은 즉시 실행 가능한 기본값이며
`tests/factor/test_shipped_configs.py` 가 코드와의 정합을 지킨다. 구독 데이터셋이
늘면 `subscribed` 를 갱신한다 — 미구독 팩터는 자동 제외된다.

**②가 핵심이다.** `--universe store` 없이 돌리면 `accessible_tickers()` 자동
탐색이 최근 분기 재무가 있는 종목만 찾아, 돈 주고 받은 폐지 종목을 그대로 버린다.

## 5. 운영 절차

```bash
export SHARADAR_API_KEY=...        # 직판 (기본)
# 또는 NASDAQ_DATA_LINK_API_KEY + --api ndl 로 폴백

# 전체 적재 — 유니버스는 자동 탐색(accessible_tickers)되고 청크로 순회한다.
# tickers 는 마지막에 둔다 (적재된 종목 기준으로 메타를 청크 조회하므로).
opt-factor ingest --store us.duckdb --provider sharadar \
  --tables sf1,sep,daily,sf3,sf2,tickers

# 특정 종목만 (파일럿)
opt-factor ingest --store us.duckdb --provider sharadar --tables sf1,sep \
  --tickers AAPL,MSFT,NVDA

# 일간 증분 (cron)
opt-factor ingest --store us.duckdb --provider sharadar --tables sf1,sep,daily \
  --since $(date -d '-3 day' +%F)

# 상태 · 검증 · 백테스트 · 리포트 · 공식 성과
opt-factor status   --store us.duckdb
opt-factor validate --store us.duckdb --config strategy.json
opt-factor backtest --store us.duckdb --config strategy.json      # 참고용
opt-factor report   --store us.duckdb --config strategy.json --out tearsheet.html
opt-factor optimize --store us.duckdb --config strategy.json --space space.json  # 공식
```

### 적재 후 반드시 확인할 것

절단 버그가 전부 "성공" 보고와 함께 조용히 발생했다. `status` 만으로는 부족하다:

```sql
-- 종목당 거래일 수가 기대치(연 252일 × 년수)에 맞는가
SELECT COUNT(*)/COUNT(DISTINCT ticker) FROM prices;
-- 시총 결측이 0 인가 (DAILY 병합 확인)
SELECT COUNT(*) FROM prices WHERE mcap IS NULL;
-- 종목당 분기 수가 기대치(4 × 년수)에 맞는가
SELECT COUNT(*)/COUNT(DISTINCT ticker) FROM fundamentals;
```

**풀 히스토리 번들 기준 정상값** (2026-08-15 실측):

| 검사 | 값 |
|---|---|
| `prices` 종목 수 | 21,963 |
| `prices` 행 수 | 46,308,506 |
| 날짜 범위 | 1997-12-31 ~ 최근 거래일 |
| `fundamentals` 종목 수 | 17,040 (종목당 38.4분기) |
| `prices.mcap` 결측 — 전체 | 13.5% |
| `prices.mcap` 결측 — 보통주 | 6.2% |

> **`mcap` 결측 0 을 기대하면 안 된다.** DAILY 는 우선주·워런트·2종 주식에
> 시총을 주지 않는다(각 100% 결측). 연도별로도 고르지 않다 — 1998년 92%,
> 2013년 3.1%, 2022년 25%. DAILY 커버리지가 SEP 보다 늦게 시작하고, 시총이
> 없는 종류의 상장 비중이 시기마다 다르기 때문이다.
>
> 시총이 없으면 유니버스 밴드 필터에서 조용히 탈락한다. 그래서 이 숫자는
> "0 인가"가 아니라 **"평소와 같은가"** 로 본다. 갑자기 오르면 벤더 커버리지가
> 아니라 DAILY 병합 실패를 의심한다.

(파일럿용 S&P500 5년 스토어 기준값: 종목당 1,237거래일 / mcap 결측 0 / 19.8분기.)
