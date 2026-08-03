# 데이터 계약 — 스토어 스키마 · PIT 규약 · 운영 절차

## 1. 스토어 스키마 (DuckDB, 벤더 중립)

| 테이블 | 키 | 내용 | 공시일 |
|---|---|---|---|
| `fundamentals` | (ticker, calendardate, dimension) | SF1 분기 재무 | `datekey` (실제 공시일) |
| `ownership` | (ticker, calendardate) | 13F 집계 + 내부자 분기 집계 | 13F: 분기말+45일 / 내부자: 분기말+3일 |
| `estimates` | (ticker, calendardate) | 애널리스트 추정치 | 수집 시점 스냅샷 |
| `prices` | (ticker, date) | 일별 가격·거래량·시총·공매도 | 해당일 |
| `tickers` | (ticker) | 섹터·소재지·유형·이름 | — |

컬럼명은 전부 `schema.FIELDS` 의 표준 이름이다. 벤더 이름은 어댑터의
`normalize_columns()` 에서 끝난다 — 벤더 교체 시 어댑터만 바꾼다.

## 2. PIT 불변식 (코드가 강제)

1. **`datekey >= calendardate`** — 위반 행은 `validate_pit_frame()` 이 스토어 진입 전에 거부.
2. **최초 공시 우선** — 같은 (ticker, 분기) 재공시(정정)는 무시된다.
   시장이 처음 본 숫자가 백테스트가 봐야 하는 숫자다.
3. **소스별 공시일 분리** — SF1(실적공시)·SF3(+45일)·SF2(+3일)의 datekey 가
   각각 관리되고, 혼합 소스 표현식은 BinOp 에서 **늦은 쪽 공시일**을 따른다
   (`Panel.avail` element-wise max 전파). `inst_shares / sharesbas` 는
   13F 공시일 이전에 절대 값을 갖지 않는다.
4. **집계의 확정 시점** — 분기 중 도착하는 데이터(Form 4)의 분기 합계는
   분기가 끝나야 확정이다. datekey = 분기말 + 마감유예.
5. **추정치는 소급 금지** — FMP 추정치는 수집 시작일부터만 PIT 성립.
   과거 백필은 구조적 look-ahead 라 지원하지 않는다.

## 3. 운영 절차 (Sharadar 구독 후)

```bash
export NASDAQ_DATA_LINK_API_KEY=...

# 초기 적재 (벌크 CSV — 구독 페이지에서 다운로드)
opt-factor ingest --store us.duckdb --provider csv --csv SHARADAR_SF1.csv --kind fundamentals
opt-factor ingest --store us.duckdb --provider csv --csv SHARADAR_SEP.csv --kind prices
opt-factor ingest --store us.duckdb --provider csv --csv SHARADAR_TICKERS.csv --kind tickers

# 일간 증분 (cron)
opt-factor ingest --store us.duckdb --provider sharadar --tables sf1,sep,daily --since $(date -d '-3 day' +%F)

# 상태 확인
opt-factor status --store us.duckdb

# 팩터 검증 → 백테스트(참고용) → walk-forward PO(공식)
opt-factor validate --store us.duckdb --config strategy.json
opt-factor backtest --store us.duckdb --config strategy.json
opt-factor optimize --store us.duckdb --config strategy.json --space space.json
```

## 4. 구독 전 확인 사항

- [ ] sharadar.com 직판 Bundle 에 SF1/SEP/DAILY/SF2/SF3A/TICKERS 포함 여부
- [ ] 직판 API 형태가 Nasdaq Data Link datatables 와 같은지
      (다르면 `sharadar.py` 의 `_TABLE_URL`·`_paginate` 만 수정)
- [ ] SF3A 컬럼명 (`shrunits`/`shrholders` 가정 — `_SF3A_RENAME` 에서 조정)
- [ ] PTP 필터: 실전 전 브로커의 IRS 1446(f) 리스트로 `extra_ptp_tickers` 보강
