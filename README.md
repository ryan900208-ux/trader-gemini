# fixed8_control Fundamental Backtest

這是「全上市嚴格強勢續漲 fixed8_control」的第一版研究型回測骨架。

目標：

- 保留既有技術面策略。
- 加入基本面分數與基本面硬性濾網。
- 產出可檢查的交易紀錄、每日權益曲線與績效摘要。
- 保持模組化，下一步可擴充成機器學習自動優化框架。

> 注意：yfinance 不提供完整台股上市股票清單，也不保證有歷史 point-in-time 基本面資料。嚴謹回測請使用 `data/fundamentals.csv`，並以 `as_of_date` 填入當時已公告、可被投資人知道的資料。

## 快速開始

1. 建立全上市股票池：

```powershell
python scripts/fetch_twse_universe.py --output data/universe_twse_all.csv
```

這會從 TWSE 官方 OpenAPI 讀取「上市公司基本資料」，並轉成 yfinance 使用的 `.TW` ticker。

2. 放入基本面資料：

```text
data/fundamentals_sample.csv
```

格式：

```csv
symbol,as_of_date,roe,revenue_growth,eps,debt_to_equity,pe,pb,gross_margin,operating_margin
2330.TW,2022-05-15,0.28,0.18,32.1,0.36,18.5,4.2,0.52,0.42
```

3. 安裝套件：

```powershell
pip install -r requirements.txt
```

4. 執行回測：

```powershell
python -m fixed8_backtest.cli --config config/fixed8_control_fundamental.json
```

輸出會放在：

```text
outputs/fixed8_control_fundamental/
```

## 策略摘要

技術面條件：

- `score >= 90`
- `RS20` 前 8%
- `RS60` 前 12%
- `Close > MA20`
- `MA20 > MA60`
- `MA20` 斜率大於 0
- `RSI14` 介於 52 到 70
- 個股 20 日報酬大於 0050 20 日報酬
- 個股 60 日報酬大於 0050 60 日報酬
- 5 日報酬小於等於 8%
- 20 日報酬小於等於 35%
- `Volume_ratio` 介於 1.1 到 2.2
- 不過度偏離 MA20

基本面條件：

- 基本面分數達門檻。
- 預設排除 EPS <= 0、ROE 太低、營收成長太弱、負債比太高、PE/PB 極端值。
- 最終分數 = 技術分數與基本面分數加權。

交易規則：

- t 日收盤判斷，t+1 日開盤買進。
- 最多 5 檔，等權配置。
- 固定停損 8%。
- 持有 40 天到期出場。
- 大盤 bear 出場。
- 跌破 MA120 出場。
- RS20 連續轉弱出場。
- 停損後冷卻 30 天。

## 下一步 ML 擴充方向

第一版先用規則式與加權分數建立 baseline。之後可把每日候選股票轉成訓練資料：

- 特徵：技術指標、相對強弱、基本面分數、財務比率、波動率、成交量結構。
- 標籤：未來 20 或 40 日風險調整報酬、是否觸發停損、是否跑贏 0050。
- 模型：LightGBM / XGBoost 排序模型。
- 驗證：walk-forward 回測與 out-of-sample 測試。
