# NKTriF — News-Knowledge Tri-modal Fusion for Stock Movement Prediction

> A Multimodal Stable Fusion Approach for Fine-Grained Stock Movement Prediction: Empirical Evidence from the U.S. Stock Market
> *Submitted to COMBELT-2026 (The International Conference on Management, Business, Economics, Law and Technology), Danang, Vietnam.*

## Overview

Short-horizon stock movement prediction requires integrating heterogeneous market
signals — historical prices, macroeconomic conditions, and event-driven news — that
differ in structure, update frequency, and reliability. Two gaps limit prior work:

- **No conflict-aware fusion.** Conventional fusion treats all modalities as equally
  reliable, so when signals disagree (e.g. price declining while news sentiment is
  positive), the resulting joint representation faithfully reflects none of them.
- **Noisy raw financial text.** A single article often covers multiple companies and is
  padded with linguistic filler around the financially relevant claims, diluting
  stock-specific signal when encoded at the headline or document level.

**NKTriF (News-Knowledge Tri-modal Fusion)** addresses both gaps with:

1. **Structured news extraction** — Subject-Relation-Object (S-R-O) triples distilled
   from raw articles via LLM few-shot prompting with a schema-constrained taxonomy,
   isolating ticker-specific events from surrounding noise before embedding.
2. **Two-stage Gated Cross-Attention fusion** — a price-conditioned stabilizing gate at
   each fusion stage suppresses auxiliary signals that conflict with the price context,
   instead of blending all modalities uniformly.

Across 8 S&P 500 tickers, NKTriF outperforms all six price/news/macro baselines on both
accuracy and Matthews Correlation Coefficient (MCC), and translates into the strongest
risk-adjusted trading performance among all compared models (Sharpe = +0.85 net of
transaction costs).

## Methodology

**Encoder → Fusion → Prediction**, three sequential stages projecting each modality into
a shared *d*-dimensional latent space, fusing them under gated conflict regulation, then
decoding into a 3-class movement label (Down / Flat / Up):

- **Price modality encoder** — per-indicator linear projection (open/high/close
  volatility-normalized log-returns) → Bidirectional GRU → residual + LayerNorm.
- **News modality encoder** — precomputed daily news embedding sequence, reweighted by a
  **temporal-decay** coefficient (stronger weight closer to the forecast date) → two-stage
  linear projection with GELU + LayerNorm.
- **Macroeconomic modality encoder** — single linear projection of 5 z-scored indices
  (already low-dimensional, structured, and lagged by one trading day).
- **Fusion layer** — two sequential **Gated Cross-Attention (GCA)** stages. Stage 1: price
  (primary/query) attends over news (auxiliary/key-value) → price-news representation.
  Stage 2: price-news (primary) attends over macro (auxiliary) → final joint
  representation. News is fused before macro to reflect their functional asymmetry:
  short-horizon event shocks vs. slow-moving regime-level context. At each stage, a
  sigmoid gate derived **strictly from the stable primary anchor** selects which
  cross-attended auxiliary signal to keep, suppressing the rest.
- **Prediction layer** — a 3-layer MLP independently compresses the temporal dimension of
  (a) the final fused representation and (b) the original price representation kept as an
  undistorted parallel stream; the two are concatenated and linearly projected to 3-class
  logits, optimized end-to-end with Cross-Entropy loss via AdamW.

![Figure 1: NKTriF framework — Multimodal Encoding, Fusion, and Prediction layers](assets/fig1_framework.png)
*Fig. 1 — Framework of the proposed multimodal NKTriF model.*

**Data construction pipeline.** Stock price (Yahoo Finance) and macroeconomic indices
(FRED + Yahoo Finance) are retrieved, cleaned, and lagged by one trading day to prevent
lookahead bias. Financial news (Alpaca API + The Guardian) is temporally aligned so that
articles published after 16:00 ET roll over to the next trading session, deduplicated,
then passed through an LLM-based knowledge extraction pipeline: three-tier prompting
distills each article into S-R-O triples (8 entity types, 14 relation types), low-
confidence/low-relevance triples are filtered out, and the remaining triples are embedded
with FinBERT and aggregated into a confidence/impact/relevance-weighted daily average
embedding per ticker. All three modalities are then synchronized by trading date (and by
ticker for price/news) into the unified dataset.

![Figure 2: Data construction pipeline — price, macro, and news branches feeding the knowledge extraction pipeline into a unified dataset](assets/fig2_data_pipeline.png)
*Fig. 2 — The preprocessing and unified dataset construction workflow.*

> The S-R-O extraction → FinBERT embedding step runs in a separate preprocessing
> project; this repository consumes its output (per-date, per-ticker embedding vectors)
> as an input file — see [`main_test.py`](main_test.py).

## Experimental Setup

- **Universe** — 8 S&P 500 constituents across 4 sectors: **TSLA, AMZN** (Consumer
  Discretionary); **AAPL, MSFT, GOOGL, META** (Technology); **BA** (Industrials); **WMT**
  (Consumer Staples).
- **Period** — January 1, 2022 – June 23, 2025 (869 valid trading sessions).
- **Labeling** — next-session simple return vs. rolling 20-day percentile thresholds
  (Up / Down / Flat), yielding a near-balanced class distribution:

  | Ticker | Sector | DOWN | FLAT | UP | Total |
  |---|---|---|---|---|---|
  | TSLA | Consumer Discretionary | 283 | 282 | 284 | 849 |
  | AAPL | Technology | 282 | 284 | 283 | 849 |
  | AMZN | Consumer Discretionary | 282 | 285 | 282 | 849 |
  | MSFT | Technology | 284 | 281 | 284 | 849 |
  | GOOGL | Technology | 283 | 282 | 284 | 849 |
  | META | Technology | 282 | 284 | 283 | 849 |
  | BA | Industrials | 284 | 281 | 284 | 849 |
  | WMT | Consumer Staples | 283 | 283 | 283 | 849 |
  | **Total** | | **2,263** | **2,262** | **2,267** | **6,792** |

  *Table 2 — Label distribution across the experimental dataset.*

- **Split** — chronological 70% train / 15% validation / 15% test, no overlap; z-score
  normalization fit on the training split only.
- **Window** — 14-day look-back per sample (`T = 14`).
- **Baselines** (6, grouped by input modality):
  - Price only: **LSTM**, **ALSTM** (temporal attention over hidden states)
  - Price + news: **ALSTM-W** (window-averaged FinBERT embeddings), **SLOT** (per-timestep
    price/text cross-projection)
  - Price + macro: **ESTIMATE** (concat + multi-head attention), **DTML** (multi-level
    Transformer over temporal/cross-variable dependencies)
- **Hyperparameter search** — grid search over learning rate {5e-5, 1e-4, 5e-4}, hidden
  dimension {32, 64, 128}, dropout {0.1, 0.2}, selecting the combination maximizing
  validation MCC. NKTriF additionally uses 4 attention heads, temporal decay = 0.1.
- **Evaluation** — Accuracy (ACC) and Matthews Correlation Coefficient (MCC), computed
  over 5 independent random seeds (mean ± std reported).
- **Trading simulation** — daily confidence-weighted long/short rebalancing driven by the
  model's softmax output, capital split evenly between long and short legs, net of a 4
  bps per-side transaction cost; evaluated via Annualized Sharpe Ratio and Geometric
  Annualized Return.

## Main Results

**Overall comparison.** NKTriF achieves the best ACC and MCC among all baselines,
surpassing the strongest baseline (DTML) by +1.36 pp ACC and +0.0158 MCC (a 12.8%
relative MCC gain). Naively fusing news via conventional cross-modal projection instead
actively *hurts* performance (ALSTM-W, SLOT both underperform the unimodal LSTM),
underscoring that the gated fusion — not merely adding more modalities — drives the gain.

| Method | Sources | ACC | MCC |
|---|---|---|---|
| LSTM | Price | 0.4193 ± 0.0087 | 0.1220 ± 0.0134 |
| ALSTM | Price | 0.4139 ± 0.0079 | 0.1162 ± 0.0117 |
| ALSTM-W | Price + news | 0.3799 ± 0.0073 | 0.0642 ± 0.0115 |
| SLOT | Price + news | 0.3709 ± 0.0073 | 0.0503 ± 0.0080 |
| ESTIMATE | Price + macro | 0.3986 ± 0.0069 | 0.0996 ± 0.0105 |
| DTML | Price + macro | 0.4188 ± 0.0045 | 0.1236 ± 0.0077 |
| **NKTriF (proposed)** | Price + news + macro | **0.4324 ± 0.0086** | **0.1394 ± 0.0152** |

*Table 3 — Comparison of stock movement prediction performance across all models on the test set.*

**Modality contribution (ablation).** Removing news (`NKTriF-PM`) is the more damaging
ablation (MCC −24.8%), confirming news carries the primary event-driven signal.
Removing macro (`NKTriF-PN`) degrades mean MCC less (−17.6%) but more than doubles the
cross-seed variance — macro indices act as a structural regularizer that stabilizes
training across regimes and initializations, even though news drives absolute accuracy.

| Variant | ACC | MCC | ΔMCC |
|---|---|---|---|
| NKTriF | 0.4324 ± 0.0086 | 0.1394 ± 0.0152 | – |
| NKTriF-PM (no news) | 0.4107 ± 0.0077 | 0.1048 ± 0.0098 | −0.0346 (−24.8%) |
| NKTriF-PN (no macro) | 0.4170 ± 0.0227 | 0.1148 ± 0.0321 | −0.0246 (−17.6%) |

*Table 4 — Ablation results on modality contribution.*

**Gating mechanism.** Removing the stabilizing gate (`NKTriF-NoGate`, direct residual
addition instead) drops MCC by 29.8% and more than doubles the standard deviation across
seeds — the gate improves both mean accuracy and cross-seed stability simultaneously.

| Variant | ACC | MCC |
|---|---|---|
| NKTriF | 0.4324 ± 0.0086 | 0.1394 ± 0.0152 |
| NKTriF-NoGate | 0.4131 ± 0.0248 | 0.1074 ± 0.0346 |

*Table 5 — Gated vs. non-gated fusion.*

**Fusion order.** Fusing news before macro (proposed, `NKTriF-NM`) outperforms the
inverted order (`NKTriF-MN`, macro before news) by 0.0231 MCC with lower variance —
short-horizon event signals should be integrated before slow-moving contextual ones to
preserve their representational capacity.

| Variant | ACC | MCC | ΔMCC |
|---|---|---|---|
| NKTriF-NM (price→news→macro, proposed) | 0.4324 ± 0.0086 | 0.1394 ± 0.0152 | – |
| NKTriF-MN (price→macro→news) | 0.4199 ± 0.0150 | 0.1163 ± 0.0228 | −0.0231 |

*Table 6 — Fusion order variants.*

**Hyperparameter sensitivity.** Hidden dimension `d = 64`, learning rate `1e-4`, and
window size `20` maximize validation MCC; both smaller and larger settings degrade
performance, with insufficient window context (`ws = 10`) the most damaging failure mode.

| ![Fig. 3 — Hidden dimension](assets/fig3_hidden_dim_sweep.png) | ![Fig. 4 — Learning rate](assets/fig4_learning_rate_sweep.png) | ![Fig. 5 — Window size](assets/fig5_window_size_sweep.png) |
|:---:|:---:|:---:|
| *Fig. 3 — Hidden dimension* | *Fig. 4 — Learning rate* | *Fig. 5 — Window size* |

**Trading simulation.** NKTriF delivers the strongest risk-adjusted profile among all
models and is the only multimodal method with a positive Sharpe ratio — naive multimodal
baselines (ALSTM-W, DTML) underperform even the price-only LSTM, showing that unregulated
auxiliary fusion erodes trading returns through excess turnover on noisy signals.

| Model | Sharpe (net TC) | Ann. Return |
|---|---|---|
| **NKTriF (proposed)** | **+0.85** | **+27.68%** |
| LSTM | +0.78 | +26.98% |
| SLOT | −0.32 | +1.55% |
| ALSTM | −0.30 | −3.17% |
| ESTIMATE | −0.24 | −4.00% |
| ALSTM-W | −0.82 | −5.37% |
| DTML | −0.86 | −19.67% |
| Buy-and-Hold | – | −4.56% |

*Table 7 — Financial performance (net-of-cost trading simulation).*

## Limitations & Future Work

- Scope is currently limited to English-language news and US large-cap equities;
  extending to emerging markets (e.g. Vietnam) requires more robust entity resolution and
  domain-adapted language models where ticker-tagging infrastructure is less mature.
- The architecture operates at daily frequency; incorporating lower-frequency periodic
  disclosures (earnings reports, 10-K/10-Q filings) with an explicit post-publication
  temporal decay is a natural extension toward multi-horizon forecasting.

---

## Repository Structure

```
configs/            Global and training configuration (paths, tickers, hyperparameters)
data_pipeline/       Raw data fetching (Yahoo Finance, Alpaca) and processing
  fetchers/          Price + macro (Yahoo) and news (Alpaca) API clients
  processors/        Price/macro cleaning, news alignment
  builder.py         Synchronizes price + macro + precomputed news embeddings into
                      the unified dataset
encoders/            Per-modality encoders (Section 3.3)
  indicator_encoder.py  Price encoder (projection + BiGRU + residual/LayerNorm)
  news_encoder.py       News encoder (temporal decay + two-stage projection)
  macro_encoder.py      Macro encoder (single linear projection)
  multi_encoder.py      Wraps all three into the multimodal encoding layer
src/
  fusion.py          Gated Cross-Attention (GCA) fusion stage
  predictor.py        Prediction layer (3-layer MLP temporal compression + classifier)
  model.py             End-to-end NKTriF model (encoders → fusion → predictor) + loss
  data_loader.py       Windowing, labeling, chronological splitting, normalization
  modules/layers.py    Shared building blocks (ThreeLayerMLP)
main.py              Training entry point (multi-ticker, saves output/best_model.pt)
main_test.py          Builds the unified dataset from raw price/macro + a precomputed
                      news-embedding JSON (output of the external S-R-O/FinBERT pipeline)
modality_comparison.py  Modality-contribution ablation runner (Table 4)
performance_analyzer.py Per-ticker diagnostic analysis of a trained checkpoint
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate         # Windows
pip install -r requirements.txt

# 1) Set API credentials
set ALPACA_API_KEY=...
set ALPACA_SECRET_KEY=...

# 2) Build the unified dataset (requires a precomputed news-embedding JSON
#    from the external S-R-O extraction + FinBERT embedding pipeline)
set NEWS_EMBEDDINGS_PATH=path\to\news_embeddings.json
python main_test.py

# 3) Train
python main.py

# 4) Analyze a trained checkpoint
python performance_analyzer.py
```
