# Adaptive Institutional Volatility Forecasting Engine (Neural-Gated Ensemble)

An institutional-grade quantitative volatility modeling framework implementing in-sample parametric fitting (2015–2021), causal zero-lookahead out-of-sample (OOS) recursive forecasting (2022–2025), and comprehensive econometric evaluation across Indian equity indices and large-cap equities.

---

## 📌 Project Overview

- **In-Sample Static Estimation Period:** 2015-01-01 to 2021-12-31
- **Out-of-Sample (OOS) Evaluation Period:** 2022-01-01 to 2025-12-31 (~987 trading days per asset)
- **Asset Universe (8 Assets):**
  - **Indices:** NIFTY 50 (`nifty50`), BANK NIFTY (`banknifty`)
  - **Equities:** Adani Enterprises (`adanient`), Tata Steel (`tatasteel`), DLF (`dlf`), Hindustan Unilever (`hindunilvr`), Nestle India (`nestleind`), Sun Pharma (`sunpharma`)
- **Evaluated Volatility Models:**
  1. **EWMA** ($\lambda = 0.94$, RiskMetrics specification)
  2. **GARCH(1,1)** (Bollerslev, 1986)
  3. **EGARCH(1,1)** (Nelson, 1991 — asymmetric log-variance)
  4. **GJR-GARCH(1,1)** (Glosten, Jagannathan & Runkle, 1993 — threshold leverage)

---

## 🛠 Project Architecture

```
├── data/
│   ├── {asset}.csv                          # Raw daily OHLC & return datasets (2015–2025)
│   ├── static_model_outputs/
│   │   ├── {asset}_static_models.csv        # In-sample fitted conditional variance series
│   │   └── model_parameters.csv             # Frozen MLE parameter estimates (2015–2021)
│   └── oos_forecasts/
│       ├── {asset}_oos_forecasts.csv        # Causal OOS one-step-ahead daily variance forecasts
│       ├── oos_model_comparison.csv         # Econometric evaluation metrics (MAE, RMSE, QLIKE)
│       ├── oos_model_ranking.csv            # Asset-by-asset and aggregate model rankings
│       └── plots/
│           └── {asset}_oos_forecasts.png    # High-resolution OOS forecast trajectory plots
├── static_volatility_models.py              # In-sample static fitting pipeline
├── oos_forecasting_evaluation.py            # Causal OOS recursive forecasting & audit suite
├── final_data_checks.py                     # Data integrity & validation tests
└── README.md
```

---

## 🔬 Forecasting Methodology & Causality Guarantee

All OOS forecasts are generated via **explicit step-by-step causal recursion** with **frozen parameters**:
1. At date $t$, the variance forecast $\hat{\sigma}_t^2$ is computed **strictly using information through $t-1$** ($\hat{\sigma}_{t-1}^2$ and $r_{t-1}$).
2. The forecast is recorded before observing $r_t$.
3. The realized return $r_t$ is observed only after the forecast is stored, updating the state for step $t+1$.
4. **Initialization:** The first OOS forecast at $t_0$ is warm-started exclusively from the terminal in-sample variance state on 2021-12-31.
5. **No Look-Ahead Bias:** Verified via automated perturbation leakage testing ($t_0$ forecast invariant to $r[t_0]$ perturbation; $t_1$ state update causally responsive).

---

## 📊 OOS Evaluation Metrics & Loss Functions

Models are benchmarked against two volatility proxies:
- **Primary Target:** Daily Realized Variance Proxy ($r_t^2$)
- **Secondary Target:** Parkinson High-Low Range Variance Estimator ($\frac{1}{4 \ln 2} [\ln(H_t/L_t)]^2$)

Evaluation Loss Functions:
- **MAE:** $\frac{1}{N} \sum |y_t - \hat{\sigma}_t^2|$
- **RMSE:** $\sqrt{\frac{1}{N} \sum (y_t - \hat{\sigma}_t^2)^2}$
- **QLIKE (Robust Quasi-Likelihood):** $\frac{1}{N} \sum \left( \frac{\hat{\sigma}_t^2}{y_t} - \ln\frac{\hat{\sigma}_t^2}{y_t} - 1 \right)$

---

## 🏆 Summary of Overall OOS Rankings (Primary Target: QLIKE)

| Model | Avg QLIKE Rank | Avg MAE Rank | Avg RMSE Rank |
|:---|:---:|:---:|:---:|
| **EWMA ($\lambda=0.94$)** | **1.500** | **1.000** | 2.750 |
| **EGARCH(1,1)** | 2.625 | 2.125 | **1.750** |
| **GJR-GARCH(1,1)** | 2.875 | 3.625 | 3.125 |
| **GARCH(1,1)** | 3.000 | 3.250 | 2.375 |

---

## 🚀 Execution Instructions

```bash
# 1. Run static parameter estimation (2015–2021)
python3 static_volatility_models.py

# 2. Run causal OOS forecasting, leakage testing & evaluation (2022–2025)
python3 oos_forecasting_evaluation.py
```
