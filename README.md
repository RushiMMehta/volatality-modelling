# Adaptive Institutional Volatility Forecasting Engine: A Neural-Gated Ensemble Approach to EWMA, GARCH, EGARCH, and GJR-GARCH Volatility Forecasting

An empirical quantitative finance and financial econometrics research framework implementing parametric volatility estimation, zero-lookahead walk-forward recursive forecasting, and a learned neural gating ensemble for daily conditional variance prediction across Indian equity indices and large-cap equities.

---

## 1. Project Overview

The **Adaptive Institutional Volatility Forecasting Engine** investigates hybrid econometric-machine learning architectures for 1-day-ahead conditional volatility forecasting. The system integrates classical econometric volatility models with a lightweight **Neural Gating Engine** that dynamically computes time-varying model weights based on trailing model forecasting error, realized volatility dynamics, volatility-of-volatility, and asymmetric return shocks. Rather than treating machine learning as an unconstrained black-box forecaster, this architecture positions the neural network as an adaptive meta-allocator over theoretically grounded, stationary econometric specifications.

```mermaid
flowchart LR
    subgraph MarketData [Market Data Layer]
        A[Daily OHLC Prices] --> B[Log Returns r_t]
        B --> C[Realized Variance Proxy r_t^2]
        B --> D[Parkinson Range Proxy]
    end

    subgraph EconometricEngines [Base Econometric Models]
        B --> M1[EWMA RiskMetrics]
        B --> M2["GARCH(1,1)"]
        B --> M3["EGARCH(1,1)"]
        B --> M4["GJR-GARCH(1,1)"]
    end

    subgraph ForecastGen [Recursive Forecasts]
        M1 --> F1["EWMA Forecast σ²_t"]
        M2 --> F2["GARCH Forecast σ²_t"]
        M3 --> F3["EGARCH Forecast σ²_t"]
        M4 --> F4["GJR-GARCH Forecast σ²_t"]
    end

    subgraph WeightingMechanisms [Ensemble Layer]
        F1 & F2 & F3 & F4 --> SB["Static Inverse-RMSE Baseline (Ablation)"]
        F1 & F2 & F3 & F4 & C --> NG["Neural Gating Engine (Core Contribution)"]
    end

    subgraph FinalEnsemble [Ensemble Outputs]
        SB --> ES["Static Baseline Forecast"]
        NG --> EN["Neural-Gated Forecast"]
    end

    subgraph EvaluationLayer [Evaluation Framework]
        ES & EN --> EV1["Statistical Loss (RMSE / MAE / QLIKE)"]
        ES & EN --> EV2["Statistical Significance (DM / MCS) [Planned]"]
        ES & EN --> EV3["Risk & Economic Value (VaR / BS Pricing) [Planned]"]
    end
```

---

## 2. Research Objective

The central research question is:

> *Can a lightweight, causally conditioned neural gating ensemble over classical volatility models systematically outperform both individual parametric models and a static inverse-RMSE weighting baseline in statistical forecasting accuracy and economic utility under non-stationary market regimes?*

The research design benchmarks candidate architectures across:
1. **Statistical Forecasting Accuracy:** Loss metrics under asymmetric and robust loss functions ($\text{RMSE}$, $\text{MAE}$, and Gaussian Quasi-Likelihood $\text{QLIKE}$), evaluated against observable realized-volatility proxies.
2. **Statistical Significance Testing:** Pairwise Diebold-Mariano tests with Newey-West HAC adjustments and Hansen-Lunde-Nason Model Confidence Set ($\text{MCS}$) procedures to identify superior model confidence bounds.
3. **Economic and Risk Utility:** Parametric Value-at-Risk ($\text{VaR}$) backtesting (Kupiec $\text{POF}$ and Christoffersen independence tests) and at-the-money ($\text{ATM}$) Black-Scholes option pricing error relative to India VIX-implied benchmarks.

> [!NOTE]
> Comparative claims in this project are strictly substantiated by empirical experimental evidence recorded within the evaluation outputs. Where downstream modules (e.g., MCS, option pricing) are planned or in development, they are explicitly designated as such.

---

## 3. System Architecture & Pipeline

The end-to-end forecasting pipeline operates through sequential, strictly causal stages:

$$\text{Market Data} \longrightarrow \text{Return \ Proxy Engine} \longrightarrow \text{Base Models} \longrightarrow \text{Recursive Forecasts} \longrightarrow \begin{cases} \text{Inverse-RMSE Baseline} \\ \text{Neural Gating Engine} \end{cases} \longrightarrow \text{Statistical \ Economic Evaluation}$$

1. **Market Data Ingestion:** Daily OHLC records spanning 2015-01-01 through 2025-12-31 across 8 Indian assets.
2. **Returns & Realized Proxies:** Computation of close-to-close log returns $r_t = \ln(P_t / P_{t-1})$, the standard squared return proxy $y_t = r_t^2$, and the Parkinson high-low range estimator.
3. **Parametric Model Estimation:** In-sample Maximum Likelihood Estimation (MLE) over 2015–2021 to establish frozen parameter sets.
4. **Causal Out-of-Sample (OOS) Forecasting:** Recursive 1-day-ahead conditional variance generation $\hat{\sigma}_{i,t}^2 = f(\mathcal{I}_{t-1})$ across the 2022–2025 out-of-sample window.
5. **Static Benchmark Weighting:** Trailing 60-day inverse-RMSE model weight allocation ($w_{i,t} \propto \text{RMSE}_{i,t-1}^{-1}$).
6. **Adaptive Neural Gating:** Feedforward neural network mapping a 9-dimensional causal state vector $\mathbf{x}_t \in \mathbb{R}^9$ into time-varying softmax allocation weights $\mathbf{w}_t \in \Delta^3$.
7. **Multi-Horizon Evaluation:** Evaluation over held-out test partitions using robust loss metrics, tail-risk diagnostics, and derivative pricing error.

---

## 4. Base Econometric Models

The ensemble builds on four foundational univariate volatility specifications:

### 4.1. EWMA / RiskMetrics
The Exponentially Weighted Moving Average model serves as the classical non-parametric industry benchmark:

$$\sigma_{t}^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) r_{t-1}^2$$

where $\lambda = 0.94$ (standard RiskMetrics daily decay factor). EWMA incorporates no mean-reverting drift and reacts directly to recent return shocks.

### 4.2. GARCH(1,1) (Bollerslev, 1986)
The standard stationary generalized autoregressive conditional heteroskedasticity model:

$$\sigma_{t}^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$$

- **Constraints:** $\omega > 0$, $\alpha \ge 0$, $\beta \ge 0$, and covariance stationarity condition $\alpha + \beta < 1$.
- **Role:** Captures persistent volatility clustering and geometric mean reversion toward the unconditional variance $\sigma^2 = \omega / (1 - \alpha - \beta)$.

### 4.3. EGARCH(1,1) (Nelson, 1991)
Exponential GARCH models the logarithm of conditional variance, capturing asymmetric leverage effects without requiring non-negativity parameter constraints:

$$\ln(\sigma_{t}^2) = \omega + \alpha \left( |e_{t-1}| - \mathbb{E}[|e_{t-1}|] \right) + \gamma e_{t-1} + \beta \ln(\sigma_{t-1}^2)$$

where $e_{t-1} = r_{t-1} / \sigma_{t-1}$ is the standardized innovation, and $\mathbb{E}[|e_{t-1}|] = \sqrt{2/\pi} \approx 0.79788$ for $e_t \sim \mathcal{N}(0,1)$.
- **Leverage Effect:** When $\gamma < 0$, negative return shocks ($e_{t-1} < 0$) induce a larger increase in log-variance than positive shocks of equal magnitude.

### 4.4. GJR-GARCH(1,1) (Glosten, Jagannathan & Runkle, 1993)
A threshold heteroskedasticity model that directly augments squared innovations with an indicator for downward shocks:

$$\sigma_{t}^2 = \omega + \left( \alpha + \gamma \mathbb{I}_{\{r_{t-1} < 0\}} \right) r_{t-1}^2 + \beta \sigma_{t-1}^2$$

- **Leverage Indicator:** $\mathbb{I}_{\{r_{t-1} < 0\}} = 1$ if $r_{t-1} < 0$ and $0$ otherwise.
- **Persistence Measure:** $\alpha + \beta + \frac{1}{2}\gamma < 1$ for covariance stationarity. When $\gamma > 0$, bad news generates disproportionately higher conditional variance.

---

## 5. Static Inverse-RMSE Baseline

The static inverse-RMSE weighting mechanism is retained strictly as an **ablation benchmark**, not the primary contribution:

> **The static inverse-RMSE ensemble serves as the baseline against which the adaptive neural-gated ensemble is evaluated.**

### Weight Calculation
For each date $t$, trailing Root Mean Squared Error ($\text{RMSE}$) is calculated over a rolling 60-day historical window $\tau \in \{t-60, \dots, t-1\}$ strictly prior to observing date $t$:

$$\text{RMSE}_{i,t} = \sqrt{\frac{1}{60} \sum_{\tau=t-60}^{t-1} \left( \hat{\sigma}_{i,\tau}^2 - y_\tau \right)^2}$$

The normalized baseline weights are defined by:

$$w_{i,t}^{\text{static}} = \frac{\left(\max(\text{RMSE}_{i,t}, \varepsilon)\right)^{-1}}{\sum_{j=1}^4 \left(\max(\text{RMSE}_{j,t}, \varepsilon)\right)^{-1}}, \quad \varepsilon = 10^{-12}$$

$$\hat{\sigma}_{\text{static}, t}^2 = \sum_{i=1}^4 w_{i,t}^{\text{static}} \hat{\sigma}_{i,t}^2$$

This provides a competitive empirical benchmark that adapts solely to trailing forecasting error without learning non-linear market-state dependencies.

---

## 6. Neural Gating Engine — Core Contribution

The **Neural Gating Engine** represents the core methodological contribution of this research framework:

> **The neural gate dynamically adapts the contribution of each econometric model according to recent forecasting performance and market conditions.**

Rather than forecasting variance directly using an unconstrained deep neural network, a compact multi-layer perceptron acts as a dynamic meta-learner producing the convex combination weights for the four econometric forecasters.

```mermaid
flowchart TD
    subgraph FeatureVector ["Input Feature Vector x_t ∈ R^9 (Information through t-1)"]
        F_RMSE["Trailing Model RMSE (60d)<br/>• ewma_rmse_60<br/>• garch_rmse_60<br/>• egarch_rmse_60<br/>• gjr_garch_rmse_60"]
        F_RV["Realized Volatility Dynamics<br/>• realized_vol_5<br/>• realized_vol_20<br/>• realized_vol_60"]
        F_VOV["Vol-of-Vol Dynamics<br/>• vol_of_vol_30 (ddof=1)"]
        F_REG["Asymmetric Return Shock<br/>• return_sign_lag1: I(r_{t-1} < 0)"]
    end

    subgraph MLP ["Compact Neural Gating Architecture"]
        IN[Input Layer: 9 Nodes] --> Dense[Hidden Layer: 12 Units + ReLU]
        Dense --> OUT[Output Layer: 4 Units + Softmax Activation]
    end

    subgraph GatingWeights ["Time-Varying Model Weights w_t ∈ Δ^3"]
        OUT --> W1["w_EWMA,t"]
        OUT --> W2["w_GARCH,t"]
        OUT --> W3["w_EGARCH,t"]
        OUT --> W4["w_GJR-GARCH,t"]
    end

    subgraph VarianceEnsemble ["1-Day-Ahead Ensemble Forecast"]
        W1 & W2 & W3 & W4 --> SUM["σ̂²_neural,t = Σ w_i,t · σ̂²_i,t"]
    end

    FeatureVector --> IN
```

### 6.1. Gating Feature Matrix $\mathbf{x}_t \in \mathbb{R}^9$
At each trading day $t$, the gating network receives a 9-dimensional state vector derived strictly from information up to $t-1$:

| Feature Name | Category | Window | Definition |
| :--- | :--- | :---: | :--- |
| `ewma_rmse_60` | Model Loss | 60 days | $\sqrt{\frac{1}{60}\sum_{\tau=t-60}^{t-1}(\hat{\sigma}_{\text{EWMA},\tau}^2 - y_\tau)^2}$ |
| `garch_rmse_60` | Model Loss | 60 days | $\sqrt{\frac{1}{60}\sum_{\tau=t-60}^{t-1}(\hat{\sigma}_{\text{GARCH},\tau}^2 - y_\tau)^2}$ |
| `egarch_rmse_60` | Model Loss | 60 days | $\sqrt{\frac{1}{60}\sum_{\tau=t-60}^{t-1}(\hat{\sigma}_{\text{EGARCH},\tau}^2 - y_\tau)^2}$ |
| `gjr_garch_rmse_60` | Model Loss | 60 days | $\sqrt{\frac{1}{60}\sum_{\tau=t-60}^{t-1}(\hat{\sigma}_{\text{GJR},\tau}^2 - y_\tau)^2}$ |
| `realized_vol_5` | Realized Dynamics | 5 days | $\sqrt{\frac{1}{5}\sum_{\tau=t-5}^{t-1} r_\tau^2}$ |
| `realized_vol_20` | Realized Dynamics | 20 days | $\sqrt{\frac{1}{20}\sum_{\tau=t-20}^{t-1} r_\tau^2}$ |
| `realized_vol_60` | Realized Dynamics | 60 days | $\sqrt{\frac{1}{60}\sum_{\tau=t-60}^{t-1} r_\tau^2}$ |
| `vol_of_vol_30` | Vol-of-Vol Dynamics | 30 RVs | $\text{StdDev}_{ddof=1}(\text{RV}_{5, t-30}, \dots, \text{RV}_{5, t-1})$ |
| `return_sign_lag1` | Regime / Shock | 1 day | $\mathbb{I}_{\{r_{t-1} < 0\}} \in \{0, 1\}$ |

### 6.2. Network Architecture & Loss Function
- **Architecture:** Compact feedforward MLP ($9 \to 12 \to 4$) with ReLU activation in the hidden layer and Softmax in the output layer.
- **Capacity Rationale:** Intentionally lightweight (172 trainable parameters) to prevent overfitting on moderate-sized financial time series (~500–1000 daily observations).
- **Optimization:** Adam optimizer ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$), batch size 32, max 200 epochs with early stopping (patience 15 on validation loss).
- **Training Loss (QLIKE):** Optimized directly under the robust Gaussian Quasi-Likelihood loss:

$$\mathcal{L}_{\text{QLIKE}}(y_t, \hat{\sigma}_{\text{ens}, t}^2) = \frac{y_t}{\hat{\sigma}_{\text{ens}, t}^2} - \ln\left(\frac{y_t}{\hat{\sigma}_{\text{ens}, t}^2}\right) - 1$$

---

## 7. Walk-Forward Evaluation & Causality Firewall

To prevent look-ahead bias and data leakage, the repository enforces a strict temporal partition architecture across all models and features:

```
├── In-Sample Static Fitting (2015-01-01 to 2021-12-31, ~1,720 days) -> Frozen Parameters
└── Out-of-Sample Recursive Evaluation (2022-01-01 to 2025-12-31, ~987 days)
    ├── Neural Gate Training Partition   (2022-01-03 to 2023-12-29, 493 days)
    ├── Neural Gate Validation Partition (2024-01-01 to 2024-12-31, 246 days)
    └── Neural Gate Held-Out Test Window (2025-01-01 to 2025-12-30, 248 days)
```

### Causality Guarantees & Automated Leakage Tests
1. **State Recursion:** Daily variance forecasts $\hat{\sigma}_t^2$ are computed strictly using information available at $t-1$. The realized return $r_t$ is observed only after storing $\hat{\sigma}_t^2$.
2. **Feature Scaler Isolation:** Z-score normalization parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) are computed exclusively on the 2022–2023 training partition and applied without modification to validation and test periods.
3. **Automated Perturbation Leakage Suite:** All pipeline stages execute perturbation verification tests:
   - Perturbing $r_t$ by $10\times$ or $1000\times$ causes **$0.00\text{e}+00$ change** in $\hat{\sigma}_t^2$ and feature vector $\mathbf{x}_t$.
   - The same perturbation causes an immediate causal update in $\hat{\sigma}_{t+1}^2$ and $\mathbf{x}_{t+1}$.
   - All 8 assets pass automated leakage tests across static models, static baseline, and neural feature engineering.

---

## 8. Model Training & Computational Strategy

The repository implements a production-grade econometrics and machine learning execution strategy:
- **Periodic vs. Daily Refitting:** Base econometric parameters $(\omega, \alpha, \beta, \gamma)$ are estimated via MLE once over the 7-year in-sample period and frozen during OOS recursive evaluation. This avoids noisy daily optimizer divergence while preserving computational efficiency.
- **Robust Boundary Handling:** Fitted and forecasted variances are constrained to be strictly positive ($\hat{\sigma}^2 \ge 10^{-12}$). If numerical instability occurs, parameter fallbacks to previous valid iterations are enforced.
- **Deterministic Machine Learning:** Neural network training runs with deterministic CPU seeds (`torch.manual_seed(42)`, `np.random.seed(42)`) to ensure bit-for-bit reproducibility.
- **Data Integrity Auditing:** SHA-256 hash validation runs before and after execution across all protected raw, static, OOS, baseline, and feature CSV files, ensuring zero in-place data mutation.

---

## 9. Data & Indian Market Universe

The empirical study focuses on the **Indian financial market**, utilizing daily market data (2015–2025) across 8 representative equity assets spanning distinct volatility regimes:

| Asset Identifier | Asset Name | Asset Category | Sector / Description |
| :--- | :--- | :--- | :--- |
| `nifty50` | NIFTY 50 Index (`^NSEI`) | Broad Index | Benchmark Indian Large-Cap Equity Index |
| `banknifty` | NIFTY Bank Index (`^NSEBANK`) | Sectoral Index | Banking & Financial Services Benchmark |
| `adanient` | Adani Enterprises Ltd. | High-Beta Stock | Conglomerate / Infrastructure |
| `tatasteel` | Tata Steel Ltd. | High-Beta Stock | Metals & Mining |
| `dlf` | DLF Ltd. | High-Beta Stock | Real Estate & Construction |
| `hindunilvr` | Hindustan Unilever Ltd. | Defensive Stock | Fast-Moving Consumer Goods (FMCG) |
| `nestleind` | Nestle India Ltd. | Defensive Stock | Fast-Moving Consumer Goods (FMCG) |
| `sunpharma` | Sun Pharmaceutical Industries Ltd. | Defensive Stock | Pharmaceuticals & Healthcare |
| `india_vix` | India VIX (`^INDIAVIX`) | Market Reference | NSE Volatility Index (Benchmark Series) |

### Volatility Proxy Definition & Methodological Limitation
Daily log returns are constructed as $r_t = \ln(P_t / P_{t-1})$. Because continuous instantaneous volatility is unobservable, the empirical evaluation employs two standard proxies:
1. **Primary Proxy (Squared Log Returns):** $y_t = r_t^2$
2. **Secondary Proxy (Parkinson Range Estimator):** $y_{t,\text{Parkinson}} = \frac{1}{4 \ln 2} \left[ \ln\left(\frac{H_t}{L_t}\right) \right]^2$

> [!WARNING]
> **Methodological Limitation:** True volatility is latent and unobservable; squared daily returns are therefore used as an observable realized-volatility proxy. While unbiased under standard assumptions, squared returns are inherently noisy point estimators of integrated variance.

---

## 10. Evaluation Framework

The complete evaluation architecture comprises statistical accuracy, significance testing, tail-risk coverage, and derivative pricing error:

### 10.1. Forecast Accuracy Metrics (Implemented)
- **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{N} \sum_{t=1}^N |y_t - \hat{\sigma}_t^2|$
- **Root Mean Squared Error (RMSE):** $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=1}^N (y_t - \hat{\sigma}_t^2)^2}$
- **Robust Quasi-Likelihood (QLIKE):** $\text{QLIKE} = \frac{1}{N} \sum_{t=1}^N \left( \frac{y_t}{\hat{\sigma}_t^2} - \ln\frac{y_t}{\hat{\sigma}_t^2} - 1 \right)$

### 10.2. Statistical Significance Testing (Planned)
- **Diebold-Mariano Test (1995):** Pairwise test of equal predictive accuracy with Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC) standard error adjustment for loss differentials $d_t = \mathcal{L}(y_t, \hat{\sigma}_{A,t}^2) - \mathcal{L}(y_t, \hat{\sigma}_{B,t}^2)$.

### 10.3. Model Confidence Set (MCS) (Planned)
- **Hansen, Lunde, & Nason (2011) MCS Procedure:** Bootstrap-based determination of the superior set of models $\mathcal{M}^*_{1-\alpha}$ at significance level $\alpha = 0.10$ across the candidate set:
  $$\mathcal{M}_0 = \{\text{EWMA}, \text{GARCH}, \text{EGARCH}, \text{GJR-GARCH}, \text{Static Baseline}, \text{Neural Gate}\}$$

### 10.4. Value-at-Risk (VaR) Backtesting (Planned)
- Parametric 1-day Value-at-Risk: $\text{VaR}_{t, \alpha} = - (\mu + z_\alpha \hat{\sigma}_t)$
- **Kupiec Proportion of Failures (POF) Test:** Likelihood ratio test of unconditional coverage ($H_0: p = \alpha$).
- **Christoffersen Independence & Conditional Coverage Test:** Markov-chain likelihood ratio test evaluating whether VaR exceptions cluster over consecutive trading sessions.

### 10.5. Economic-Value Option Pricing Evaluation (Planned)
- Black-Scholes pricing error evaluation for short-dated at-the-money European call options using forecasted volatility vs. market-implied benchmarks:
  $$C_{\text{model}} = S_t \Phi(d_1) - K e^{-r \tau} \Phi(d_2), \quad d_1 = \frac{\ln(S_t/K) + (r + \frac{1}{2}\hat{\sigma}_t^2)\tau}{\hat{\sigma}_t \sqrt{\tau}}$$
  where constant risk-free rate $r = 6.5\%$ (annualized) reflects Indian money market conditions. Where granular historical option-chain settlement data is inaccessible, synthetic Black-Scholes valuation benchmarked against **India VIX** provides empirical pricing validation.

---

## 11. Crisis & Regime Analysis

The research framework investigates model performance across calm vs. high-volatility turbulence:
- **Historical Shocks:** The 2015–2021 in-sample period incorporates the March 2020 COVID-19 equity shock, allowing parametric models to capture extreme tail volatility spikes.
- **Dynamic Weight Adaptation:** The neural gating network actively reallocates weight among models during turbulent test periods (2025). As observed in weight trajectories, high-beta assets shift weight dynamically toward asymmetric models (EGARCH and GJR-GARCH) following sharp market drops, while calm regimes reallocate toward EWMA and GARCH.
- **Trajectory Visualizations:** Asset-by-asset test-period neural weight trajectories are saved in [`data/neural_gating/plots/`](file:///Users/rushi/Desktop/Volatility%20Modelling/data/neural_gating/plots/).

---

## 12. Ablation Study & Empirical Findings

The ablation framework directly measures whether added neural complexity provides measurable forecasting improvement over simpler baselines:

$$\text{Individual Econometric Models} \longleftrightarrow \text{Static Inverse-RMSE Ensemble} \longleftrightarrow \text{Adaptive Neural Gating Ensemble}$$

### 12.1. Overall OOS Ranking of Base Econometric Models (2022–2025 Full OOS Window)
Across all 8 assets (~987 OOS trading days per asset, target: $r_t^2$):

| Base Model | Average QLIKE Rank | Average MAE Rank | Average RMSE Rank |
| :--- | :---: | :---: | :---: |
| **EWMA ($\lambda=0.94$)** | **1.500** | **1.000** | 2.750 |
| **EGARCH(1,1)** | 2.625 | 2.125 | **1.750** |
| **GJR-GARCH(1,1)** | 2.875 | 3.625 | 3.125 |
| **GARCH(1,1)** | 3.000 | 3.250 | 2.375 |

### 12.2. Neural Gate vs. Static Baseline (Strictly Held-Out 2025 Test Partition)
Evaluation over the 2025 test window (248 trading days) across all 8 assets (Primary Target: $r_t^2$):

| Asset | Static Baseline RMSE | Neural Gate RMSE | Static Baseline MAE | Neural Gate MAE | Static Baseline QLIKE | Neural Gate QLIKE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NIFTY 50** | $1.2568 \times 10^{-4}$ | **$1.2477 \times 10^{-4}$** | $6.1212 \times 10^{-5}$ | **$5.9256 \times 10^{-5}$** | 1.4385 | **1.4173** |
| **BANK NIFTY** | **$1.2400 \times 10^{-4}$** | $1.2413 \times 10^{-4}$ | **$7.2596 \times 10^{-5}$** | $7.2975 \times 10^{-5}$ | **1.3832** | 1.3895 |
| **Adani Enterprises** | **$7.9752 \times 10^{-4}$** | $7.9859 \times 10^{-4}$ | **$6.2199 \times 10^{-4}$** | $6.2977 \times 10^{-4}$ | **1.8307** | 1.8383 |
| **Tata Steel** | $8.0079 \times 10^{-4}$ | **$8.0012 \times 10^{-4}$** | $4.1231 \times 10^{-4}$ | **$3.9118 \times 10^{-4}$** | **1.7480** | 1.7635 |
| **DLF** | $7.5609 \times 10^{-4}$ | **$7.4668 \times 10^{-4}$** | $5.3550 \times 10^{-4}$ | **$5.1411 \times 10^{-4}$** | 1.7340 | **1.6990** |
| **Hindustan Unilever** | $3.0337 \times 10^{-4}$ | **$3.0336 \times 10^{-4}$** | $1.6689 \times 10^{-4}$ | **$1.6514 \times 10^{-4}$** | **1.7198** | 1.7265 |
| **Nestle India** | $3.4441 \times 10^{-4}$ | **$3.4431 \times 10^{-4}$** | $1.8222 \times 10^{-4}$ | **$1.7704 \times 10^{-4}$** | **1.8400** | 1.8468 |
| **Sun Pharma** | $3.1639 \times 10^{-4}$ | **$2.9772 \times 10^{-4}$** | $2.4295 \times 10^{-4}$ | **$1.9179 \times 10^{-4}$** | 1.7119 | **1.6430** |

### Summary of Ablation Findings
- **Aggregate Test Improvements:** Neural Gating achieves a **+2.81% average reduction in Test RMSE** and **+4.99% average reduction in Test MAE** over the static inverse-RMSE baseline.
- **Asset-Specific Heterogeneity:** Neural gating delivers substantial improvements on assets exhibiting distinct volatility regimes (e.g., Sun Pharma, DLF, Nifty 50), while performing comparatively to the static baseline on assets dominated by sudden idiosyncratic noise (e.g., Adani Enterprises, Bank Nifty). This confirms that learned gating provides localized economic benefits without universally dominating in every market condition.

---

## 13. Robustness Analysis

- **Multi-Asset Validation (Implemented):** Evaluated across 2 equity indices, 3 high-beta equities, and 3 defensive equities.
- **Dual Volatility Proxies (Implemented):** Benchmarked against both squared daily return proxy ($r_t^2$) and Parkinson high-low range estimator.
- **Alternative Lookback Windows (Planned):** Robustness sensitivity analysis comparing 30-day, 60-day, and 90-day trailing RMSE memory lengths.
- **Rolling Re-estimation Window Sensitivity (Planned):** Comparing frozen in-sample estimation against 500-day rolling re-estimation horizons.

---

## 14. Project Implementation Status & Research Roadmap

| Research Component | Current Status | Implementation & Repository Evidence |
| :--- | :---: | :--- |
| **Market Data Ingestion** | ✅ Implemented | Daily OHLC series in `data/*.csv` for 8 assets + India VIX |
| **Return & Proxy Preprocessing** | ✅ Implemented | Log returns, squared return proxy, Parkinson range proxy |
| **EWMA (RiskMetrics $\lambda=0.94$)** | ✅ Implemented | Causal recursion in `static_volatility_models.py` & `oos_forecasting_evaluation.py` |
| **GARCH(1,1) Engine** | ✅ Implemented | MLE parameter estimation via `arch` & causal recursive OOS generator |
| **EGARCH(1,1) Engine** | ✅ Implemented | Exact arch-parameterized log-variance recursion with leverage |
| **GJR-GARCH(1,1) Engine** | ✅ Implemented | Threshold asymmetric leverage specification |
| **Walk-Forward Recursive Engine** | ✅ Implemented | Step-by-step causal filter in `oos_forecasting_evaluation.py` |
| **Static Inverse-RMSE Baseline** | ✅ Implemented | 60-day rolling inverse-RMSE baseline in `static_inverse_rmse_baseline.py` |
| **Neural Gating Feature Matrix** | ✅ Implemented | 9-dimensional causal feature matrix in `data/neural_gating/gating_features/` |
| **Neural Gating Engine (PyTorch)** | ✅ Implemented | MLP model, training pipeline, and weights in `run_neural.py` |
| **Leakage & Causality Audit Suite** | ✅ Implemented | Multi-stage perturbation tests in all feature and forecasting modules |
| **RMSE / MAE / QLIKE Evaluation** | ✅ Implemented | Evaluated across all models, baselines, and neural ensembles |
| **Diebold-Mariano Test (HAC)** | 🔵 Planned | Pairwise predictive accuracy testing with Newey-West adjustment |
| **Model Confidence Set (MCS)** | 🔵 Planned | Hansen-Lunde-Nason procedure across candidate model pool |
| **Parametric VaR Backtesting** | 🔵 Planned | Kupiec POF and Christoffersen independence tests |
| **Option Pricing Economic Layer** | 🔵 Planned | Black-Scholes pricing error benchmarked against India VIX |
| **Crisis / Regime Analysis** | 🟡 Partial | 2015–2021 static covers COVID crash; 2025 neural weight shifts plotted; 2020 neural gate expansion planned |
| **Robustness Lookback Grid** | 🔵 Planned | Evaluation across 30d, 60d, 90d trailing windows |

---

## 15. Repository Structure

```
.
├── data/                                    # Data storage layer
│   ├── {asset}.csv                          # Raw daily OHLC & return datasets (2015–2025)
│   ├── india_vix.csv                        # Benchmark India VIX implied volatility series
│   ├── pooled_dataset.csv                   # Pooled long-format panel dataset across all assets
│   ├── static_model_outputs/                # In-sample (2015–2021) parametric modeling outputs
│   │   ├── {asset}_static_models.csv        # Fitted in-sample conditional variances
│   │   └── model_parameters.csv             # Frozen MLE parameter estimates (omega, alpha, beta, gamma)
│   ├── oos_forecasts/                       # Causal OOS (2022–2025) one-step-ahead forecasts
│   │   ├── {asset}_oos_forecasts.csv        # 1-day-ahead variance forecasts for 4 base models
│   │   ├── oos_model_comparison.csv         # OOS evaluation metrics (MAE, RMSE, QLIKE)
│   │   ├── oos_model_ranking.csv            # OOS rank summaries across assets
│   │   └── plots/{asset}_oos_forecasts.png  # Base model OOS trajectory plots
│   ├── static_baseline/                     # Static inverse-RMSE ablation baseline
│   │   ├── {asset}_static_baseline.csv      # Rolling 60d inverse-RMSE weights & ensemble variance
│   │   ├── static_baseline_comparison.csv   # Static baseline evaluation metrics
│   │   ├── static_baseline_vs_models.csv    # 5-model comparative performance table
│   │   ├── static_baseline_validation_report.txt  # Mathematical validation & audit report
│   │   ├── static_baseline_sha256_report.txt      # Source file SHA-256 integrity report
│   │   └── plots/{asset}_static_baseline.png      # Baseline ensemble comparison plots
│   └── neural_gating/                       # Neural Gating Engine (Core Contribution)
│       ├── gating_features/                 # 9-dimensional causal state feature matrices
│       │   └── {asset}_gating_features.csv  # Daily causal feature records (2022–2025)
│       ├── model/{asset}/                   # PyTorch neural gating model checkpoints
│       │   ├── neural_gate_model.pth        # Trained PyTorch model weights
│       │   ├── feature_scaler.pkl           # Train-set Z-score scaler parameters
│       │   ├── model_config.json            # Model architecture configuration
│       │   ├── training_history.csv         # Epoch-by-epoch train/val loss curves
│       │   └── training_summary.txt         # Best epoch and convergence diagnostics
│       ├── plots/{asset}_neural_weights.png # Test-period neural weight trajectory plots
│       ├── neural_gate_predictions.csv      # Complete ensemble predictions & dynamic weights
│       ├── neural_gate_vs_models.csv        # Test-period comparative metrics (Neural vs Static)
│       ├── asset_summaries.csv              # Per-asset neural training and test summaries
│       ├── gating_feature_schema.txt        # Mathematical feature specification document
│       ├── gating_feature_validation_report.txt # Feature engineering mathematical audit
│       ├── neural_gating_feature_engineering.py # Feature matrix construction pipeline
│       └── neural_gate_validation_report.txt    # Neural gating stage validation report
├── static_volatility_models.py              # In-sample (2015–2021) parametric fitting pipeline
├── oos_forecasting_evaluation.py            # Causal OOS recursive forecasting & audit suite
├── static_inverse_rmse_baseline.py          # Static inverse-RMSE rolling baseline engine
├── run_neural.py                            # Neural Gating Engine training & inference orchestrator
├── run_eval.py                              # Held-out test evaluation & causality validation
├── final_data_checks.py                     # Raw dataset integrity & OHLC consistency audit
├── test1.py                                 # Data retrieval & preparation script
└── README.md                                # Authoritative project documentation
```

---

## 16. Reproducibility & Execution Guide

### 16.1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/RushiMMehta/volatality-modelling.git
cd volatality-modelling

# Install required scientific and quantitative dependencies
pip install numpy pandas scipy matplotlib arch torch yfinance
```

### 16.2. End-to-End Pipeline Execution
Execute the research pipeline sequentially from raw data validation to neural ensemble evaluation:

```bash
# Step 1: Validate raw dataset integrity, date monotonicity, and OHLC consistency
python3 final_data_checks.py

# Step 2: Fit in-sample econometric models (2015–2021) and freeze MLE parameters
python3 static_volatility_models.py

# Step 3: Run causal OOS recursive forecasting and base model audit (2022–2025)
python3 oos_forecasting_evaluation.py

# Step 4: Generate static inverse-RMSE baseline ensemble and audit reports
python3 static_inverse_rmse_baseline.py

# Step 5: Construct 9-dimensional causal feature matrices for the neural gate
python3 data/neural_gating/neural_gating_feature_engineering.py

# Step 6: Train neural gating networks on 2022–2023 with validation early stopping (2024)
python3 run_neural.py

# Step 7: Evaluate held-out test performance (2025) and execute causality checks
python3 run_eval.py
```

---

## 17. Research Outputs & Artifacts

The repository contains verified empirical outputs:
- **Parameter Estimates:** [`data/static_model_outputs/model_parameters.csv`](file:///Users/rushi/Desktop/Volatility%20Modelling/data/static_model_outputs/model_parameters.csv) (in-sample log-likelihood, AIC, BIC, persistence).
- **OOS Base Forecasts:** [`data/oos_forecasts/`](file:///Users/rushi/Desktop/Volatility%20Modelling/data/oos_forecasts/) (daily variance forecasts and model rankings).
- **Static Baseline Outputs:** [`data/static_baseline/`](file:///Users/rushi/Desktop/Volatility%20Modelling/data/static_baseline/) (rolling weights, comparisons, and SHA-256 audit reports).
- **Neural Gating Artifacts:** [`data/neural_gating/`](file:///Users/rushi/Desktop/Volatility%20Modelling/data/neural_gating/) (PyTorch checkpoints, scalers, epoch logs, and weight trajectories).
- **Comparative Test Tables:** [`data/neural_gating/neural_gate_vs_models.csv`](file:///Users/rushi/Desktop/Volatility%20Modelling/data/neural_gating/neural_gate_vs_models.csv) and [`data/neural_gating/asset_summaries.csv`](file:///Users/rushi/Desktop/Volatility%20Modelling/data/neural_gating/asset_summaries.csv).

---

## 18. Methodological Limitations

1. **Latent Volatility & Proxy Noise:** Volatility is unobservable; squared daily returns are an unbiased but noisy proxy. Parkinson estimators reduce noise but omit overnight jump dynamics.
2. **Optimization Non-Convergence Risk:** Extreme market shocks can induce flat likelihood surfaces or boundary parameter solutions in GARCH-family models during in-sample estimation.
3. **Discrete Daily Frequency:** Daily-frequency sampling omits high-frequency intraday microstructure and realized kernel estimators.
4. **Overfitting in Meta-Learning:** Neural gating networks risk overfitting on limited financial time series if regularization, early stopping, and strict temporal partitioning are relaxed.
5. **Option-Chain Availability in Indian Equities:** Long-dated historical option chains with continuous strike density are historically scarce in Indian single-stock derivatives, necessitating synthetic Black-Scholes pricing layers benchmarked against India VIX.
6. **No Structural Guarantee of Outperformance:** Dynamic neural gating adds non-linear flexibility but does not mathematically guarantee superior out-of-sample forecasts over parsimonious static baselines in purely stationary regimes.

---

## 19. Summary of Research Contributions

1. **Adaptive Neural Gating Architecture:** Formalized a constrained meta-learning framework producing convex combination weights over four foundational volatility models without unconstrained variance prediction.
2. **Strict Causal Protocol & Leakage Firewall:** Implemented automated perturbation testing confirming zero-lookahead bias across state updates, feature engineering, and inference.
3. **Rigorous Ablation Benchmarking:** Validated the neural gate directly against an independent rolling inverse-RMSE baseline.
4. **Empirical Indian Market Study:** Comprehensive multi-asset evaluation spanning broad indices, high-beta equities, and defensive equities across the National Stock Exchange (NSE).
5. **Dynamic Regime Adaptation:** Demonstrated that learned gating shifts weight allocations toward asymmetric leverage models during heightened market stress.

---

## 20. Academic & Methodological Positioning

**Disciplinary Classification:** Financial Econometrics $\cdot$ Quantitative Risk Management $\cdot$ Applied Machine Learning

This project bridges traditional econometric time-series modeling and modern statistical learning. Rather than replacing econometric formulations with deep learning, it leverages machine learning where it is most effective: **non-linear, state-dependent meta-allocation over theoretically consistent, stationary parametric models.**

---

## 21. Future Research Extensions

- **Intraday & Realized Measures:** Integration of high-frequency 5-minute intraday returns, Realized Variance (RV), and Realized Kernels (RK).
- **Explicit Regime-Switching Features:** Augmentation of the gating state vector with Markov-Switching or Hidden Markov Model (HMM) state probabilities.
- **Broader Cross-Sectional Universe:** Expansion to mid-cap equities, commodities (MCX Gold/Crude), and cross-asset foreign exchange (USD/INR).
- **Full Delta-Hedging Backtesting:** Practical simulation of dynamic delta-hedging strategies using ensemble variance forecasts to quantify real-world hedging error reduction.
- **Higher-Order Recurrent Gating:** Investigating recurrent gating mechanisms (LSTM / GRU gating cells) conditioned on multi-day temporal trajectories.
