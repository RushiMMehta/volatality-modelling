"""
Data Retrieval & Preparation
Adaptive Institutional Volatility Forecasting Engine (Neural-Gated Ensemble)

Fetches, cleans, and prepares daily OHLC price data for:
  - 2 indices        : Nifty 50, Bank Nifty
  - 3 high-beta stocks: Adani Enterprises, Tata Steel, DLF
  - 3 defensive stocks: Hindustan Unilever, Nestle India, Sun Pharma

Produces, per asset:
  - open/high/low/close prices
  - close-to-close log returns (input to the four base volatility models)
  - realized_var_proxy      : squared log return (r_t^2), the standard proxy
  - parkinson_var_proxy     : range-based (High/Low) variance estimator,
                              a less noisy secondary proxy for evaluation

Output files:
  - One clean CSV per asset  (data/<name>.csv)
  - One pooled long-format CSV across all assets (data/pooled_dataset.csv)
    -> this is the file you'll use later to train the shared neural gate

Run:
  pip install yfinance pandas numpy --break-system-packages
  python data_loader.py
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# 1. Asset universe
# ---------------------------------------------------------------------------
# Chosen for sector spread, not just count:
#   - Indices give you the broad-market baseline
#   - High-beta names are picked from three DIFFERENT high-volatility sectors
#     (conglomerate/infra, metals, realty) rather than three similar stocks
#   - Defensive names are picked from two classically low-beta sectors
#     (FMCG x2, pharma) for the same reason

@dataclass
class Asset:
    name: str        # short key used for filenames / pooled dataset
    ticker: str       # yfinance ticker
    category: str      # "index" | "high_beta" | "defensive"
    sector: str


ASSETS = [
    Asset("nifty50",     "^NSEI",         "index",      "Broad Market"),
    Asset("banknifty",   "^NSEBANK",      "index",      "Banking"),

    Asset("adanient",    "ADANIENT.NS",   "high_beta",  "Conglomerate / Infra"),
    Asset("tatasteel",   "TATASTEEL.NS",  "high_beta",  "Metals"),
    Asset("dlf",         "DLF.NS",        "high_beta",  "Realty"),

    Asset("hindunilvr",  "HINDUNILVR.NS", "defensive",  "FMCG"),
    Asset("nestleind",   "NESTLEIND.NS",  "defensive",  "FMCG"),
    Asset("sunpharma",   "SUNPHARMA.NS",  "defensive",  "Pharma"),
]

# Fetched separately as a benchmark series, not part of the ensemble universe
INDIA_VIX_TICKER = "^INDIAVIX"

START = "2015-01-01"
END = "2025-12-31"

# Simplifying assumption for Week 10.5's Black-Scholes economic-value layer
# (Section 3.3 of the blueprint needs an r). A single constant annualized
# risk-free rate is a standard, documented simplification for a project at
# this scope -- state it explicitly in the paper rather than pretending it's
# time-varying. Update this to roughly track the prevailing RBI repo rate /
# 91-day T-bill yield at the time you run the pricing step.
RISK_FREE_RATE_ANNUAL = 0.065  # ~6.5%, approximate -- document this assumption in the paper

OUT_DIR = "data"


# ---------------------------------------------------------------------------
# 2. Fetch + clean
# ---------------------------------------------------------------------------

def fetch_prices(ticker: str) -> pd.DataFrame:
    """Download daily OHLC prices for a ticker and return a clean DataFrame."""
    df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} -- check the ticker symbol.")

    # Newer yfinance versions can return MultiIndex columns (e.g. ('Close', '^NSEI'))
    # even for a single ticker. Flatten to plain column names before doing anything else.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep Open/High/Low too, not just Close: not needed for fitting the base
    # models (those use close-to-close log returns, the field standard), but
    # High/Low lets us build a less noisy realized-volatility proxy below.
    df = df[["Open", "High", "Low", "Close"]].rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
    )
    df.index.name = "date"
    df = df.dropna()

    # Drop duplicate dates if any, keep the last observation
    df = df[~df.index.duplicated(keep="last")]
    return df


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily log returns and a squared-return realized-volatility proxy."""
    df = df.copy()

    close = df["close"]
    if isinstance(close, pd.DataFrame):
        # Defensive fallback: if "close" somehow resolved to a DataFrame
        # (e.g. duplicate column names), collapse it to a plain Series.
        close = close.iloc[:, 0]

    df["log_return"] = np.log(close / close.shift(1))
    assert "log_return" in df.columns and df["log_return"].ndim == 1, (
        "log_return did not come out as a flat column -- check df.columns for "
        "unexpected MultiIndex or duplicate names before proceeding."
    )
    df = df.dropna(subset=["log_return"])
    df["realized_var_proxy"] = df["log_return"] ** 2  # r_t^2, the standard (noisy) proxy

    # Parkinson (1980) range-based variance estimator -- uses High/Low instead of
    # just the close-to-close return. Statistically more efficient (less noisy)
    # than the squared-return proxy above, so it's useful as a secondary /
    # robustness realized-volatility target when evaluating forecasts later.
    # Formula: (1 / (4 * ln2)) * (ln(High/Low))^2
    if "high" in df.columns and "low" in df.columns:
        df["parkinson_var_proxy"] = (1.0 / (4.0 * np.log(2.0))) * (
            np.log(df["high"] / df["low"]) ** 2
        )

    return df


def flag_data_quality(df: pd.DataFrame, name: str) -> None:
    """Print basic sanity checks -- run this for every asset before trusting it."""
    n_rows = len(df)
    date_span_days = (df.index.max() - df.index.min()).days
    expected_trading_days = date_span_days * (252 / 365)
    coverage_pct = 100 * n_rows / expected_trading_days if expected_trading_days else 0

    extreme_moves = df[df["log_return"].abs() > 0.10]  # >10% single-day move

    print(f"  [{name}] rows={n_rows}, span={df.index.min().date()} -> {df.index.max().date()}, "
          f"coverage~{coverage_pct:.0f}% of expected trading days, "
          f"extreme-move days={len(extreme_moves)}")
    if len(extreme_moves) > 0:
        print(f"    -> extreme-move dates: {list(extreme_moves.index.date)}")
        print("       Check these against known splits/bonus issues/corporate actions "
              "before trusting them as genuine market moves.")


# ---------------------------------------------------------------------------
# 3. Build per-asset files + pooled long-format dataset
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled_frames = []

    print("Fetching individual assets...")
    for asset in ASSETS:
        print(f"\n{asset.name} ({asset.ticker}) -- {asset.category}, {asset.sector}")
        raw = fetch_prices(asset.ticker)
        clean = add_log_returns(raw)
        flag_data_quality(clean, asset.name)

        out_path = os.path.join(OUT_DIR, f"{asset.name}.csv")
        clean.to_csv(out_path)
        print(f"  -> saved {out_path}")

        pooled = clean.reset_index()[
            ["date", "open", "high", "low", "close", "log_return",
             "realized_var_proxy", "parkinson_var_proxy"]
        ].copy()
        pooled["asset"] = asset.name
        pooled["category"] = asset.category
        pooled["sector"] = asset.sector
        pooled_frames.append(pooled)

    print("\nFetching India VIX (benchmark series, not part of the ensemble universe)...")
    vix_raw = fetch_prices(INDIA_VIX_TICKER)
    vix_raw = vix_raw.rename(columns={"close": "india_vix"})
    vix_path = os.path.join(OUT_DIR, "india_vix.csv")
    vix_raw.to_csv(vix_path)
    print(f"  -> saved {vix_path}")

    # ---- Pooled long-format dataset for training the shared neural gate ----
    pooled_df = pd.concat(pooled_frames, ignore_index=True)
    pooled_df = pooled_df.sort_values(["date", "asset"]).reset_index(drop=True)
    pooled_path = os.path.join(OUT_DIR, "pooled_dataset.csv")
    pooled_df.to_csv(pooled_path, index=False)

    print(f"\nPooled dataset saved -> {pooled_path}")
    print(f"  {pooled_df['asset'].nunique()} assets, {len(pooled_df)} total rows")
    print(pooled_df.groupby(["category", "asset"]).size().rename("rows"))

    print(
        "\nReminder: when you split this pooled dataset for the gating network, "
        "split by DATE (walk-forward), not by asset or randomly -- every asset's "
        "training window must end before any asset's validation window begins."
    )
    print(
        f"\nRisk-free rate for the Week 10.5 Black-Scholes economic-value layer is "
        f"set as a constant ({RISK_FREE_RATE_ANNUAL:.1%} annualized) in this script. "
        f"Document this simplification explicitly in Section 3 of the paper."
    )


if __name__ == "__main__":
    main()
