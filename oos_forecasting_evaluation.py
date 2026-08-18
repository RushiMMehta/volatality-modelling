"""
oos_forecasting_evaluation.py
==============================
Out-of-Sample (OOS) Volatility Forecasting and Evaluation — 2022-2025
======================================================================

IMPORTANT DESIGN NOTES
-----------------------
This script produces TRUE ONE-STEP-AHEAD OOS forecasts with ZERO look-ahead bias.

Key principle:
    forecast_variance[t]  is computed using ONLY information through t-1.
    return[t] is observed ONLY AFTER storing forecast[t], and is then
    used to update the volatility state for forecasting t+1.

Previous implementation used arch_model.fix() + conditional_volatility on the
full concatenated (in-sample + OOS) series, then sliced the OOS window. That
approach is look-ahead biased because the filtered conditional_volatility[t]
incorporates r[t] as an input. This script replaces that with explicit causal
recursive loops for every GARCH-family model.

EGARCH parameterization:
    Exactly reproduces the arch library EGARCH(1,1) recursion as used in
    static_volatility_models.py:
        arch_model(r_scaled, vol="EGARCH", p=1, o=1, q=1, mean="Zero", dist="normal")
    The recursion (from arch/univariate/recursions_python.py egarch_recursion) is:
        lnsigma2[t] = omega + alpha*(|e_{t-1}| - SQRT2_OV_PI) + gamma*e_{t-1} + beta*lnsigma2[t-1]
        sigma2[t]   = exp(lnsigma2[t])
        e_{t-1}     = resid_{t-1} / sqrt(sigma2_{t-1})
    where SQRT2_OV_PI = 0.79788456080286541 = E[|e|] for e ~ N(0,1).
    Working in percent-scaled space (r*100, sigma_scaled^2 = sigma_decimal^2 * 10000)
    exactly as the arch library does, using omega_raw = omega_dec + ln(10000).

No arch_model.fix() calls are made on OOS data. No model is refitted.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EGARCH: E[|e|] for e ~ N(0,1) — exactly as used in arch library recursions_python.py
SQRT2_OV_PI = 0.79788456080286541

ASSETS = [
    "nifty50",
    "banknifty",
    "adanient",
    "tatasteel",
    "dlf",
    "hindunilvr",
    "nestleind",
    "sunpharma",
]

MODELS = ["EWMA", "GARCH", "EGARCH", "GJR-GARCH"]

OOS_START_DATE = "2022-01-01"
OOS_END_DATE   = "2025-12-31"

INPUT_DATA_DIR   = "data"
PARAM_FILE_PATH  = os.path.join("data", "static_model_outputs", "model_parameters.csv")
STATIC_OUT_DIR   = os.path.join("data", "static_model_outputs")
OOS_OUTPUT_DIR   = os.path.join("data", "oos_forecasts")
PLOTS_OUTPUT_DIR = os.path.join(OOS_OUTPUT_DIR, "plots")

EWMA_LAMBDA = 0.94


# ===========================================================================
#  CAUSAL OOS FORECAST FUNCTIONS
# ===========================================================================

def ewma_oos_forecasts(returns_full, oos_mask, sigma2_init, lam=EWMA_LAMBDA):
    """
    Causal EWMA one-step-ahead variance forecasts for OOS period.

    Recursion:
        sigma2[t] = lam * sigma2[t-1] + (1-lam) * r[t-1]^2

    sigma2[t] is the forecast FOR date t, built using only r[t-1] and sigma2[t-1].
    Returns array of length = number of OOS dates.
    """
    T = len(returns_full)
    sigma2 = np.full(T, np.nan)
    oos_indices = np.where(oos_mask)[0]
    if len(oos_indices) == 0:
        return np.array([])

    first_oos_idx = oos_indices[0]
    current_sigma2 = sigma2_init

    for t in range(first_oos_idx, T):
        r_prev = returns_full[t - 1]
        forecast = lam * current_sigma2 + (1 - lam) * (r_prev ** 2)
        sigma2[t] = forecast
        current_sigma2 = forecast

    return sigma2[oos_mask]


def garch_oos_forecasts(returns_full, oos_mask, omega_dec, alpha, beta, sigma2_init):
    """
    Causal GARCH(1,1) one-step-ahead variance forecasts for OOS period.

    Recursion (decimal scale):
        sigma2[t] = omega + alpha * r[t-1]^2 + beta * sigma2[t-1]

    omega_dec is already in decimal scale (omega_raw / 10000 from static estimation).
    sigma2_init is the last in-sample conditional variance (decimal scale).
    """
    T = len(returns_full)
    sigma2 = np.full(T, np.nan)
    oos_indices = np.where(oos_mask)[0]
    if len(oos_indices) == 0:
        return np.array([])

    first_oos_idx = oos_indices[0]
    current_sigma2 = sigma2_init

    for t in range(first_oos_idx, T):
        r_prev = returns_full[t - 1]
        forecast = omega_dec + alpha * (r_prev ** 2) + beta * current_sigma2
        if forecast <= 0:
            forecast = max(omega_dec, 1e-12)
        sigma2[t] = forecast
        current_sigma2 = forecast

    return sigma2[oos_mask]


def egarch_oos_forecasts(returns_full, oos_mask, omega_dec, alpha, gamma, beta,
                          sigma2_init_dec, r_last_is):
    """
    Causal EGARCH(1,1) one-step-ahead variance forecasts for OOS period.

    Reproduces EXACTLY the arch library EGARCH(1,1) recursion used in
    static_volatility_models.py:
        arch_model(r*100, vol="EGARCH", p=1, o=1, q=1, mean="Zero", dist="normal")

    From arch/univariate/recursions_python.py (egarch_recursion_python):

        lnsigma2[t] = omega_raw
                     + alpha * (|e_{t-1}| - SQRT2_OV_PI)
                     + gamma * e_{t-1}
                     + beta  * lnsigma2[t-1]

        sigma2_scaled[t] = exp(lnsigma2[t])
        e_{t-1} = r_scaled_{t-1} / sqrt(sigma2_scaled_{t-1})

    where:
        r_scaled     = r_decimal * 100
        sigma2_scaled = sigma2_decimal * 10000
        omega_raw    = omega_dec + ln(10000)

    After each step: sigma2_decimal[t] = sigma2_scaled[t] / 10000

    Parameters
    ----------
    omega_dec : float
        omega stored in model_parameters.csv (decimal = omega_raw - ln(10000))
    sigma2_init_dec : float
        Last in-sample EGARCH conditional variance in decimal scale.
    r_last_is : float
        Last in-sample return (decimal) to compute e_{t-1} for first OOS step.
    """
    SCALE     = 100.0
    SCALE_VAR = SCALE ** 2  # 10000
    LNSIGMA_MAX = float(np.log(np.finfo(np.double).max) - 0.1)

    omega_raw = omega_dec + np.log(SCALE_VAR)

    T = len(returns_full)
    sigma2_dec = np.full(T, np.nan)
    oos_indices = np.where(oos_mask)[0]
    if len(oos_indices) == 0:
        return np.array([])

    first_oos_idx = oos_indices[0]

    # Convert init state to scaled space
    sigma2_scaled_prev = max(sigma2_init_dec * SCALE_VAR, 1e-300)
    lnsigma2_scaled_prev = np.log(sigma2_scaled_prev)

    # Standardized residual from last in-sample observation
    r_scaled_prev = r_last_is * SCALE
    e_prev = r_scaled_prev / np.sqrt(max(sigma2_scaled_prev, 1e-300))

    for t in range(first_oos_idx, T):
        # Forecast lnsigma2_scaled[t] using only t-1 information
        abs_e_prev = abs(e_prev)
        lnsigma2_t = (
            omega_raw
            + alpha * (abs_e_prev - SQRT2_OV_PI)
            + gamma * e_prev
            + beta  * lnsigma2_scaled_prev
        )
        lnsigma2_t = min(lnsigma2_t, LNSIGMA_MAX)
        sigma2_scaled_t = np.exp(lnsigma2_t)

        sigma2_dec[t] = sigma2_scaled_t / SCALE_VAR

        # Now observe r[t] to update state for t+1
        r_scaled_t = returns_full[t] * SCALE
        e_t = r_scaled_t / np.sqrt(max(sigma2_scaled_t, 1e-300))

        lnsigma2_scaled_prev = lnsigma2_t
        e_prev               = e_t

    return sigma2_dec[oos_mask]


def gjr_garch_oos_forecasts(returns_full, oos_mask, omega_dec, alpha, gamma, beta, sigma2_init):
    """
    Causal GJR-GARCH(1,1) one-step-ahead variance forecasts for OOS period.

    Recursion (decimal scale):
        sigma2[t] = omega + alpha*r[t-1]^2 + gamma*I(r[t-1]<0)*r[t-1]^2 + beta*sigma2[t-1]

    omega_dec is in decimal scale (omega_raw / 10000).
    """
    T = len(returns_full)
    sigma2 = np.full(T, np.nan)
    oos_indices = np.where(oos_mask)[0]
    if len(oos_indices) == 0:
        return np.array([])

    first_oos_idx = oos_indices[0]
    current_sigma2 = sigma2_init

    for t in range(first_oos_idx, T):
        r_prev = returns_full[t - 1]
        r_prev_sq = r_prev ** 2
        leverage = 1.0 if r_prev < 0 else 0.0
        forecast = (
            omega_dec
            + alpha * r_prev_sq
            + gamma * leverage * r_prev_sq
            + beta  * current_sigma2
        )
        if forecast <= 0:
            forecast = max(omega_dec, 1e-12)
        sigma2[t] = forecast
        current_sigma2 = forecast

    return sigma2[oos_mask]


# ===========================================================================
#  LOOK-AHEAD BIAS (LEAKAGE) TEST
# ===========================================================================

def run_leakage_test(asset, df_raw, last_is_row, asset_params):
    """
    Leakage / causality test.

    For the first OOS date t0:
      - forecast[t0] must NOT change when we perturb r[t0].
      - forecast[t0+1] MUST change when we perturb r[t0].

    Returns dict with PASS/FAIL per model and diagnostic details.
    """
    returns_full = df_raw["log_return"].values
    oos_mask = (
        (df_raw["date"] >= pd.to_datetime(OOS_START_DATE)) &
        (df_raw["date"] <= pd.to_datetime(OOS_END_DATE))
    ).values

    oos_indices = np.where(oos_mask)[0]
    if len(oos_indices) < 2:
        return {"status": "SKIP", "reason": "Fewer than 2 OOS observations"}

    first_oos_idx = oos_indices[0]

    # Perturbed copy: r[t0] *= 1000
    # Large perturbation is needed to detect causality even when model parameters
    # like alpha are effectively zero (optimizer boundary). The test verifies the
    # STRUCTURAL property that forecast[t0] is independent of r[t0], which holds
    # regardless of parameter values.
    returns_perturbed = returns_full.copy()
    returns_perturbed[first_oos_idx] *= 1000.0

    sigma2_ewma_init   = float(last_is_row["ewma_variance"])
    sigma2_garch_init  = float(last_is_row["garch_variance"])
    sigma2_egarch_init = float(last_is_row["egarch_variance"])
    sigma2_gjr_init    = float(last_is_row["gjr_garch_variance"])
    is_mask_raw = df_raw["date"] < pd.to_datetime(OOS_START_DATE)
    r_last_is   = float(df_raw.loc[is_mask_raw, "log_return"].iloc[-1])

    def _run_all(ret_arr):
        ew = ewma_oos_forecasts(ret_arr, oos_mask, sigma2_ewma_init)

        gr = asset_params.loc["GARCH"]
        ga = garch_oos_forecasts(ret_arr, oos_mask, float(gr["omega"]),
                                  float(gr["alpha"]), float(gr["beta"]), sigma2_garch_init)

        er = asset_params.loc["EGARCH"]
        eg = egarch_oos_forecasts(ret_arr, oos_mask, float(er["omega"]),
                                   float(er["alpha"]), float(er["gamma"]), float(er["beta"]),
                                   sigma2_egarch_init, r_last_is)

        jr = asset_params.loc["GJR-GARCH"]
        gj = gjr_garch_oos_forecasts(ret_arr, oos_mask, float(jr["omega"]),
                                      float(jr["alpha"]), float(jr["gamma"]), float(jr["beta"]),
                                      sigma2_gjr_init)
        return ew, ga, eg, gj

    orig_forecasts = _run_all(returns_full)
    pert_forecasts = _run_all(returns_perturbed)

    model_names = ["EWMA", "GARCH", "EGARCH", "GJR-GARCH"]
    results = {}
    all_pass = True

    for i, mname in enumerate(model_names):
        f_orig = orig_forecasts[i]
        f_pert = pert_forecasts[i]

        if len(f_orig) < 2:
            results[mname] = {"t0_unchanged": "SKIP", "t1_changed": "SKIP"}
            continue

        # t0 forecast must be EXACTLY identical (no r[t0] input at all)
        t0_unchanged = np.isclose(f_orig[0], f_pert[0], rtol=1e-12)
        # t1 forecast must differ — use atol to catch even tiny differences
        # (robust to near-zero alpha at optimizer boundary)
        t1_changed   = not np.isclose(f_orig[1], f_pert[1], rtol=1e-12, atol=1e-16)

        if not t0_unchanged or not t1_changed:
            all_pass = False

        results[mname] = {
            "t0_forecast_orig": f_orig[0],
            "t0_forecast_pert": f_pert[0],
            "t0_unchanged":     "PASS" if t0_unchanged else "FAIL",
            "t1_forecast_orig": f_orig[1],
            "t1_forecast_pert": f_pert[1],
            "t1_changed":       "PASS" if t1_changed else "FAIL",
        }

    results["overall"] = "PASS" if all_pass else "FAIL"
    return results


# ===========================================================================
#  EVALUATION METRICS
# ===========================================================================

def compute_metrics(y, f):
    """MAE, RMSE, QLIKE between target y and forecast f."""
    valid_mask = (y > 0) & (f > 0) & np.isfinite(y) & np.isfinite(f)
    y_v = y[valid_mask]
    f_v = f[valid_mask]
    n_valid = valid_mask.sum()
    n_zero  = len(y) - n_valid

    mae  = np.mean(np.abs(y - f))
    rmse = np.sqrt(np.mean((y - f) ** 2))

    if n_valid > 0:
        ratio = f_v / y_v
        qlike = np.mean(ratio - np.log(ratio) - 1.0)
    else:
        qlike = np.nan

    return {"MAE": mae, "RMSE": rmse, "QLIKE": qlike,
            "n_valid_qlike": n_valid, "n_zero_target": n_zero}


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    os.makedirs(OOS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)

    print("=" * 78)
    print("OOS VOLATILITY FORECASTING & EVALUATION (2022-2025)")
    print("Causal one-step-ahead forecasts — frozen 2015-2021 parameters")
    print("=" * 78)

    if not os.path.exists(PARAM_FILE_PATH):
        raise FileNotFoundError(f"Parameter file not found: {PARAM_FILE_PATH}")

    df_params = pd.read_csv(PARAM_FILE_PATH)
    print(f"\n[OK] Loaded frozen parameters from: {PARAM_FILE_PATH}")

    oos_forecast_files = []
    validation_reports = []
    comparison_rows    = []
    oos_dfs            = {}
    leakage_results    = {}

    for asset in ASSETS:
        print(f"\n{'=' * 78}")
        print(f"  ASSET: {asset.upper()}")
        print(f"{'=' * 78}")

        # Load raw full history
        raw_csv_path = os.path.join(INPUT_DATA_DIR, f"{asset}.csv")
        df_raw = pd.read_csv(raw_csv_path)
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        df_raw = df_raw.sort_values("date").reset_index(drop=True)

        returns_full = df_raw["log_return"].values
        T_total = len(returns_full)

        oos_mask = (
            (df_raw["date"] >= pd.to_datetime(OOS_START_DATE)) &
            (df_raw["date"] <= pd.to_datetime(OOS_END_DATE))
        ).values

        # Load last in-sample static model output for warm-start
        static_csv_path = os.path.join(STATIC_OUT_DIR, f"{asset}_static_models.csv")
        df_static_is = pd.read_csv(static_csv_path)
        df_static_is["date"] = pd.to_datetime(df_static_is["date"])
        df_static_is = df_static_is.sort_values("date").reset_index(drop=True)
        last_is_row = df_static_is.iloc[-1]

        sigma2_ewma_init   = float(last_is_row["ewma_variance"])
        sigma2_garch_init  = float(last_is_row["garch_variance"])
        sigma2_egarch_init = float(last_is_row["egarch_variance"])
        sigma2_gjr_init    = float(last_is_row["gjr_garch_variance"])

        is_mask_raw = df_raw["date"] < pd.to_datetime(OOS_START_DATE)
        r_last_is   = float(df_raw.loc[is_mask_raw, "log_return"].iloc[-1])

        print(f"  Warm-start from last IS date: {last_is_row['date'].strftime('%Y-%m-%d')}")
        print(f"    EWMA   sigma2_init = {sigma2_ewma_init:.6e}")
        print(f"    GARCH  sigma2_init = {sigma2_garch_init:.6e}")
        print(f"    EGARCH sigma2_init = {sigma2_egarch_init:.6e}")
        print(f"    GJR    sigma2_init = {sigma2_gjr_init:.6e}")
        print(f"    r_last_is          = {r_last_is:.8f}")

        asset_p_df = df_params[df_params["asset"] == asset].set_index("model")

        # EWMA
        ewma_var_oos = ewma_oos_forecasts(returns_full, oos_mask, sigma2_ewma_init)
        ewma_vol_oos = np.sqrt(ewma_var_oos)

        # GARCH
        gr = asset_p_df.loc["GARCH"]
        garch_var_oos = garch_oos_forecasts(
            returns_full, oos_mask,
            float(gr["omega"]), float(gr["alpha"]), float(gr["beta"]), sigma2_garch_init)
        garch_vol_oos = np.sqrt(garch_var_oos)

        # EGARCH — exact arch library parameterization
        er = asset_p_df.loc["EGARCH"]
        egarch_var_oos = egarch_oos_forecasts(
            returns_full, oos_mask,
            float(er["omega"]), float(er["alpha"]), float(er["gamma"]), float(er["beta"]),
            sigma2_egarch_init, r_last_is)
        egarch_vol_oos = np.sqrt(egarch_var_oos)

        # GJR-GARCH
        jr = asset_p_df.loc["GJR-GARCH"]
        gjr_var_oos = gjr_garch_oos_forecasts(
            returns_full, oos_mask,
            float(jr["omega"]), float(jr["alpha"]), float(jr["gamma"]), float(jr["beta"]),
            sigma2_gjr_init)
        gjr_vol_oos = np.sqrt(gjr_var_oos)

        # Build OOS DataFrame — preserve existing column names
        df_oos = df_raw.loc[oos_mask].copy().reset_index(drop=True)
        df_oos["ewma_forecast_variance"]        = ewma_var_oos
        df_oos["ewma_forecast_volatility"]      = ewma_vol_oos
        df_oos["garch_forecast_variance"]       = garch_var_oos
        df_oos["garch_forecast_volatility"]     = garch_vol_oos
        df_oos["egarch_forecast_variance"]      = egarch_var_oos
        df_oos["egarch_forecast_volatility"]    = egarch_vol_oos
        df_oos["gjr_garch_forecast_variance"]   = gjr_var_oos
        df_oos["gjr_garch_forecast_volatility"] = gjr_vol_oos
        df_oos["date_str"] = df_oos["date"].dt.strftime("%Y-%m-%d")

        # Save OOS CSV
        forecast_cols = [
            "date_str", "log_return", "realized_var_proxy", "parkinson_var_proxy",
            "ewma_forecast_variance", "ewma_forecast_volatility",
            "garch_forecast_variance", "garch_forecast_volatility",
            "egarch_forecast_variance", "egarch_forecast_volatility",
            "gjr_garch_forecast_variance", "gjr_garch_forecast_volatility",
        ]
        df_out = df_oos[forecast_cols].copy()
        df_out.rename(columns={"date_str": "date"}, inplace=True)

        out_csv = os.path.join(OOS_OUTPUT_DIR, f"{asset}_oos_forecasts.csv")
        df_out.to_csv(out_csv, index=False)
        oos_forecast_files.append(out_csv)
        oos_dfs[asset] = df_oos

        # Print first 5 OOS observations
        print(f"\n  First 5 OOS observations (GARCH — causal ordering verification):")
        print(f"  {'Date':>12}  {'Prev Return':>14}  {'Fcast Var (G)':>16}  "
              f"{'Actual Return':>14}  {'Actual RV':>14}")
        oos_indices_all = np.where(oos_mask)[0]
        for k in range(min(5, len(oos_indices_all))):
            t_idx      = oos_indices_all[k]
            row_date   = df_raw.iloc[t_idx]["date"].strftime("%Y-%m-%d")
            r_prev     = returns_full[t_idx - 1]
            fcast_g    = garch_var_oos[k]
            r_actual   = returns_full[t_idx]
            rv_actual  = df_raw.iloc[t_idx]["realized_var_proxy"]
            print(f"  {row_date:>12}  {r_prev:>14.8f}  {fcast_g:>16.8e}  "
                  f"{r_actual:>14.8f}  {rv_actual:>14.8e}")

        # Validation
        n_obs      = len(df_oos)
        first_date = df_oos["date_str"].min()
        last_date  = df_oos["date_str"].max()

        var_cols = ["ewma_forecast_variance", "garch_forecast_variance",
                    "egarch_forecast_variance", "gjr_garch_forecast_variance"]
        vol_cols = ["ewma_forecast_volatility", "garch_forecast_volatility",
                    "egarch_forecast_volatility", "gjr_garch_forecast_volatility"]

        nan_count      = df_out.isna().sum().sum()
        inf_count      = np.isinf(df_out.select_dtypes(include=[np.number])).sum().sum()
        neg_var_count  = sum((df_oos[c] < 0).sum() for c in var_cols)
        zero_var_count = sum((df_oos[c] <= 0).sum() for c in var_cols)
        dup_dates      = df_oos["date_str"].duplicated().sum()
        miss_forecasts = sum(df_oos[c].isna().sum() for c in var_cols)

        vol_sqrt_check = all(
            np.allclose(df_oos[vc].values,
                        np.sqrt(df_oos[vc.replace("volatility", "variance")].values),
                        rtol=1e-9)
            for vc in vol_cols
        )

        pass_all = (nan_count == 0 and inf_count == 0 and neg_var_count == 0 and
                    zero_var_count == 0 and dup_dates == 0 and miss_forecasts == 0 and
                    vol_sqrt_check)

        validation_reports.append({
            "asset": asset, "n_obs": n_obs,
            "first_date": first_date, "last_date": last_date,
            "nan_count": nan_count, "inf_count": inf_count,
            "neg_var_count": neg_var_count, "zero_var_count": zero_var_count,
            "dup_dates": dup_dates, "miss_forecasts": miss_forecasts,
            "vol_eq_sqrt_var": "PASS" if vol_sqrt_check else "FAIL",
            "params_frozen": True, "no_oos_fit": True,
            "status": "PASS" if pass_all else "FAIL",
        })

        print(f"\n  Validation: n_obs={n_obs}  NaN={nan_count}  Inf={inf_count}  "
              f"NegVar={neg_var_count}  ZeroVar={zero_var_count}  "
              f"VolSqrtVar={vol_sqrt_check}  -> {'PASS' if pass_all else 'FAIL'}")

        # Leakage test
        lt = run_leakage_test(asset, df_raw, last_is_row, asset_p_df)
        leakage_results[asset] = lt
        print(f"  Leakage test: {lt.get('overall', 'N/A')}")

        # Evaluation
        model_forecast_map = {
            "EWMA":      "ewma_forecast_variance",
            "GARCH":     "garch_forecast_variance",
            "EGARCH":    "egarch_forecast_variance",
            "GJR-GARCH": "gjr_garch_forecast_variance",
        }
        for target in ["realized_var_proxy", "parkinson_var_proxy"]:
            y = df_oos[target].values
            for model_name, col_name in model_forecast_map.items():
                f = df_oos[col_name].values
                metrics = compute_metrics(y, f)
                comparison_rows.append({
                    "asset": asset, "model": model_name, "target": target,
                    "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "QLIKE": metrics["QLIKE"],
                    "n_observations": n_obs,
                    "n_valid_qlike": metrics["n_valid_qlike"],
                    "n_zero_target": metrics["n_zero_target"],
                })

        # Plot
        plt.figure(figsize=(12, 6), dpi=300)
        plt.plot(df_oos["date"], df_oos["realized_var_proxy"],
                 label="Realized Var Proxy (r²)", color="black", alpha=0.35, linewidth=0.8)
        plt.plot(df_oos["date"], df_oos["ewma_forecast_variance"],
                 label="EWMA", color="#1f77b4", alpha=0.85, linewidth=1.2)
        plt.plot(df_oos["date"], df_oos["garch_forecast_variance"],
                 label="GARCH(1,1)", color="#ff7f0e", alpha=0.85, linewidth=1.2)
        plt.plot(df_oos["date"], df_oos["egarch_forecast_variance"],
                 label="EGARCH(1,1)", color="#2ca02c", alpha=0.85, linewidth=1.2)
        plt.plot(df_oos["date"], df_oos["gjr_garch_forecast_variance"],
                 label="GJR-GARCH(1,1)", color="#d62728", alpha=0.85, linewidth=1.2)
        plt.title(
            f"{asset.upper()} — OOS Daily Variance Forecasts vs Realized Variance Proxy (2022-2025)",
            fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Date", fontsize=11, labelpad=8)
        plt.ylabel("Conditional Variance", fontsize=11, labelpad=8)
        plt.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_OUTPUT_DIR, f"{asset}_oos_forecasts.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"  Plot saved: {plot_path}")

    # Save comparison CSV
    df_comparison = pd.DataFrame(comparison_rows)
    comp_csv_cols = ["asset", "model", "target", "MAE", "RMSE", "QLIKE", "n_observations"]
    df_comp_out = df_comparison[comp_csv_cols].copy()
    comp_csv_path = os.path.join(OOS_OUTPUT_DIR, "oos_model_comparison.csv")
    df_comp_out.to_csv(comp_csv_path, index=False)

    # Save ranking CSV
    ranking_rows = []
    for asset in ASSETS:
        for target in ["realized_var_proxy", "parkinson_var_proxy"]:
            sub = df_comparison[
                (df_comparison["asset"] == asset) & (df_comparison["target"] == target)
            ].copy()
            sub["rank_QLIKE"] = sub["QLIKE"].rank(ascending=True, method="min").astype(int)
            sub["rank_MAE"]   = sub["MAE"].rank(ascending=True, method="min").astype(int)
            sub["rank_RMSE"]  = sub["RMSE"].rank(ascending=True, method="min").astype(int)
            for _, row in sub.iterrows():
                ranking_rows.append({
                    "asset": row["asset"], "target": row["target"], "model": row["model"],
                    "rank_QLIKE": row["rank_QLIKE"], "rank_MAE": row["rank_MAE"],
                    "rank_RMSE": row["rank_RMSE"],
                    "QLIKE": row["QLIKE"], "MAE": row["MAE"], "RMSE": row["RMSE"],
                })
    df_ranking = pd.DataFrame(ranking_rows)
    ranking_csv_path = os.path.join(OOS_OUTPUT_DIR, "oos_model_ranking.csv")
    df_ranking.to_csv(ranking_csv_path, index=False)

    # -----------------------------------------------------------------------
    # FINAL DETAILED REPORT
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("FINAL OUT-OF-SAMPLE FORECASTING & EVALUATION REPORT")
    print("=" * 78)

    print("\n1. FILES MODIFIED (OOS outputs regenerated):")
    for f in oos_forecast_files:
        print(f"   [REGENERATED] {f}")
    print(f"   [REGENERATED] {comp_csv_path}")
    print(f"   [REGENERATED] {ranking_csv_path}")
    for asset in ASSETS:
        print(f"   [REGENERATED] {os.path.join(PLOTS_OUTPUT_DIR, asset + '_oos_forecasts.png')}")

    print("\n2. FILES UNTOUCHED (read-only, never written):")
    print("   data/*.csv  (all raw asset CSVs)")
    print("   data/static_model_outputs/model_parameters.csv")
    for asset in ASSETS:
        print(f"   data/static_model_outputs/{asset}_static_models.csv")

    print("\n3. PARAMETER SOURCE:")
    print(f"   {PARAM_FILE_PATH}")
    print("   Parameters frozen — 2015-01-01 to 2021-12-31. No OOS refitting.")

    print("\n4. STATIC ESTIMATION PERIOD: 2015-01-01 to 2021-12-31")
    print(f"5. OOS PERIOD:               {OOS_START_DATE} to {OOS_END_DATE}")
    print(f"6. NUMBER OF ASSETS:         {len(ASSETS)}")
    print(f"7. NUMBER OF MODELS:         {len(MODELS)} (EWMA, GARCH, EGARCH, GJR-GARCH)")

    print("\n8. OOS OBSERVATION COUNTS PER ASSET:")
    for vr in validation_reports:
        print(f"   {vr['asset'].upper():12s}: {vr['n_obs']} obs  "
              f"({vr['first_date']} to {vr['last_date']})")

    print("\n9. VALIDATION PASS/FAIL PER ASSET:")
    df_val = pd.DataFrame(validation_reports)
    cols_p = ["asset", "n_obs", "first_date", "last_date", "nan_count",
              "neg_var_count", "zero_var_count", "vol_eq_sqrt_var", "status"]
    print(df_val[cols_p].to_string(index=False))

    print("\n10. LOOK-AHEAD BIAS (LEAKAGE) TEST RESULTS:")
    hdr = (f"  {'Asset':12s}  {'Overall':8s}  "
           f"{'EWMA t0':8s}  {'EWMA t1':8s}  "
           f"{'GARCH t0':9s}  {'GARCH t1':9s}  "
           f"{'EGARCH t0':10s}  {'EGARCH t1':10s}  "
           f"{'GJR t0':8s}  {'GJR t1':8s}")
    print(hdr)
    for asset, lt in leakage_results.items():
        row_parts = [f"  {asset.upper():12s}  {lt.get('overall','N/A'):8s}"]
        for mname in ["EWMA", "GARCH", "EGARCH", "GJR-GARCH"]:
            md = lt.get(mname, {})
            row_parts.append(f"  {md.get('t0_unchanged','?'):8s}  {md.get('t1_changed','?'):8s}")
        print("".join(row_parts))

    print("\n11. FIRST 5 OOS OBSERVATIONS — GARCH (all assets, causal ordering):")
    for asset in ASSETS:
        df_a   = oos_dfs[asset]
        df_raw_a = pd.read_csv(os.path.join(INPUT_DATA_DIR, f"{asset}.csv"))
        df_raw_a["date"] = pd.to_datetime(df_raw_a["date"])
        df_raw_a = df_raw_a.sort_values("date").reset_index(drop=True)
        oos_mask_a = (
            (df_raw_a["date"] >= pd.to_datetime(OOS_START_DATE)) &
            (df_raw_a["date"] <= pd.to_datetime(OOS_END_DATE))
        ).values
        oos_idx_a = np.where(oos_mask_a)[0]
        rets_a = df_raw_a["log_return"].values
        print(f"\n  {asset.upper()}:")
        print(f"  {'Date':>12}  {'prev_return':>12}  {'fcast_var(GARCH)':>18}  "
              f"{'actual_return':>14}  {'actual_RV':>14}")
        for k in range(min(5, len(oos_idx_a))):
            t_idx = oos_idx_a[k]
            row   = df_a.iloc[k]
            print(f"  {row['date'].strftime('%Y-%m-%d'):>12}  "
                  f"{rets_a[t_idx-1]:>12.8f}  "
                  f"{row['garch_forecast_variance']:>18.8e}  "
                  f"{rets_a[t_idx]:>14.8f}  "
                  f"{row['realized_var_proxy']:>14.8e}")

    print("\n12. OOS MODEL COMPARISON (PRIMARY TARGET: realized_var_proxy):")
    df_rv = df_comparison[df_comparison["target"] == "realized_var_proxy"][comp_csv_cols]
    print(df_rv.to_string(index=False))

    print("\n    OOS MODEL COMPARISON (SECONDARY TARGET: parkinson_var_proxy):")
    df_pk = df_comparison[df_comparison["target"] == "parkinson_var_proxy"][comp_csv_cols]
    print(df_pk.to_string(index=False))

    print("\n13. OOS MODEL RANKING (mean rank across 8 assets — PRIMARY TARGET QLIKE):")
    mean_ranks = (
        df_ranking[df_ranking["target"] == "realized_var_proxy"]
        .groupby("model")[["rank_QLIKE", "rank_MAE", "rank_RMSE"]]
        .mean().reset_index().sort_values("rank_QLIKE").reset_index(drop=True)
    )
    mean_ranks.rename(columns={"rank_QLIKE": "Avg QLIKE Rank",
                                "rank_MAE":   "Avg MAE Rank",
                                "rank_RMSE":  "Avg RMSE Rank"}, inplace=True)
    print(mean_ranks.to_string(index=False))

    print("\n14. CONFIRMATION — MODEL PARAMETERS:")
    print("    [CONFIRMED] No model was refitted during the OOS period.")
    print("    [CONFIRMED] All parameters frozen at 2015-2021 in-sample estimates.")

    print("\n15. CONFIRMATION — DATA INTEGRITY:")
    print("    [CONFIRMED] No raw data file (data/*.csv) was modified.")
    print("    [CONFIRMED] No static model output file was modified.")

    all_val_pass  = all(vr["status"] == "PASS" for vr in validation_reports)
    all_leak_pass = all(lt.get("overall") == "PASS" for lt in leakage_results.values())

    print("\n" + "=" * 78)
    print("FINAL STATEMENT ON FORECAST CAUSALITY")
    print("=" * 78)
    if all_val_pass and all_leak_pass:
        print(
            "\n  *** CONFIRMED: This implementation produces TRUE ONE-STEP-AHEAD        ***\n"
            "  *** OUT-OF-SAMPLE FORECASTS with FROZEN 2015-2021 parameters and       ***\n"
            "  *** ZERO LOOK-AHEAD BIAS.                                               ***\n"
            "\n"
            "  For every asset and every OOS date t:\n"
            "    - forecast_variance[t] was computed using ONLY r[0..t-1] and the\n"
            "      warm-started conditional variance state from the in-sample filter.\n"
            "    - r[t] was observed only AFTER storing forecast_variance[t].\n"
            "    - The leakage test confirms: perturbing r[t] does NOT change\n"
            "      forecast[t] but DOES change forecast[t+1] for all models and\n"
            "      all assets."
        )
    else:
        print(
            "\n  *** WARNING: Validation or leakage test did NOT fully pass. ***\n"
            f"  Validation overall PASS: {all_val_pass}\n"
            f"  Leakage test overall PASS: {all_leak_pass}\n"
            "  Review individual results above."
        )

    print("\n" + "=" * 78)
    print("STAGE COMPLETE: ALL OOS FORECASTS, VALIDATION & EVALUATION FINISHED")
    print("=" * 78)


if __name__ == "__main__":
    main()
