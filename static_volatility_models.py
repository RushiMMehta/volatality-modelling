import os
import sys
import warnings
import numpy as np
import pandas as pd

# Suppress minor non-critical convergence warnings from arch/scipy during output printing
warnings.filterwarnings("ignore")

# Define assets to process
ASSETS = [
    "nifty50",
    "banknifty",
    "adanient",
    "tatasteel",
    "dlf",
    "hindunilvr",
    "nestleind",
    "sunpharma"
]

# Fitting window bounds
START_DATE = "2015-01-01"
END_DATE = "2021-12-31"

OUTPUT_DIR = os.path.join("data", "static_model_outputs")


def load_asset_data(asset_name: str) -> pd.DataFrame:
    """
    Loads data for a given asset from data/<asset_name>.csv,
    parses dates, filters to 2015-01-01 -> 2021-12-31, and sorts chronologically.
    Does NOT alter, fill missing dates, or modify original files.
    """
    filepath = os.path.join("data", f"{asset_name}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Filter for in-sample static window 2015-01-01 to 2021-12-31
    mask = (df["date"] >= pd.to_datetime(START_DATE)) & (df["date"] <= pd.to_datetime(END_DATE))
    df_window = df.loc[mask].copy().reset_index(drop=True)

    if df_window.empty:
        raise ValueError(f"No data available for {asset_name} in window {START_DATE} to {END_DATE}")

    return df_window


def fit_ewma(returns: np.ndarray, lam: float = 0.94) -> tuple:
    """
    Implements EWMA conditional variance series:
    sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2

    Parameters:
    - returns: array of log returns (decimal scale)
    - lam: decay factor (default 0.94)

    Returns:
    - variance: array of in-sample conditional variance estimates (decimal scale)
    - volatility: array of conditional volatilities (sqrt(variance))
    """
    T = len(returns)
    variance = np.zeros(T)
    # Initialize with sample variance of the in-sample window
    variance[0] = np.var(returns, ddof=1) if T > 1 else returns[0] ** 2

    for t in range(1, T):
        variance[t] = lam * variance[t - 1] + (1 - lam) * (returns[t - 1] ** 2)

    volatility = np.sqrt(variance)
    return variance, volatility


def fit_garch(returns_decimal: np.ndarray) -> dict:
    """
    Fits standard GARCH(1,1) model using `arch` package.
    Returns scaled by 100 for numerical stability during optimization.
    Fitted variance and parameters converted back to original decimal scale.
    """
    from arch import arch_model

    returns_scaled = returns_decimal * 100.0
    am = arch_model(returns_scaled, vol="Garch", p=1, q=1, mean="Zero", dist="normal")

    try:
        res = am.fit(disp="off", show_warning=False)
        converged = (res.convergence_flag == 0)

        # Scale variance back to decimal-return scale
        var_scaled = res.conditional_volatility ** 2
        var_decimal = var_scaled / 10000.0
        vol_decimal = np.sqrt(var_decimal)

        params = res.params
        omega_raw = params.get("omega", np.nan)
        alpha_raw = params.get("alpha[1]", np.nan)
        beta_raw = params.get("beta[1]", np.nan)

        # Rescale omega to decimal scale
        omega_dec = omega_raw / 10000.0 if not np.isnan(omega_raw) else np.nan

        # Persistence diagnostic for GARCH(1,1)
        persistence = alpha_raw + beta_raw if (not np.isnan(alpha_raw) and not np.isnan(beta_raw)) else np.nan

        # Adjust log likelihood for return scaling by 100: logL(r) = logL(r*100) + T * ln(100)
        T = len(returns_decimal)
        log_lik = res.loglikelihood + T * np.log(100.0)

        return {
            "model_name": "GARCH",
            "variance": var_decimal.values if hasattr(var_decimal, "values") else np.array(var_decimal),
            "volatility": vol_decimal.values if hasattr(vol_decimal, "values") else np.array(vol_decimal),
            "omega": omega_dec,
            "alpha": alpha_raw,
            "gamma": np.nan,
            "beta": beta_raw,
            "persistence": persistence,
            "log_likelihood": log_lik,
            "aic": res.aic,
            "bic": res.bic,
            "converged": converged,
            "status": "PASS" if converged else "WARNING: Optimization did not converge"
        }
    except Exception as e:
        return {
            "model_name": "GARCH",
            "variance": np.full(len(returns_decimal), np.nan),
            "volatility": np.full(len(returns_decimal), np.nan),
            "omega": np.nan, "alpha": np.nan, "gamma": np.nan, "beta": np.nan,
            "persistence": np.nan, "log_likelihood": np.nan, "aic": np.nan, "bic": np.nan,
            "converged": False,
            "status": f"FAIL: {str(e)}"
        }


def fit_egarch(returns_decimal: np.ndarray) -> dict:
    """
    Fits EGARCH(1,1) model using `arch` package.
    Models log variance: ln(sigma_t^2) = omega + alpha*(|e_{t-1}| - E[|e|]) + gamma*e_{t-1} + beta*ln(sigma_{t-1}^2)
    Note: Standard alpha + beta stationarity does not apply mechanically to EGARCH.
    Beta is reported as the primary persistence parameter.
    """
    from arch import arch_model

    returns_scaled = returns_decimal * 100.0
    am = arch_model(returns_scaled, vol="EGARCH", p=1, o=1, q=1, mean="Zero", dist="normal")

    try:
        res = am.fit(disp="off", show_warning=False)
        converged = (res.convergence_flag == 0)

        var_scaled = res.conditional_volatility ** 2
        var_decimal = var_scaled / 10000.0
        vol_decimal = np.sqrt(var_decimal)

        params = res.params
        omega_raw = params.get("omega", np.nan)
        alpha_raw = params.get("alpha[1]", np.nan)
        gamma_raw = params.get("gamma[1]", np.nan)
        beta_raw = params.get("beta[1]", np.nan)

        # EGARCH models log variance in % scale: ln(sigma^2_scaled) = ln(10000 * sigma^2_decimal)
        # = ln(10000) + ln(sigma^2_decimal)
        # Thus omega_decimal = omega_scaled - ln(10000)
        omega_dec = omega_raw - np.log(10000.0) if not np.isnan(omega_raw) else np.nan

        # For EGARCH, beta is the persistence parameter
        persistence = beta_raw

        T = len(returns_decimal)
        log_lik = res.loglikelihood + T * np.log(100.0)

        return {
            "model_name": "EGARCH",
            "variance": var_decimal.values if hasattr(var_decimal, "values") else np.array(var_decimal),
            "volatility": vol_decimal.values if hasattr(vol_decimal, "values") else np.array(vol_decimal),
            "omega": omega_dec,
            "alpha": alpha_raw,
            "gamma": gamma_raw,
            "beta": beta_raw,
            "persistence": persistence,
            "log_likelihood": log_lik,
            "aic": res.aic,
            "bic": res.bic,
            "converged": converged,
            "status": "PASS" if converged else "WARNING: Optimization did not converge"
        }
    except Exception as e:
        return {
            "model_name": "EGARCH",
            "variance": np.full(len(returns_decimal), np.nan),
            "volatility": np.full(len(returns_decimal), np.nan),
            "omega": np.nan, "alpha": np.nan, "gamma": np.nan, "beta": np.nan,
            "persistence": np.nan, "log_likelihood": np.nan, "aic": np.nan, "bic": np.nan,
            "converged": False,
            "status": f"FAIL: {str(e)}"
        }


def fit_gjr_garch(returns_decimal: np.ndarray) -> dict:
    """
    Fits GJR-GARCH(1,1) model using `arch` package.
    vol='GARCH', p=1, o=1, q=1
    sigma_t^2 = omega + alpha * r_{t-1}^2 + gamma * I_{t-1} * r_{t-1}^2 + beta * sigma_{t-1}^2
    Persistence approximation: persistence = alpha + beta + 0.5 * gamma
    """
    from arch import arch_model

    returns_scaled = returns_decimal * 100.0
    am = arch_model(returns_scaled, vol="GARCH", p=1, o=1, q=1, mean="Zero", dist="normal")

    try:
        res = am.fit(disp="off", show_warning=False)
        converged = (res.convergence_flag == 0)

        var_scaled = res.conditional_volatility ** 2
        var_decimal = var_scaled / 10000.0
        vol_decimal = np.sqrt(var_decimal)

        params = res.params
        omega_raw = params.get("omega", np.nan)
        alpha_raw = params.get("alpha[1]", np.nan)
        gamma_raw = params.get("gamma[1]", np.nan)
        beta_raw = params.get("beta[1]", np.nan)

        omega_dec = omega_raw / 10000.0 if not np.isnan(omega_raw) else np.nan

        # Persistence diagnostic for GJR-GARCH: alpha + beta + 0.5 * gamma
        if not np.isnan(alpha_raw) and not np.isnan(beta_raw) and not np.isnan(gamma_raw):
            persistence = alpha_raw + beta_raw + 0.5 * gamma_raw
        else:
            persistence = np.nan

        T = len(returns_decimal)
        log_lik = res.loglikelihood + T * np.log(100.0)

        return {
            "model_name": "GJR-GARCH",
            "variance": var_decimal.values if hasattr(var_decimal, "values") else np.array(var_decimal),
            "volatility": vol_decimal.values if hasattr(vol_decimal, "values") else np.array(vol_decimal),
            "omega": omega_dec,
            "alpha": alpha_raw,
            "gamma": gamma_raw,
            "beta": beta_raw,
            "persistence": persistence,
            "log_likelihood": log_lik,
            "aic": res.aic,
            "bic": res.bic,
            "converged": converged,
            "status": "PASS" if converged else "WARNING: Optimization did not converge"
        }
    except Exception as e:
        return {
            "model_name": "GJR-GARCH",
            "variance": np.full(len(returns_decimal), np.nan),
            "volatility": np.full(len(returns_decimal), np.nan),
            "omega": np.nan, "alpha": np.nan, "gamma": np.nan, "beta": np.nan,
            "persistence": np.nan, "log_likelihood": np.nan, "aic": np.nan, "bic": np.nan,
            "converged": False,
            "status": f"FAIL: {str(e)}"
        }


def print_diagnostics(asset_name: str, model_res: dict, num_obs: int, start_d: str, end_d: str):
    """
    Prints comprehensive parameter diagnostics and warnings for an asset x model fit.
    """
    m_name = model_res["model_name"]
    print(f"\n==================================================")
    print(f"ASSET: {asset_name.upper()} | MODEL: {m_name}")
    print(f"==================================================")
    print(f"Number of observations : {num_obs}")
    print(f"Fitting date range     : {start_d} to {end_d}")
    print(f"Convergence status     : {'CONVERGED' if model_res['converged'] else 'FAILED/NOT CONVERGED'}")

    if m_name == "EWMA":
        print(f"Lambda parameter       : {model_res['lambda']:.4f}")
        return

    print("Estimated Parameters:")
    if not np.isnan(model_res['omega']):
        print(f"  omega : {model_res['omega']:.8e}")
    if not np.isnan(model_res['alpha']):
        print(f"  alpha : {model_res['alpha']:.6f}")
    if not np.isnan(model_res['gamma']):
        print(f"  gamma : {model_res['gamma']:.6f}")
    if not np.isnan(model_res['beta']):
        print(f"  beta  : {model_res['beta']:.6f}")

    print("Model Selection Criteria:")
    print(f"  Log Likelihood : {model_res['log_likelihood']:.4f}")
    print(f"  AIC            : {model_res['aic']:.4f}")
    print(f"  BIC            : {model_res['bic']:.4f}")

    print("Persistence Diagnostic:")
    pers = model_res['persistence']
    if m_name == "GARCH":
        print(f"  alpha + beta = {pers:.6f}")
        if pers >= 1.0:
            print("  WARNING: Standard GARCH stationarity condition violated (alpha + beta >= 1).")
        elif pers > 0.98:
            print("  DIAGNOSTIC: VERY HIGH VOLATILITY PERSISTENCE (volatility shocks decay slowly).")
        else:
            print("  DIAGNOSTIC: Moderate persistence (volatility mean-reverts quickly).")

        # Parameter boundary sanity checks
        if model_res['omega'] <= 0:
            print("  WARNING: omega <= 0")
        if model_res['alpha'] < 0:
            print("  WARNING: alpha < 0")
        if model_res['beta'] < 0:
            print("  WARNING: beta < 0")

    elif m_name == "GJR-GARCH":
        print(f"  alpha + beta + 0.5*gamma = {pers:.6f}")
        if pers >= 1.0:
            print("  DIAGNOSTIC: VERY HIGH VOLATILITY PERSISTENCE (persistence >= 1).")
        else:
            print("  DIAGNOSTIC: Covariance-stationary GJR-GARCH process.")

        g = model_res['gamma']
        if g > 0:
            print(f"  LEVERAGE DIAGNOSTIC: gamma ({g:.6f}) > 0 -> Negative shocks increase volatility more than positive shocks.")
        elif abs(g) < 1e-4:
            print(f"  LEVERAGE DIAGNOSTIC: gamma ({g:.6f}) approx 0 -> Minimal asymmetric leverage effect.")

    elif m_name == "EGARCH":
        print(f"  beta (log-variance persistence) = {pers:.6f}")
        print("  NOTE: EGARCH models log-variance. Ordinary GARCH alpha+beta stationarity does not apply.")
        g = model_res['gamma']
        if g != 0 and not np.isnan(g):
            print(f"  LEVERAGE DIAGNOSTIC: gamma = {g:.6f} captures asymmetric response to shocks.")

    # Variance sanity check
    var_series = model_res['variance']
    if np.any(np.isnan(var_series)) or np.any(np.isinf(var_series)):
        print("  WARNING: NaN or infinite fitted variance detected!")
    elif np.any(var_series <= 0):
        print("  WARNING: Non-positive fitted variance detected!")
    else:
        print("  VARIANCE INTEGRITY: All fitted conditional variances are strictly positive and finite.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_results = []
    param_table_rows = []
    output_files_created = []

    print("==========================================================")
    print("STARTING STATIC VOLATILITY MODEL FITTING (2015-2021)")
    print("==========================================================")

    for asset in ASSETS:
        print(f"\nProcessing Asset: {asset.upper()}...")
        try:
            df_asset = load_asset_data(asset)
        except Exception as e:
            print(f"ERROR loading data for {asset}: {e}")
            summary_results.append({
                "Asset": asset.upper(), "EWMA": "FAIL", "GARCH": "FAIL", "EGARCH": "FAIL", "GJR-GARCH": "FAIL"
            })
            continue

        num_obs = len(df_asset)
        start_d = df_asset["date"].min().strftime("%Y-%m-%d")
        end_d = df_asset["date"].max().strftime("%Y-%m-%d")
        returns = df_asset["log_return"].values

        # 1. EWMA (lambda = 0.94)
        ewma_var, ewma_vol = fit_ewma(returns, lam=0.94)
        ewma_res = {
            "model_name": "EWMA", "variance": ewma_var, "volatility": ewma_vol,
            "omega": np.nan, "alpha": np.nan, "gamma": np.nan, "beta": np.nan,
            "lambda": 0.94, "persistence": np.nan, "log_likelihood": np.nan,
            "aic": np.nan, "bic": np.nan, "converged": True, "status": "PASS"
        }
        print_diagnostics(asset, ewma_res, num_obs, start_d, end_d)

        # 2. GARCH(1,1)
        garch_res = fit_garch(returns)
        print_diagnostics(asset, garch_res, num_obs, start_d, end_d)

        # 3. EGARCH(1,1)
        egarch_res = fit_egarch(returns)
        print_diagnostics(asset, egarch_res, num_obs, start_d, end_d)

        # 4. GJR-GARCH(1,1)
        gjr_res = fit_gjr_garch(returns)
        print_diagnostics(asset, gjr_res, num_obs, start_d, end_d)

        # Create output DataFrame for asset
        df_out = pd.DataFrame({
            "date": df_asset["date"].dt.strftime("%Y-%m-%d"),
            "log_return": df_asset["log_return"],
            "realized_var_proxy": df_asset["realized_var_proxy"],
            "parkinson_var_proxy": df_asset["parkinson_var_proxy"],
            "ewma_variance": ewma_res["variance"],
            "ewma_volatility": ewma_res["volatility"],
            "garch_variance": garch_res["variance"],
            "garch_volatility": garch_res["volatility"],
            "egarch_variance": egarch_res["variance"],
            "egarch_volatility": egarch_res["volatility"],
            "gjr_garch_variance": gjr_res["variance"],
            "gjr_garch_volatility": gjr_res["volatility"]
        })

        out_csv_path = os.path.join(OUTPUT_DIR, f"{asset}_static_models.csv")
        df_out.to_csv(out_csv_path, index=False)
        output_files_created.append(out_csv_path)

        # Record parameter summary rows
        for res in [ewma_res, garch_res, egarch_res, gjr_res]:
            param_table_rows.append({
                "asset": asset,
                "model": res["model_name"],
                "omega": res["omega"],
                "alpha": res["alpha"],
                "gamma": res["gamma"],
                "beta": res["beta"],
                "lambda": res.get("lambda", np.nan),
                "persistence": res["persistence"],
                "log_likelihood": res["log_likelihood"],
                "AIC": res["aic"],
                "BIC": res["bic"],
                "converged": res["converged"]
            })

        summary_results.append({
            "Asset": asset.upper(),
            "EWMA": "PASS" if ewma_res["converged"] else "FAIL",
            "GARCH": "PASS" if garch_res["converged"] else "FAIL",
            "EGARCH": "PASS" if egarch_res["converged"] else "FAIL",
            "GJR-GARCH": "PASS" if gjr_res["converged"] else "FAIL",
        })

    # Save summary parameter CSV
    df_params = pd.DataFrame(param_table_rows)
    params_csv_path = os.path.join(OUTPUT_DIR, "model_parameters.csv")
    df_params.to_csv(params_csv_path, index=False)
    output_files_created.append(params_csv_path)

    # Print Summary Table
    print("\n" + "=" * 60)
    print("FINAL MODEL FITTING SUMMARY TABLE")
    print("=" * 60)
    df_summary = pd.DataFrame(summary_results)
    print(df_summary.to_string(index=False))

    total_fits = len(summary_results) * 4
    successful_fits = sum(
        (1 if row["EWMA"] == "PASS" else 0) +
        (1 if row["GARCH"] == "PASS" else 0) +
        (1 if row["EGARCH"] == "PASS" else 0) +
        (1 if row["GJR-GARCH"] == "PASS" else 0)
        for row in summary_results
    )
    failed_fits = total_fits - successful_fits

    print("\n----------------------------------------------------------")
    print(f"Total Model Fits Executed : {total_fits}")
    print(f"Successful Fits           : {successful_fits}")
    print(f"Failed Fits               : {failed_fits}")
    print("----------------------------------------------------------")

    print("\nOutput CSV Files Created:")
    all_exist = True
    for fpath in output_files_created:
        exists = os.path.exists(fpath)
        status = "EXISTS" if exists else "MISSING"
        if not exists:
            all_exist = False
        print(f"  [{status}] {fpath}")

    if all_exist and len(output_files_created) == 9:
        print("\nSUCCESS: All 8 asset CSVs and model_parameters.csv exist!")
    else:
        print("\nWARNING: Some output files are missing!")


if __name__ == "__main__":
    main()
