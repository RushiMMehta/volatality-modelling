"""
Final Data Quality Checks
Adaptive Institutional Volatility Forecasting Engine

IMPORTANT:
- This script DOES NOT download data.
- This script DOES NOT modify the existing CSV files.
- This script DOES NOT add/remove columns.
- This script DOES NOT remove observations.
- This script only reads the existing datasets and performs validation checks.

Existing files expected inside ./data:

    nifty50.csv
    banknifty.csv
    adanient.csv
    tatasteel.csv
    dlf.csv
    hindunilvr.csv
    nestleind.csv
    sunpharma.csv
    india_vix.csv
    pooled_dataset.csv

Run:

    python final_data_checks.py
"""


import os
import numpy as np
import pandas as pd


# ============================================================
# 1. Configuration
# ============================================================

DATA_DIR = "data"

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

POOLED_FILE = os.path.join(DATA_DIR, "pooled_dataset.csv")
VIX_FILE = os.path.join(DATA_DIR, "india_vix.csv")


# ============================================================
# 2. Load existing CSV
# ============================================================

def load_csv(filename):
    """
    Load an existing CSV without modifying it.
    """

    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        print(f"WARNING: File not found -> {path}")
        return None

    df = pd.read_csv(path)

    # Convert date column if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


# ============================================================
# 3. Basic checks for individual asset files
# ============================================================

def check_asset_file(asset_name, df):
    """
    Perform checks on one asset's existing CSV.
    """

    print("\n")
    print("=" * 75)
    print(f"CHECKING: {asset_name.upper()}")
    print("=" * 75)

    if df is None:
        return

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print("\nDataset shape:")
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    print("\nColumns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Date coverage
    # --------------------------------------------------------

    if "date" in df.columns:

        print("\nDate coverage:")

        print(
            f"First date : {df['date'].min().date()}"
        )

        print(
            f"Last date  : {df['date'].max().date()}"
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values:")

    missing = df.isna().sum()

    print(missing[missing > 0])

    total_missing = missing.sum()

    if total_missing == 0:
        print("✓ PASS: No missing values.")
    else:
        print(
            f"⚠ WARNING: {total_missing} missing values."
        )

    # --------------------------------------------------------
    # Duplicate dates
    # --------------------------------------------------------

    if "date" in df.columns:

        duplicate_dates = df.duplicated(
            subset=["date"]
        ).sum()

        print(
            f"\nDuplicate dates: {duplicate_dates}"
        )

        if duplicate_dates == 0:
            print("✓ PASS: No duplicate dates.")
        else:
            print("⚠ WARNING: Duplicate dates found.")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "log_return",
        "realized_var_proxy",
        "parkinson_var_proxy",
    ]

    print("\nRequired column check:")

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if len(missing_columns) == 0:

        print("✓ PASS: All expected columns present.")

    else:

        print(
            "⚠ WARNING: Missing columns:"
        )

        print(missing_columns)

    # --------------------------------------------------------
    # OHLC consistency
    # --------------------------------------------------------

    required_ohlc = [
        "open",
        "high",
        "low",
        "close",
    ]

    if all(col in df.columns for col in required_ohlc):

        # Positive prices
        non_positive = df[
            (df["open"] <= 0) |
            (df["high"] <= 0) |
            (df["low"] <= 0) |
            (df["close"] <= 0)
        ]

        # High must be >= Open, Low, Close
        invalid_high = df[
            (df["high"] < df["open"]) |
            (df["high"] < df["low"]) |
            (df["high"] < df["close"])
        ]

        # Low must be <= Open, High, Close
        invalid_low = df[
            (df["low"] > df["open"]) |
            (df["low"] > df["high"]) |
            (df["low"] > df["close"])
        ]

        print("\nOHLC consistency:")

        print(
            f"Non-positive prices : {len(non_positive)}"
        )

        print(
            f"Invalid HIGH values : {len(invalid_high)}"
        )

        print(
            f"Invalid LOW values  : {len(invalid_low)}"
        )

        if (
            len(non_positive) == 0
            and len(invalid_high) == 0
            and len(invalid_low) == 0
        ):

            print(
                "✓ PASS: OHLC data is internally consistent."
            )

        else:

            print(
                "⚠ WARNING: Potentially corrupted OHLC observations."
            )

    # --------------------------------------------------------
    # Log return sanity
    # --------------------------------------------------------

    if "log_return" in df.columns:

        invalid_returns = df[
            ~np.isfinite(df["log_return"])
        ]

        print("\nLog-return sanity:")

        print(
            f"Invalid/infinite returns: "
            f"{len(invalid_returns)}"
        )

        if len(invalid_returns) == 0:

            print(
                "✓ PASS: All log returns are finite."
            )

        else:

            print(
                "⚠ WARNING: Invalid log returns found."
            )

    # --------------------------------------------------------
    # Variance sanity
    # --------------------------------------------------------

    variance_columns = [
        "realized_var_proxy",
        "parkinson_var_proxy",
    ]

    print("\nVariance proxy sanity:")

    for column in variance_columns:

        if column not in df.columns:
            print(
                f"{column}: COLUMN NOT FOUND"
            )
            continue

        negative_values = (
            df[column] < 0
        ).sum()

        infinite_values = (
            ~np.isfinite(df[column])
        ).sum()

        print(
            f"{column}: "
            f"negative={negative_values}, "
            f"non-finite={infinite_values}"
        )

        if (
            negative_values == 0
            and infinite_values == 0
        ):

            print(
                f"✓ PASS: {column}"
            )

        else:

            print(
                f"⚠ WARNING: Problem with {column}"
            )

    # --------------------------------------------------------
    # Extreme observations
    # --------------------------------------------------------

    if "log_return" in df.columns:

        extreme = df[
            df["log_return"].abs() > 0.10
        ]

        print("\nExtreme observations:")

        print(
            f"Number of |return| > 10%: "
            f"{len(extreme)}"
        )

        if len(extreme) > 0:

            print(
                "\nExtreme dates and returns:"
            )

            print(
                extreme[
                    ["date", "log_return"]
                ].to_string(index=False)
            )

            print(
                "\nNOTE:"
            )

            print(
                "These observations are NOT being removed."
            )

            print(
                "They should be investigated as genuine "
                "market events/corporate actions."
            )

        else:

            print(
                "No extreme observations found."
            )


# ============================================================
# 4. Pooled dataset checks
# ============================================================

def check_pooled_dataset(df):
    """
    Perform final checks on pooled_dataset.csv.
    """

    print("\n\n")
    print("#" * 75)
    print("POOLED DATASET CHECK")
    print("#" * 75)

    if df is None:

        print(
            "ERROR: pooled_dataset.csv not found."
        )

        return

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print("\nDataset shape:")

    print(
        f"Rows    : {len(df)}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    print("\nColumns:")

    print(
        list(df.columns)
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values:")

    missing = df.isna().sum()

    if missing.sum() == 0:

        print(
            "✓ PASS: No missing values."
        )

    else:

        print(
            missing[missing > 0]
        )

        print(
            f"⚠ WARNING: "
            f"{missing.sum()} missing values."
        )

    # --------------------------------------------------------
    # Duplicate asset/date combinations
    # --------------------------------------------------------

    print("\nDuplicate asset-date combinations:")

    if (
        "asset" in df.columns
        and "date" in df.columns
    ):

        duplicates = df.duplicated(
            subset=["asset", "date"]
        )

        duplicate_count = duplicates.sum()

        print(
            f"Duplicate combinations: "
            f"{duplicate_count}"
        )

        if duplicate_count == 0:

            print(
                "✓ PASS: No duplicate asset-date combinations."
            )

        else:

            print(
                "⚠ WARNING: Duplicate asset-date combinations found."
            )

            print(
                df[
                    duplicates
                ][
                    ["asset", "date"]
                ].to_string(index=False)
            )

    # --------------------------------------------------------
    # Asset coverage
    # --------------------------------------------------------

    print("\nAsset coverage:")

    if (
        "asset" in df.columns
        and "date" in df.columns
    ):

        coverage = (
            df.groupby("asset")["date"]
            .agg(
                first_date="min",
                last_date="max",
                observations="count"
            )
            .sort_index()
        )

        print(
            coverage
        )

        print(
            f"\nNumber of assets: "
            f"{df['asset'].nunique()}"
        )

    # --------------------------------------------------------
    # Category coverage
    # --------------------------------------------------------

    if "category" in df.columns:

        print("\nCategory coverage:")

        print(
            df.groupby(
                ["category", "asset"]
            ).size()
        )

    # --------------------------------------------------------
    # Extreme observations
    # --------------------------------------------------------

    if "log_return" in df.columns:

        print("\nExtreme observations by asset:")

        extreme_counts = (
            df.assign(
                extreme=(
                    df["log_return"].abs() > 0.10
                )
            )
            .groupby("asset")["extreme"]
            .sum()
        )

        print(
            extreme_counts
        )

    # --------------------------------------------------------
    # OHLC corruption
    # --------------------------------------------------------

    required_ohlc = [
        "open",
        "high",
        "low",
        "close",
    ]

    if all(
        column in df.columns
        for column in required_ohlc
    ):

        invalid_ohlc = df[
            (df["open"] <= 0) |
            (df["high"] <= 0) |
            (df["low"] <= 0) |
            (df["close"] <= 0) |

            (df["high"] < df["open"]) |
            (df["high"] < df["low"]) |
            (df["high"] < df["close"]) |

            (df["low"] > df["open"]) |
            (df["low"] > df["high"]) |
            (df["low"] > df["close"])
        ]

        print("\nObvious OHLC corruption:")

        print(
            f"Potentially corrupted rows: "
            f"{len(invalid_ohlc)}"
        )

        if len(invalid_ohlc) == 0:

            print(
                "✓ PASS: No obvious OHLC corruption."
            )

        else:

            print(
                "⚠ WARNING: Potentially corrupted observations:"
            )

            print(
                invalid_ohlc[
                    [
                        "asset",
                        "date",
                        "open",
                        "high",
                        "low",
                        "close"
                    ]
                ].to_string(index=False)
            )


# ============================================================
# 5. Main
# ============================================================

def main():

    print("\n")
    print("=" * 75)
    print("FINAL DATA VALIDATION")
    print("=" * 75)

    print(
        "\nIMPORTANT:"
    )

    print(
        "This script only READS your existing CSV files."
    )

    print(
        "No files will be modified."
    )

    print(
        "No observations will be removed."
    )

    print(
        "No columns will be added."
    )

    print(
        "No data will be downloaded."
    )

    # --------------------------------------------------------
    # Check individual asset files
    # --------------------------------------------------------

    for asset in ASSETS:

        filename = f"{asset}.csv"

        df = load_csv(filename)

        check_asset_file(
            asset,
            df
        )

    # --------------------------------------------------------
    # Check India VIX
    # --------------------------------------------------------

    print("\n\n")
    print("=" * 75)
    print("INDIA VIX CHECK")
    print("=" * 75)

    vix_df = load_csv("india_vix.csv")

    if vix_df is not None:

        print(
            f"Rows: {len(vix_df)}"
        )

        print(
            f"Columns: {list(vix_df.columns)}"
        )

        if "date" in vix_df.columns:

            print(
                f"Date range: "
                f"{vix_df['date'].min().date()} "
                f"-> "
                f"{vix_df['date'].max().date()}"
            )

        print("\nMissing values:")

        print(
            vix_df.isna().sum()
        )

    # --------------------------------------------------------
    # Check pooled dataset
    # --------------------------------------------------------

    pooled_df = load_csv(
        "pooled_dataset.csv"
    )

    check_pooled_dataset(
        pooled_df
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n\n")
    print("=" * 75)
    print("VALIDATION COMPLETE")
    print("=" * 75)

    print(
        "\nYour original CSV files were NOT modified."
    )

    print(
        "If all checks pass, you can freeze the dataset "
        "and move to the modeling stage."
    )

    print(
        "\nNext stage:"
    )

    print(
        "Feature Engineering → "
        "EWMA → GARCH → GJR-GARCH → EGARCH"
    )


if __name__ == "__main__":
    main()