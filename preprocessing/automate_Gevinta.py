"""
automate_Gevinta.py
====================
Task 2 - Script preprocessing otomatis untuk dataset RiskTrace.

Fungsi-fungsi yang tersedia:
  - load_data(path)        : Load CSV dataset
  - validate_data(df)      : Validasi missing values dan tipe data
  - preprocess(df)         : Encode categorical, scale numerical, handle imbalance (SMOTE)
  - save_output(df, path)  : Simpan hasil preprocessing ke CSV
  - main()                 : Jalankan seluruh pipeline, return df siap train

Cara menjalankan:
    python automate_Gevinta.py

Output:
    risktrace_preprocessing.csv
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ─── Konfigurasi ────────────────────────────────────────────────────────────────
INPUT_PATH  = os.path.join(os.path.dirname(__file__), "..", "risktrace_raw.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "risktrace_preprocessing.csv")

CATEGORICAL_COLS = ["profile", "channel"]
DROP_COLS        = ["account_id", "is_gambling"]   # kolom non-fitur
TARGET_COL       = "is_suspicious"

NUMERICAL_COLS = [
    "step", "day", "hour_of_day", "is_night",
    "night_ratio_7d", "night_ratio_14d", "temporal_shift",
    "amount", "amount_log", "amount_vs_avg_7d", "total_amount_7d",
    "tx_count_24h", "tx_count_7d", "burst_score",
    "unique_recv_7d", "unique_recv_24h", "qris_ratio_7d",
    "drain_cycle_flag", "dormant_flag", "round_amount_flag",
]
# ────────────────────────────────────────────────────────────────────────────────


def load_data(path: str) -> pd.DataFrame:
    """
    Memuat dataset CSV dari path yang diberikan.

    Parameters
    ----------
    path : str
        Path ke file CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame hasil load.
    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"[ERROR] File tidak ditemukan: {abs_path}")

    df = pd.read_csv(abs_path)
    print(f"[load_data] Dataset dimuat: {abs_path}")
    print(f"            Shape: {df.shape}")
    print(f"            Kolom: {df.columns.tolist()}")
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Memvalidasi integritas data: cek missing values, duplikat, dan tipe data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame yang akan divalidasi.

    Returns
    -------
    pd.DataFrame
        DataFrame setelah validasi dan cleaning dasar.
    """
    print("\n[validate_data] Memulai validasi data...")

    # --- Missing values ---
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) == 0:
        print("  [OK] Tidak ada missing values.")
    else:
        print(f"  [!] Missing values ditemukan:\n{missing_cols}")
        for col in missing_cols.index:
            if df[col].dtype == "object":
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        print("  [FIX] Missing values diimputasi (mode untuk kategorik, median untuk numerik).")

    # --- Duplikat ---
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f"  [FIX] {dup_count} baris duplikat dihapus.")
    else:
        print("  [OK] Tidak ada duplikat.")

    # --- Tipe data ---
    print(f"\n  Tipe data per kolom:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<30} {str(dtype)}")

    # --- Distribusi target ---
    print(f"\n  Distribusi target '{TARGET_COL}':")
    print(df[TARGET_COL].value_counts().to_string())

    print(f"\n[validate_data] Selesai. Shape setelah validasi: {df.shape}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan preprocessing lengkap:
      1. Drop kolom non-fitur
      2. Label Encoding untuk kolom kategorik
      3. Standard Scaling untuk kolom numerik
      4. SMOTE untuk menangani class imbalance

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame yang sudah divalidasi.

    Returns
    -------
    pd.DataFrame
        DataFrame siap training (sudah balanced).
    """
    print("\n[preprocess] Memulai preprocessing...")

    # 1. Drop kolom non-fitur
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  [1] Kolom dihapus: {cols_to_drop}")

    # 2. Label Encoding untuk kolom kategorik
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"  [2] Label Encoded: {col}")

    # 3. Pisahkan fitur dan target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # 4. Standard Scaling untuk kolom numerik yang ada
    num_cols = [c for c in NUMERICAL_COLS if c in X.columns]
    scaler   = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    print(f"  [3] StandardScaler diterapkan pada {len(num_cols)} kolom numerik.")

    # 5. Gabungkan kembali (SMOTE dipindahkan ke tahap modelling)
    df_processed = X.copy()
    df_processed[TARGET_COL] = y

    print(f"\n[preprocess] Selesai. Shape akhir: {df_processed.shape}")
    return df_processed


def save_output(df: pd.DataFrame, path: str) -> None:
    """
    Menyimpan DataFrame hasil preprocessing ke file CSV.

    Parameters
    ----------
    df   : pd.DataFrame
        DataFrame yang akan disimpan.
    path : str
        Path output file CSV.
    """
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    df.to_csv(abs_path, index=False)
    print(f"\n[save_output] File disimpan: {abs_path}")
    print(f"              Shape: {df.shape}")
    print(f"              Kolom: {df.columns.tolist()}")


def main() -> pd.DataFrame:
    """
    Menjalankan seluruh pipeline preprocessing:
      load → validate → preprocess → save

    Returns
    -------
    pd.DataFrame
        DataFrame siap training.
    """
    print("=" * 60)
    print("  RiskTrace — Automated Preprocessing Pipeline")
    print("=" * 60)

    df = load_data(INPUT_PATH)
    df = validate_data(df)
    df = preprocess(df)
    save_output(df, OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("  Pipeline selesai!")
    print(f"  Output: {os.path.abspath(OUTPUT_PATH)}")
    print("=" * 60)
    return df


if __name__ == "__main__":
    main()
