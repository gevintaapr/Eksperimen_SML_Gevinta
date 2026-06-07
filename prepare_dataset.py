"""
prepare_dataset.py
==================
Task 1 - Script persiapan dataset RiskTrace.
Dataset risktrace_raw.csv sudah ada dengan is_suspicious dan profile yang benar.
Script ini menambahkan kolom yang masih kurang:
  - round_amount_flag : 1 jika amount % 50000 == 0, else 0
  - channel           : kategori channel transaksi (QRIS/Transfer/Top-up/Withdrawal)

Cara menjalankan:
    python prepare_dataset.py

Output:
    risktrace_raw.csv (diperbarui dengan kolom tambahan)
"""

import pandas as pd
import numpy as np
import os

def add_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Menambahkan kolom yang belum ada ke dataset."""

    # Tambah round_amount_flag: 1 jika amount kelipatan 50.000
    if "round_amount_flag" not in df.columns:
        df["round_amount_flag"] = (df["amount"] % 50000 == 0).astype(int)
        print(f"  [+] round_amount_flag ditambahkan — distribusi: {df['round_amount_flag'].value_counts().to_dict()}")
    else:
        print("  [=] round_amount_flag sudah ada, skip.")

    # Tambah channel jika belum ada
    if "channel" not in df.columns:
        np.random.seed(42)
        # Buat channel berdasarkan qris_ratio_7d dan burst_score
        channels = []
        for _, row in df.iterrows():
            if row["qris_ratio_7d"] > 0.5:
                ch = "QRIS"
            elif row["burst_score"] > 0.6:
                ch = "Transfer"
            elif row["amount"] < 100000:
                ch = "Top-up"
            else:
                ch = "Withdrawal"
            channels.append(ch)
        df["channel"] = channels
        print(f"  [+] channel ditambahkan — distribusi: {df['channel'].value_counts().to_dict()}")
    else:
        print("  [=] channel sudah ada, skip.")

    return df


def main():
    input_path = "risktrace_raw.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")

    print(f"[1] Memuat dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"    Shape awal: {df.shape}")
    print(f"    Kolom: {df.columns.tolist()}")

    print("\n[2] Menambahkan kolom yang belum ada...")
    df = add_missing_columns(df)

    print(f"\n[3] Kolom akhir ({len(df.columns)}): {df.columns.tolist()}")
    print(f"    Shape akhir: {df.shape}")
    print(f"    Distribusi is_suspicious:\n{df['is_suspicious'].value_counts()}")
    print(f"    Distribusi profile:\n{df['profile'].value_counts()}")

    df.to_csv(input_path, index=False)
    print(f"\n[4] Dataset disimpan: {input_path}")
    return df


if __name__ == "__main__":
    main()
