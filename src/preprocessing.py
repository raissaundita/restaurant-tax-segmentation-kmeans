"""
PREPROCESSING: Restaurant Tax Potential Data
Cleans, imputes, and standardizes restaurant operational data
before it is fed into the K-Means clustering model.

Note: the original dataset is confidential (sourced from a
government tax authority) and is not included in this repo.
Point EXCEL_PATH to your own data file with the same column
structure to reproduce this pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

EXCEL_PATH = "path/to/your/data.xlsx"   # ganti dengan path data-mu
SHEET_NAME = "Restoran 2026_v1"
OUTPUT_PATH = "Data_Restoran_Preprocessed.xlsx"

# Fitur yang dipakai untuk clustering
FEATURES = [
    "JUMLAH KURSI",
    "RATA2 BILL PER ORANG (Rp)",
    "TURNOVER WEEKDAYS",
    "TURNOVER WEEKEND",
    "RATA-RATA PENGUNJUNG WEEKDAYS",
    "RATA-RATA PENGUNJUNG WEEKEND",
    "TOTAL OMZET/BULAN"
]

# Kolom identitas/label untuk dilampirkan di output
# (di data asli, kolom ini sudah dianonimisasi sebelum diolah)
ID_COLS = ["NOP", "NAMA", "ALAMAT", "WIL.", "KATEGORI"]

# 1) BACA DATA
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

missing_features = [c for c in FEATURES if c not in df.columns]
if missing_features:
    raise ValueError(f"Kolom berikut tidak ditemukan di sheet: {missing_features}")

# 2) BERSIHKAN FORMAT ANGKA
def to_float_series(s: pd.Series) -> pd.Series:
    """
    Ubah series teks berisi angka dengan tanda pemisah ribuan (koma)
    menjadi float. Contoh "35,000" -> 35000.0
    """
    # ubah ke string, hilangkan spasi, hilangkan koma
    s_clean = (
        s.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
    )
    # ganti "" atau "nan" ke NaN agar bisa diimputasi
    s_clean = s_clean.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s_clean, errors="coerce")

X_raw = df[FEATURES].copy()
for col in FEATURES:
    X_raw[col] = to_float_series(X_raw[col])

# 3) CEK MISSING & IMPUTASI MEDIAN (nilai 0 dibiarkan apa adanya)
before_missing = X_raw.isna().sum().to_dict()

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X_raw)
X_imputed = pd.DataFrame(X_imputed, columns=FEATURES)

after_missing = X_imputed.isna().sum().to_dict()

# 4) STANDARDISASI (Z-SCORE)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_scaled = pd.DataFrame(X_scaled, columns=[f"{c}_scaled" for c in FEATURES])

# 5) GABUNGKAN & SIMPAN
out_df = pd.concat(
    [
        df[ID_COLS] if all(c in df.columns for c in ID_COLS) else df.iloc[:, :0],
        X_imputed.add_suffix("_clean"),
        X_scaled
    ],
    axis=1
)

with pd.ExcelWriter(OUTPUT_PATH, engine="xlsxwriter") as w:
    out_df.to_excel(w, index=False, sheet_name="Data_Bersih_Scaled")

print("=== PREPROCESSING SELESAI ===")
print(f"Simpan ke: {OUTPUT_PATH}\n")

# 6) RINGKASAN / VALIDASI
print(">> Missing sebelum imputasi (tiap kolom):")
for k, v in before_missing.items():
    print(f"  {k:40s} : {v}")

print("\n>> Missing sesudah imputasi (harus 0 semua):")
for k, v in after_missing.items():
    print(f"  {k:40s} : {v}")

print("\n>> Contoh 5 baris pertama (kolom bersih & scaled):")
print(out_df[[f"{c}_clean" for c in FEATURES] + [f"{c}_scaled" for c in FEATURES]].head())

desc = X_imputed.describe().T
desc["missing_before"] = pd.Series(before_missing)
desc.to_csv("ringkasan_preprocessing.csv")
print("\nRingkasan statistik tersimpan sebagai ringkasan_preprocessing.csv")
