"""
MODELING: K-Means Clustering for Restaurant Tax Potential Segmentation

Groups restaurants into tax-potential segments (low, medium, high)
based on operational characteristics, using:
  - Elbow Method & Silhouette Score to choose the optimal number
    of clusters (k)
  - K-Means for the clustering itself
  - PCA (2D & 3D) to visualize the resulting clusters

Run this after preprocessing.py.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1) KONFIGURASI
PREP_PATH = "Data_Restoran_Preprocessed.xlsx"
SHEET_NAME = "Data_Bersih_Scaled"
OUT_XLSX = "Hasil_Cluster_Restoran.xlsx"

# Kolom identitas
ID_COLS = ["NOP", "NAMA", "ALAMAT", "WIL.", "KATEGORI"]

# Nama fitur yang dipilih
FEATURES = [
    "JUMLAH KURSI",
    "RATA2 BILL PER ORANG (Rp)",
    "TURNOVER WEEKDAYS",
    "TURNOVER WEEKEND",
    "RATA-RATA PENGUNJUNG WEEKDAYS",
    "RATA-RATA PENGUNJUNG WEEKEND",
    "TOTAL OMZET/BULAN"
]

# Bentuk nama kolom hasil preprocessing:
CLEAN_COLS = [f"{c}_clean" for c in FEATURES]    # untuk interpretasi (mean per klaster)
SCALED_COLS = [f"{c}_scaled" for c in FEATURES]  # untuk modeling K-Means

# Range kandidat k
K_MIN, K_MAX = 2, 8

# 2) BACA DATA PREPROCESSED
df = pd.read_excel(PREP_PATH, sheet_name=SHEET_NAME)

for col in SCALED_COLS + CLEAN_COLS:
    if col not in df.columns:
        raise ValueError(f"Kolom '{col}' tidak ditemukan. Cek kembali file preprocessing.")

X = df[SCALED_COLS].to_numpy()  # input ke K-Means (sudah Z-score)

# 3) TENTUKAN JUMLAH KLASTER (Elbow & Silhouette)
k_vals = list(range(K_MIN, K_MAX + 1))
wcss, sils = [], []

for k in k_vals:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels_k = km.fit_predict(X)
    wcss.append(km.inertia_)
    sils.append(silhouette_score(X, labels_k))

# Plot Elbow
plt.figure()
plt.plot(k_vals, wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("WCSS")
plt.grid(True)
plt.savefig("elbow_kmeans.png", bbox_inches="tight")
plt.show()

# Plot Silhouette
plt.figure()
plt.plot(k_vals, sils, marker="o")
plt.title("Silhouette")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.savefig("silhouette_kmeans.png", bbox_inches="tight")
plt.show()

best_k_idx = int(np.argmax(sils))
best_k = k_vals[best_k_idx]
print(f"[INFO] k terbaik (berdasarkan silhouette): k={best_k} (score={sils[best_k_idx]:.4f})")

# 4) FIT K-MEANS DENGAN k TERBAIK
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
labels = kmeans.fit_predict(X)

df_out = df.copy()
df_out["Cluster"] = labels

# 5) HASIL
# Jumlah anggota tiap klaster
counts = df_out["Cluster"].value_counts().sort_index().rename("count")

# Rata-rata fitur (pakai _clean agar mudah diinterpretasi rupiah/satuan asli)
means = df_out.groupby("Cluster")[CLEAN_COLS].mean().round(2)
means.columns = [c.replace("_clean", "") for c in means.columns]

print("\n=== Jumlah data per klaster ===")
print(counts)
print("\n=== Rata-rata fitur per klaster ===")
print(means)

# 6) VISUALISASI PCA
# PCA 2D
pca2 = PCA(n_components=2, random_state=42)
Xp2 = pca2.fit_transform(X)

plt.figure()
for k in np.unique(labels):
    mask = (labels == k)
    plt.scatter(Xp2[mask, 0], Xp2[mask, 1], s=30, alpha=0.9, label=f"Cluster {k}")

plt.title(f"PCA 2D K-Means (k={best_k})")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(title="Keterangan", frameon=True)
plt.tight_layout()
plt.savefig("pca2d_kmeans.png", bbox_inches="tight")
plt.show()

# PCA 3D
try:
    from mpl_toolkits.mplot3d import Axes3D
    pca3 = PCA(n_components=3, random_state=42)
    Xp3 = pca3.fit_transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for k in np.unique(labels):
        mask = (labels == k)
        ax.scatter(Xp3[mask, 0], Xp3[mask, 1], Xp3[mask, 2],
                   s=25, alpha=0.9, label=f"Cluster {k}")

    ax.set_title(f"PCA 3D K-Means (k={best_k})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.legend(title="Keterangan", loc="upper left")
    plt.tight_layout()
    plt.savefig("pca3d_kmeans.png", bbox_inches="tight")
    plt.show()
except Exception as e:
    print("[WARN] Plot 3D gagal (opsional):", e)

# 7) SIMPAN KE EXCEL
cols_to_export = []
for c in ID_COLS:
    if c in df_out.columns:
        cols_to_export.append(c)
cols_to_export += CLEAN_COLS
cols_to_export.append("Cluster")

with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as w:
    df_out[cols_to_export].to_excel(w, index=False, sheet_name="data_dengan_klaster")
    counts.to_frame().to_excel(w, sheet_name="jumlah_per_klaster")
    means.to_excel(w, sheet_name="rata_rata_per_klaster")

print(f"\n[OK] Semua hasil disimpan ke: {OUT_XLSX}")
print("Gambar: elbow_kmeans.png, silhouette_kmeans.png, pca2d_kmeans.png, pca3d_kmeans.png")
