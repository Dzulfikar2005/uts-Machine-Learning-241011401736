# UTS Machine Learning — Linear Regression & Logistic Regression

## Identitas Mahasiswa

- **Nama:** Mufti Dzulfikar Basir
- **NIM:** 241011401736
- **Kelas:** Reg B
- **Random state:** `36`

## Tujuan

Tugas ini menerapkan Linear Regression untuk memodelkan hubungan biaya iklan dan penjualan, serta Logistic Regression untuk mengklasifikasikan email spam. Implementasi utama model, pembagian data, dan metrik evaluasi dibuat menggunakan NumPy sesuai instruksi tugas. Scikit-learn hanya digunakan pada bagian bonus sebagai pembanding.

## Dataset

- **`Ads_Data.csv`** berisi 40 baris dengan kolom `BiayaIklan` dan `Penjualan`. Fitur biaya iklan berada pada rentang 0,8–9,9 juta rupiah.
- **`Email_Data.csv`** berisi 60 email dengan fitur `JumlahLink` dan `KataPromo`, serta target `Spam` (0 = Bukan Spam, 1 = Spam). Dataset terdiri dari 35 email bukan spam dan 25 email spam.

## Hasil Bagian A — Linear Regression

Model dilatih menggunakan **Normal Equation** pada pembagian data `test_size=0.2` dan `random_state=36`. Persamaan garis yang diperoleh adalah:

\[
\widehat{Penjualan} = 28.5295 + 9.0249 \times BiayaIklan
\]

Koefisien positif menunjukkan bahwa kenaikan biaya iklan 1 juta rupiah berkaitan dengan kenaikan prediksi penjualan sekitar 9,02 satuan pada dataset ini.

| Data | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Training (32 baris) | 5.0748 | 43.0396 | 6.5605 | 0.9345 |
| Testing (8 baris) | 2.3982 | 8.3409 | 2.8881 | 0.9835 |

RMSE testing sebesar 2,8881 dan R² testing sebesar 0,9835 menunjukkan kecocokan yang baik pada subset testing ini. Namun, dataset hanya berisi 40 baris, sehingga hasil tersebut tidak menjamin performa yang sama pada data lain. Prediksi pada biaya iklan 12 juta rupiah merupakan ekstrapolasi karena berada di luar rentang dataset.

### Eksperimen Gradient Descent

Eksperimen memakai 5.000 iterasi dengan tiga learning rate:

| Learning rate | Intercept akhir | Koefisien akhir | Status |
|---:|---:|---:|---|
| 0,001 | 19.327982 | 10.414271 | Masih jauh dari solusi Normal Equation pada batas iterasi ini |
| 0,02 | 28.529487 | 9.024889 | Konvergen; cocok dengan Normal Equation pada presisi yang ditampilkan |
| 0,1 | Tidak finite | Tidak finite | Divergen pada iterasi ke-372 |

Learning rate 0,001 bergerak terlalu lambat dalam 5.000 iterasi, sedangkan 0,02 memberikan hasil stabil. Learning rate 0,1 menghasilkan nilai non-finite akibat langkah pembaruan terlalu besar; hasil tersebut tidak diperlakukan sebagai model yang berhasil konvergen.

## Hasil Bagian B — Logistic Regression

Model Logistic Regression diimplementasikan dengan Gradient Descent, `learning_rate=0.1`, `n_iterations=10000`, dan pembagian data `test_size=0.25`, `random_state=36`. Parameter yang diperoleh:

\[
P(Spam)=\sigma(-8.4186 + 0.6935\,JumlahLink + 0.7784\,KataPromo)
\]

Log loss turun dari **0,682070** menjadi **0,306269** selama training.

Pada data testing (15 email) dengan threshold 0,5, hasil evaluasi adalah:

- **Accuracy:** 0,9333
- **Precision:** 0,8571
- **Recall:** 1,0000
- **F1-score:** 0,9231
- **Confusion matrix** dengan urutan baris label aktual `[Bukan Spam, Spam]` dan kolom prediksi `[Bukan Spam, Spam]`:

```text
[[TN, FP], [FN, TP]] = [[8, 1], [0, 6]]
```

Pada eksperimen threshold, nilai 0,3 menghasilkan precision 0,6667 dan recall 1,0000; threshold 0,5 menghasilkan precision 0,8571 dan recall 1,0000; threshold 0,7 menghasilkan precision 1,0000 dan recall 0,8333. Threshold 0,5 dipakai untuk prediksi email baru karena pada split ini mempertahankan recall penuh dengan satu false positive. Model ini merupakan filter sederhana berbasis dua fitur dan bukan pengganti sistem keamanan email yang lebih menyeluruh.

## Bonus — Perbandingan dengan scikit-learn

- **Linear Regression:** implementasi NumPy dan scikit-learn menghasilkan intercept 28,529487, koefisien 9,024889, RMSE testing 2,888060, dan R² testing 0,983543 pada split yang sama.
- **Logistic Regression:** pada threshold 0,5, implementasi NumPy dan scikit-learn menghasilkan confusion matrix yang sama, yaitu `[[8, 1], [0, 6]]`, dengan accuracy 0,9333, precision 0,8571, recall 1,0000, dan F1-score 0,9231. Parameter Logistic Regression tidak identik karena proses optimasi library berbeda dari Gradient Descent manual.

## Struktur Repository

```text
uts-machine-learning-241011401736/
├── README.md
├── tugas_UTS.ipynb
├── Ads_Data.csv
└── Email_Data.csv
```

## Menjalankan Notebook

Notebook telah dieksekusi dari awal hingga akhir dan output tabel, metrik, prediksi, serta grafik disimpan di `tugas_UTS.ipynb`.

Dependensi yang digunakan pada lingkungan verifikasi:

- Python 3.13.5
- NumPy 2.3.5
- Pandas 2.2.3
- Matplotlib 3.10.8
- scikit-learn 1.8.0 (hanya untuk bonus)
- ipykernel 7.2.0
- nbformat 5.10.4 dan nbclient 0.10.4 untuk verifikasi eksekusi notebook

Untuk menjalankan di lingkungan sendiri, pasang dependensi yang diperlukan, buka notebook melalui Jupyter atau VS Code, lalu pilih **Run All**. Pastikan direktori kerja notebook sama dengan folder yang berisi kedua file CSV.
