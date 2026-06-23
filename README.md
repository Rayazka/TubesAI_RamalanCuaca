# TubesAI_RamalanCuaca
Proyek Akhir Project Based Learning (PBL) Mata Kuliah Kecerdasan Buatan (Artificial Intelligence) - S1 Rekayasa Perangkat Lunak.

Proyek ini bertujuan untuk memprediksi cuaca harian (Hujan vs Tidak Hujan) di wilayah Kelapa Gading, Jakarta Utara menggunakan dua model klasifikasi yang dibangun dari awal (*from scratch*):
1. **K-Nearest Neighbors (KNN)**
2. **Gaussian Naive Bayes**

## Anggota Kelompok
* **Rayazka** (NIM: ...) - Implementasi KNN, Preprocessing, Evaluasi, dan Integrasi.
* **Partner** (NIM: ...) - Implementasi Gaussian Naive Bayes.

## Struktur Folder
```
├── data/
│   └── cuaca-harian-dki2-kelapagading.csv  # Dataset cuaca harian
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                    # Preprocessing, Min-Max scaling, dan Train-test split
│   ├── model_knn.py                        # Model KNN Classifier dari awal (from scratch)
│   ├── model_naive_bayes.py                # Model Gaussian Naive Bayes dari awal (from scratch)
│   └── evaluation.py                       # Metrik evaluasi (Akurasi, F1-Score, Confusion Matrix)
├── main.py                                 # Pipeline utama untuk melatih dan membandingkan model
└── README.md
```

## Dataset
Dataset yang digunakan berasal dari data cuaca harian Kelapa Gading (`data/cuaca-harian-dki2-kelapagading.csv`).
* Target klasifikasi: `is_rain` (1 jika curah hujan > 3.0 mm, 0 jika <= 3.0 mm).
* Jumlah data: 5.722 baris.
* Fitur input: Suhu udara, Kelembapan, Tekanan udara, Awan, Kecepatan angin, dan Radiasi matahari (total 21 fitur prediktor setelah menghapus kolom bocor).

## Cara Menjalankan Program
Jalankan file `main.py` menggunakan Python:
```bash
python main.py
```
Program akan memproses data secara otomatis, melatih model KNN dan Naive Bayes, serta menampilkan hasil evaluasi komparatif kedua model di terminal.
