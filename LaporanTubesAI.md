# LAPORAN TUGAS BESAR PROJECT BASED LEARNING (PBL)
## MATA KULIAH KECERDASAN BUATAN (ARTIFICIAL INTELLIGENCE)
### PROGRAM STUDI S1 REKAYASA PERANGKAT LUNAK

---

## IDENTITAS KELOMPOK & PERAN ANGGOTA

* **Kelas**: SE48XX (Silakan sesuaikan dengan kelas Anda)
* **Nomor Kelompok**: XX (Silakan sesuaikan dengan nomor kelompok Anda)
* **Anggota Kelompok**:
  1. **Rayazka Aris Fadhilahn** (NIM: 130222XXXX - *Silakan lengkapi NIM Anda*)
     * **Peran & Kontribusi**:
       * Merancang dan mengimplementasikan modul preprocessing data (`src/preprocessing.py`) termasuk pemuatan CSV ke NumPy, pemisahan data latih/uji (*train-test split*), dan normalisasi skala fitur (*Min-Max scaling*).
       * Membangun model *K-Nearest Neighbors (KNN) Classifier* dari awal (*from scratch*) dengan optimasi komputasi matriks/vektorisasi NumPy (`src/model_knn.py`).
       * Mengintegrasikan metrik evaluasi (`src/evaluation.py`) dan merancang alur utama pipeline program (`main.py`).
       * Melakukan eksperimen tuning hyperparameter $K$ pada KNN dan menyusun visualisasi grafik laporan.
  2. **Partner Kelompok** (NIM: 130222XXXX - *Silakan lengkapi nama dan NIM rekan kelompok*)
     * **Peran & Kontribusi**:
       * Membangun model *Gaussian Naive Bayes Classifier* dari awal (*from scratch*) dengan memanfaatkan operasi kolom NumPy untuk perhitungan mean, varians, prior, dan likelihood secara efisien (`src/model_naive_bayes.py`).
       * Membantu memvalidasi implementasi matematika model terhadap hasil pengujian serta melakukan analisis perbandingan performa.

---

## 1. DESKRIPSI KASUS & DATASET

### A. Kasus yang Dipilih
Kasus yang diselesaikan dalam proyek ini adalah **klasifikasi cuaca harian biner (Hujan vs Tidak Hujan)** di wilayah **Kelapa Gading, Jakarta Utara** berdasarkan parameter cuaca harian. Prediksi cuaca harian yang akurat sangat krusial untuk mendukung berbagai aktivitas masyarakat, mulai dari transportasi, logistik, hingga mitigasi bencana banjir lokal. 

Dalam perspektif pembelajaran mesin, kasus ini dimodelkan sebagai **Supervised Learning - Binary Classification**, di mana model menerima parameter-parameter kondisi atmosfer harian dan memprediksi apakah curah hujan pada hari tersebut melampaui ambang batas tertentu atau tidak.

### B. Analisis Dataset
Dataset yang digunakan disimpan pada berkas `data/cuaca-harian-dki2-kelapagading.csv` dengan total **5.722 baris data**. 

Target klasifikasi biner disimbolkan sebagai variabel $y \in \{0, 1\}$ (diberi nama `is_rain` di memori) yang diturunkan dari kolom asli `precipitation_sum (mm)` dengan aturan ambang batas (*threshold*):
$$y = \begin{cases} 
1, & \text{jika } \text{precipitation\_sum} > 3.0 \text{ mm (Hari Hujan)} \\ 
0, & \text{jika } \text{precipitation\_sum} \le 3.0 \text{ mm (Hari Tidak Hujan)} 
\end{cases}$$

Setelah proses pembersihan dan eliminasi kolom yang tidak relevan, didapatkan **21 fitur prediktor numerik** sebagai berikut:

| No | Nama Kolom Fitur | Satuan | Deskripsi |
|----|------------------|--------|-----------|
| 1  | `temperature_2m_max` | °C | Suhu Udara Maksimum pada ketinggian 2 meter |
| 2  | `temperature_2m_min` | °C | Suhu Udara Minimum pada ketinggian 2 meter |
| 3  | `wind_speed_10m_max` | km/h | Kecepatan Angin Maksimum pada ketinggian 10 meter |
| 4  | `wind_direction_10m_dominant` | ° | Arah Angin Dominan pada ketinggian 10 meter |
| 5  | `shortwave_radiation_sum` | MJ/m² | Total Radiasi Gelombang Pendek Matahari |
| 6  | `temperature_2m_mean` | °C | Rata-rata Suhu Udara Harian |
| 7  | `relative_humidity_2m_mean` | % | Rata-rata Kelembapan Relatif Harian |
| 8  | `cloud_cover_mean` | % | Rata-rata Persentase Tutupan Awan |
| 9  | `surface_pressure_mean` | hPa | Rata-rata Tekanan Udara Permukaan |
| 10 | `wind_gusts_10m_max` | km/h | Hembusan Angin Maksimum Harian |
| 11 | `winddirection_10m_dominant` | ° | Arah Hembusan Angin Dominan |
| 12 | `relative_humidity_2m_max` | % | Kelembapan Relatif Maksimum |
| 13 | `relative_humidity_2m_min` | % | Kelembapan Relatif Minimum |
| 14 | `cloud_cover_max` | % | Persentase Tutupan Awan Maksimum |
| 15 | `cloud_cover_min` | % | Persentase Tutupan Awan Minimum |
| 16 | `wind_gusts_10m_mean` | km/h | Rata-rata Hembusan Angin Harian |
| 17 | `wind_speed_10m_mean` | km/h | Rata-rata Kecepatan Angin Harian |
| 18 | `wind_gusts_10m_min` | km/h | Hembusan Angin Minimum Harian |
| 19 | `wind_speed_10m_min` | km/h | Kecepatan Angin Minimum Harian |
| 20 | `surface_pressure_max` | hPa | Tekanan Udara Permukaan Maksimum |
| 21 | `surface_pressure_min` | hPa | Tekanan Udara Permukaan Minimum |

### C. Penanganan Kebocoran Data (Data Leakage)
*Data Leakage* terjadi ketika fitur-fitur yang mengandung informasi masa depan atau informasi target secara tidak sengaja disertakan dalam proses training model. Hal ini membuat model tampak berkinerja sangat baik saat pengujian, namun gagal total saat diterapkan di dunia nyata. Untuk mencegah hal tersebut, tiga kolom berikut dieliminasi secara ketat dari fitur prediktor:
1. **`time`**: Merupakan metadata waktu/tanggal yang tidak merepresentasikan kondisi atmosfer fisis secara langsung untuk prediksi cuaca instan.
2. **`precipitation_sum (mm)`**: Kolom ini merupakan sumber langsung pembentukan label target biner `is_rain`. Menyertakan fitur ini akan membuat model melakukan prediksi secara trivial (hanya membaca kolom ini) dan menghasilkan akurasi palsu 100%.
3. **`precipitation_hours (h)`**: Jumlah jam hujan secara langsung berkolerasi dengan curah hujan harian. Fitur ini hanya diketahui *setelah* hari tersebut selesai dilewati, sehingga tidak valid digunakan sebagai prediktor sebelum kejadian.

---

## 2. ANALISIS PROSES PREPROCESSING

Proses preprocessing data dilakukan secara terstruktur dalam modul `src/preprocessing.py` dengan memanfaatkan optimasi array NumPy.

### A. Pemuatan Data (Data Loading)
Fungsi `load_data` membaca data dari file CSV menggunakan pustaka standar Python `csv` baris demi baris guna meminimalisasi overhead memori. Baris kosong dilewati secara aman. Setelah menyaring kolom-kolom *data leakage*, fitur prediktor dan label target dikonversi menjadi tipe array NumPy `np.ndarray`. Penggunaan tipe data NumPy ini sangat vital karena memungkinkan komputasi berikutnya (normalisasi, perhitungan jarak, dsb.) memanfaatkan instruksi vektorisasi tingkat rendah yang jauh lebih cepat daripada menggunakan struktur data list bawaan Python.

### B. Pembagian Data (Train-Test Split)
Untuk menguji kemampuan generalisasi model pada data yang belum pernah dilihat sebelumnya, dataset dibagi menjadi **80% data latih (train)** dan **20% data uji (test)** melalui fungsi `train_test_split`.
* Jumlah Data Latih (*Train*): **4.577 sampel**
* Jumlah Data Uji (*Test*): **1.145 sampel**

Pembagian ini memanfaatkan fungsi generator acak NumPy (`np.random.default_rng`) dengan parameter seed tetap (`seed = 42`). Hal ini memastikan pembagian baris data dilakukan secara acak namun konsisten (reproducible) setiap kali program dijalankan.

Distribusi kelas setelah pembagian data adalah sebagai berikut:
* **Data Latih**: Hujan = 2.399 (52,4%), Tidak Hujan = 2.178 (47,6%)
* **Data Uji**: Hujan = 632 (55,2%), Tidak Hujan = 513 (44,8%)
Data terlihat cukup seimbang (*well-balanced*), sehingga metrik akurasi dapat digunakan dengan aman bersama dengan metrik evaluasi lainnya seperti F1-Score.

### C. Normalisasi Fitur (Min-Max Scaling)
Skala nilai antar fitur pada dataset asli sangat timpang (misal, fitur tekanan udara berkisar di angka 1000-an hPa, sedangkan suhu berada di angka 20-30an °C). Algoritma seperti KNN yang bertumpu pada jarak spasial akan sangat terdistorsi oleh fitur berskala besar. Oleh karena itu, diterapkan **Min-Max Scaling** untuk menyetarakan rentang seluruh fitur ke dalam interval $[0, 1]$ menggunakan rumus:
$$X_{\text{scaled}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$$

Untuk menghindari **kebocoran data uji (data leakage)**, parameter normalisasi ($X_{\text{min}}$ dan $X_{\text{max}}$) dihitung secara eksklusif hanya dari data latih (`fit_min_max`):
```python
min_val = np.min(X_train, axis=0)
max_val = np.max(X_train, axis=0)
```
Setelah parameter didapatkan, transformasi matematika dilakukan secara vektorisasi terarah (*broadcasting*) pada data latih dan data uji menggunakan fungsi `transform_min_max`:
```python
range_val = np.where(max_val - min_val == 0.0, 1.0, max_val - min_val)
X_scaled = (X - min_val) / range_val
```
Penggunaan `np.where` di atas bertujuan untuk mengantisipasi pembagian dengan nol apabila terdapat fitur konstan di mana $X_{\text{max}} - X_{\text{min}} = 0$.

---

## 3. DESAIN ALGORITMA (FROM SCRATCH)

Sesuai ketentuan tugas, kedua model klasifikasi dibangun sepenuhnya dari awal (*from scratch*) tanpa pustaka tingkat tinggi seperti *scikit-learn*.

### A. K-Nearest Neighbors (KNN) Classifier
Algoritma KNN bekerja berdasarkan prinsip bahwa sampel data baru cenderung memiliki kelas yang sama dengan sampel terdekat di ruang fitur. Modul `src/model_knn.py` mengimplementasikan alur berikut:

1. **Penyimpanan Data Latih (`fit`)**: KNN adalah tipe *lazy learner*, sehingga proses training hanya menyimpan matriks fitur berskala `X_train` dan vektor label `y_train` ke dalam memori.
2. **Kalkulasi Jarak Vektoral (`predict_one`)**: Untuk memprediksi satu baris data uji $x$, dihitung kuadrat jarak Euclidean ke seluruh sampel di data latih secara paralel menggunakan fitur *broadcasting* NumPy:
   $$d(x, x')^2 = \sum_{i=1}^{D} (x_i - x'_i)^2$$
   Kode implementasinya:
   ```python
   dists_sq = np.sum((self.X_train - x) ** 2, axis=1)
   ```
   Operasi ini memproses ribuan baris data latih sekaligus dalam satu operasi CPU tingkat rendah, menghindari loop iterasi `for` Python yang lambat.
3. **Pencarian Tetangga Terdekat**: Menggunakan fungsi `np.argpartition` untuk mengisolasi $K$ indeks dengan jarak terkecil dalam kompleksitas waktu rata-rata $O(N)$, diikuti dengan pengurutan `np.argsort`:
   ```python
   k_indices = np.argpartition(dists_sq, self.k)[:self.k]
   k_indices = k_indices[np.argsort(dists_sq[k_indices])]
   ```
4. **Voting Mayoritas & Pemecah Seri (Tie-Breaker)**: Label dari $K$ tetangga terdekat dihitung kemunculannya. Jika terjadi hasil voting seri (jumlah suara kelas 0 dan kelas 1 sama), model akan memilih kelas milik tetangga yang memiliki jarak spasial paling dekat (*tie-breaker* berbasis kedekatan urutan):
   ```python
   k_labels = self.y_train[k_indices]
   unique_labels, counts = np.unique(k_labels, return_counts=True)
   max_count = np.max(counts)
   candidates = unique_labels[counts == max_count]
   if len(candidates) == 1:
       return int(candidates[0])
   for label in k_labels:
       if label in candidates:
           return int(label)
   ```

### B. Gaussian Naive Bayes Classifier
Gaussian Naive Bayes didasarkan pada Teorema Bayes dengan asumsi bahwa setiap fitur bersifat independen satu sama lain (naif) dan mengikuti distribusi normal (Gaussian). Modul `src/model_naive_bayes.py` mengimplementasikan alur berikut:

1. **Fase Pelatihan (`fit`)**: Menghitung probabilitas prior $P(y = c)$ untuk tiap kelas $c$, serta nilai rata-rata ($\mu_c$) dan varians ($\sigma_c^2$) untuk setiap dimensi fitur pada data latih:
   $$P(y = c) = \frac{N_c}{N}, \quad \mu_c = \frac{1}{N_c} \sum_{i=1}^{N_c} x_i, \quad \sigma_c^2 = \frac{1}{N_c} \sum_{i=1}^{N_c} (x_i - \mu_c)^2$$
   Operasi ini dihitung secara efisien dengan NumPy:
   ```python
   self.priors[c] = len(X_c) / n_samples
   self.means[c] = np.mean(X_c, axis=0)
   self.variances[c] = np.var(X_c, axis=0)
   ```
2. **Kalkulasi Log-Likelihood Gaussian (`_calculate_log_likelihood`)**: Probabilitas kontinu fitur terhadap kelas $P(x_i | y = c)$ dihitung menggunakan rumus Probability Density Function (PDF) Gaussian:
   $$P(x_i | y=c) = \frac{1}{\sqrt{2\pi\sigma_{c,i}^2}} \exp\left(-\frac{(x_i - \mu_{c,i})^2}{2\sigma_{c,i}^2}\right)$$
   Untuk menghindari fenomena *numerical underflow* (perkalian probabilitas kecil yang menghasilkan nilai mendekati nol), perhitungan ditransformasikan ke dalam domain logaritma alami (log-likelihood):
   $$\ln P(x_i | y=c) = -0.5 \ln(2\pi\sigma_{c,i}^2) - \frac{(x_i - \mu_{c,i})^2}{2\sigma_{c,i}^2}$$
   Seluruh nilai log-likelihood fitur dijumlahkan secara vektor untuk satu sampel:
   ```python
   eps = 1e-9 # Mencegah pembagian nol
   var = variance + eps
   log_pdf = -0.5 * np.log(2.0 * np.pi * var) - ((x - mean) ** 2) / (2.0 * var)
   return np.sum(log_pdf)
   ```
3. **Keputusan Posterior (`predict_one`)**: Posterior probabilitas ditentukan berdasarkan penjumlahan log-prior dan log-likelihood. Kelas dengan skor posterior terbesar dipilih sebagai hasil prediksi:
   $$\text{Posterior}(c) = \ln P(y = c) + \sum_{i=1}^{D} \ln P(x_i | y=c)$$
   ```python
   posteriors[c] = np.log(self.priors[c]) + self._calculate_log_likelihood(x, self.means[c], self.variances[c])
   return int(max(posteriors, key=posteriors.get))
   ```

---

## 4. MODEL EVALUASI & HASIL EKSPERIMEN

### A. Metrik Evaluasi
Metrik performa dihitung secara manual pada modul `src/evaluation.py` menggunakan komponen Confusion Matrix:
* **True Positive (TP)**: Aktual Hujan, Prediksi Hujan
* **True Negative (TN)**: Aktual Tidak Hujan, Prediksi Tidak Hujan
* **False Positive (FP)**: Aktual Tidak Hujan, Prediksi Hujan
* **False Negative (FN)**: Aktual Hujan, Prediksi Tidak Hujan

Rumus metrik evaluasi yang digunakan:
1. **Akurasi**: Proporsi prediksi yang tepat dari keseluruhan data uji.
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Presisi**: Tingkat ketepatan prediksi cuaca hujan.
   $$\text{Precision} = \frac{TP}{TP + FP}$$
3. **Recall (Sensitivitas)**: Kemampuan model menjaring seluruh hari yang sebenarnya hujan.
   $$\text{Recall} = \frac{TP}{TP + FN}$$
4. **F1-Score**: Rata-rata harmonik antara presisi dan recall, berguna sebagai penyeimbang evaluasi.
   $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### B. Eksperimen Hyperparameter Tuning K pada KNN
Eksperimen dilakukan dengan melatih model KNN pada data training ternormalisasi dan mengevaluasinya pada data testing menggunakan beberapa variasi nilai $K$ ganjil (untuk menghindari voting seimbang):

| Nilai K | Akurasi | F1-Score | Waktu Prediksi (Detik) |
| :---: | :---: | :---: | :---: |
| K = 1 | 76.07% | 78.66% | 1.976 |
| K = 3 | 79.39% | 81.93% | 2.371 |
| K = 5 | 81.66% | 84.02% | 1.950 |
| K = 7 | 81.40% | 83.88% | 1.952 |
| **K = 9 (Terbaik)** | **82.10%** | **84.57%** | **2.251** |

Hasil tuning menunjukkan bahwa performa klasifikasi meningkat seiring bertambahnya nilai $K$, di mana nilai **$K = 9$** menghasilkan akurasi tertinggi sebesar **82,10%** dan F1-Score **84,57%**. 

### C. Komparasi Performa: KNN (K=9) vs Gaussian Naive Bayes
Berikut adalah hasil perbandingan performa komparatif antara model KNN terbaik ($K=9$) dengan Gaussian Naive Bayes pada dataset uji yang sama:

| Metrik Evaluasi | KNN Classifier (K = 9) | Gaussian Naive Bayes |
|---|:---:|:---:|
| **Akurasi** | **82.10%** | 81.66% |
| **Presisi** | **80.63%** | 79.97% |
| **Recall** | 88.92% | **89.08%** |
| **F1-Score** | **84.57%** | 84.28% |
| **Waktu Pelatihan (Train)** | $\approx$ 0.00 detik (Lazy Learning) | $\approx$ **0.01 detik** |
| **Waktu Prediksi (Test)** | $\approx$ 2.25 detik | $\approx$ **0.07 detik** |

#### Analisis Hasil Komparasi:
1. **Performa Prediksi**: Kedua model menunjukkan performa prediksi yang sangat kompetitif dengan akurasi di atas 81% dan F1-score di kisaran 84%. KNN ($K=9$) sedikit lebih unggul dalam Akurasi (+0,44%), Presisi (+0,66%), dan F1-score (+0,29%). Sementara itu, Gaussian Naive Bayes memiliki sensitivitas/Recall yang sedikit lebih baik (+0,16%), yang berarti NB sedikit lebih sensitif dalam mendeteksi kejadian hujan meskipun memiliki rasio kesalahan prediksi hujan (*False Positive*) yang sedikit lebih tinggi.
2. **Efisiensi Waktu Komputasi**: Perbedaan paling dramatis terletak pada waktu prediksi. **Gaussian Naive Bayes melakukan prediksi 32 kali lebih cepat** ($\approx 0.07$ detik) dibandingkan KNN ($\approx 2.25$ detik). Hal ini dikarenakan KNN harus menghitung jarak Euclidean dari setiap data uji ke seluruh data latih ($1.145 \times 4.577 = 5.240.665$ perhitungan jarak), sedangkan Naive Bayes hanya mengevaluasi fungsi densitas probabilitas Gaussian sederhana berbasis parameter rata-rata dan varians kelas yang telah dihitung sebelumnya saat proses training.

---

## 5. SCREENSHOT & OUTPUT RUNNING PROGRAM

Berikut adalah keluaran (*output*) langsung dari terminal saat menjalankan pipeline utama proyek (`python main.py`):

```text
==================================================
   MEMULAI PIPELINE PREDIKSI CUACA KELAPA GADING 
==================================================

[1/5] Memuat data dari berkas: data/cuaca-harian-dki2-kelapagading.csv...
      -> Berhasil memuat 5722 sampel data dengan 21 kolom fitur prediktor.
      -> Sampel Fitur: temperature_2m_max (°C), temperature_2m_min (°C), wind_speed_10m_max (km/h), wind_direction_10m_dominant (°) ... (+ 17 fitur lainnya)

[2/5] Melakukan Train-Test Split (Rasio 80:20)...
      -> Jumlah sampel Data Latih (Train): 4577 sampel
      -> Jumlah sampel Data Uji (Test)  : 1145 sampel
      -> Distribusi Kelas Train  : Hujan = 2399 (52.4%), Tidak Hujan = 2178 (47.6%)
      -> Distribusi Kelas Test   : Hujan = 632 (55.2%), Tidak Hujan = 513 (44.8%)

[3/5] Melakukan Normalisasi Fitur (Min-Max Scaling [0,1]) secara manual dengan NumPy...
      -> Normalisasi selesai tanpa terjadi kebocoran data uji (data leakage).

[4/5] Melatih dan Menguji Model KNN Classifier (From Scratch - Optimasi NumPy)...
      -> Menguji KNN dengan K = 1...
         Akurasi: 76.07% | F1-Score: 78.66% | Waktu Prediksi: 1.976 detik
      -> Menguji KNN dengan K = 3...
         Akurasi: 79.39% | F1-Score: 81.93% | Waktu Prediksi: 2.371 detik
      -> Menguji KNN dengan K = 5...
         Akurasi: 81.66% | F1-Score: 84.02% | Waktu Prediksi: 1.950 detik
      -> Menguji KNN dengan K = 7...
         Akurasi: 81.40% | F1-Score: 83.88% | Waktu Prediksi: 1.952 detik
      -> Menguji KNN dengan K = 9...
         Akurasi: 82.10% | F1-Score: 84.57% | Waktu Prediksi: 2.251 detik

      => Hasil Terbaik KNN diperoleh pada nilai K = 9

==============================================
  KNN CLASSIFIER TERBAIK (K = 9)
==============================================
 Accuracy  : 0.8210 (82.10%)
 Precision : 0.8063 (80.63%)
 Recall    : 0.8892 (88.92%)
 F1-Score  : 0.8457 (84.57%)
----------------------------------------------
 Confusion Matrix:
                    Predicted Hujan    Predicted Tidak Hujan
 Actual Hujan            562                70             
 Actual Tidak Hujan      135                378            
==============================================
      => Waktu Pengujian KNN: 2.251 detik

[5/5] Melatih dan Menguji Model Gaussian Naive Bayes (From Scratch - Optimasi NumPy)...
[Naive Bayes] Model berhasil dilatih secara efisien dengan NumPy.

==============================================
  GAUSSIAN NAIVE BAYES
==============================================
 Accuracy  : 0.8166 (81.66%)
 Precision : 0.7997 (79.97%)
 Recall    : 0.8908 (89.08%)
 F1-Score  : 0.8428 (84.28%)
----------------------------------------------
 Confusion Matrix:
                    Predicted Hujan    Predicted Tidak Hujan
 Actual Hujan            563                69             
 Actual Tidak Hujan      141                372            
==============================================
      => Waktu Latih NB  : 0.010601 detik
      => Waktu Prediksi NB: 0.068564 detik

[Visualisasi] Menggambar dan menyimpan grafik laporan ke folder 'reports/'...
[Visualisasi] Confusion Matrix Heatmap berhasil disimpan ke: reports/confusion_matrices.png
[Visualisasi] Perbandingan metrik berhasil disimpan ke: reports/metrics_comparison.png
[Visualisasi] Perbandingan waktu komputasi berhasil disimpan ke: reports/time_comparison.png
[Visualisasi] Grafik tuning parameter K KNN berhasil disimpan ke: reports/knn_k_tuning.png
      -> Semua diagram visualisasi berhasil disimpan di folder 'reports/'.

==================================================
    HASIL PREDIKSI UNTUK 7 HARI PERTAMA (DATA UJI) 
==================================================

Hari ke-1:
      [Info Cuaca] Rata-rata Suhu: 25.6°C | Kelembapan: 87.0% | Awan: 79.0% | Kec. Angin: 4.8 km/h
      [Aktual]     : HUJAN
      [Model KNN]  : HUJAN (Persentase Keyakinan: 100.0%)
      [Model GNB]  : HUJAN (Persentase Keyakinan: 99.9%)

Hari ke-2:
      [Info Cuaca] Rata-rata Suhu: 27.1°C | Kelembapan: 74.0% | Awan: 65.0% | Kec. Angin: 7.9 km/h
      [Aktual]     : TIDAK HUJAN
      [Model KNN]  : TIDAK HUJAN (Persentase Keyakinan: 100.0%)
      [Model GNB]  : TIDAK HUJAN (Persentase Keyakinan: 100.0%)

Hari ke-3:
      [Info Cuaca] Rata-rata Suhu: 26.5°C | Kelembapan: 84.0% | Awan: 87.0% | Kec. Angin: 4.0 km/h
      [Aktual]     : HUJAN
      [Model KNN]  : HUJAN (Persentase Keyakinan: 55.6%)
      [Model GNB]  : HUJAN (Persentase Keyakinan: 99.9%)

Hari ke-4:
      [Info Cuaca] Rata-rata Suhu: 26.4°C | Kelembapan: 85.0% | Awan: 77.0% | Kec. Angin: 6.5 km/h
      [Aktual]     : HUJAN
      [Model KNN]  : HUJAN (Persentase Keyakinan: 88.9%)
      [Model GNB]  : HUJAN (Persentase Keyakinan: 100.0%)

Hari ke-5:
      [Info Cuaca] Rata-rata Suhu: 26.9°C | Kelembapan: 82.0% | Awan: 95.0% | Kec. Angin: 7.6 km/h
      [Aktual]     : HUJAN
      [Model KNN]  : HUJAN (Persentase Keyakinan: 66.7%)
      [Model GNB]  : HUJAN (Persentase Keyakinan: 99.9%)

Hari ke-6:
      [Info Cuaca] Rata-rata Suhu: 25.5°C | Kelembapan: 92.0% | Awan: 95.0% | Kec. Angin: 8.1 km/h
      [Aktual]     : HUJAN
      [Model KNN]  : HUJAN (Persentase Keyakinan: 100.0%)
      [Model GNB]  : HUJAN (Persentase Keyakinan: 100.0%)

Hari ke-7:
      [Info Cuaca] Rata-rata Suhu: 27.8°C | Kelembapan: 77.0% | Awan: 83.0% | Kec. Angin: 6.5 km/h
      [Aktual]     : TIDAK HUJAN
      [Model KNN]  : TIDAK HUJAN (Persentase Keyakinan: 77.8%)
      [Model GNB]  : TIDAK HUJAN (Persentase Keyakinan: 100.0%)

==================================================

==================================================
      EKSKUSI PIPELINE SELESAI DENGAN SUKSES      
==================================================
```

### Visualisasi Hasil Eksperimen (Tersimpan di Folder `reports/`)

Berikut adalah visualisasi grafis yang dihasilkan secara otomatis untuk mendukung analisis laporan ini:

1. **Confusion Matrix Heatmap**
   
   Menunjukkan rincian prediksi benar dan salah untuk masing-masing model.
   ![Confusion Matrices](reports/confusion_matrices.png)

2. **Metrics Comparison Chart**
   
   Perbandingan akurasi, presisi, recall, dan F1-score secara visual antara KNN ($K=9$) dan Gaussian Naive Bayes.
   ![Metrics Comparison](reports/metrics_comparison.png)

3. **Computation Time Comparison**
   
   Perbandingan waktu prediksi yang menunjukkan efisiensi luar biasa dari model Naive Bayes dibandingkan KNN.
   ![Computation Time](reports/time_comparison.png)

4. **KNN Hyperparameter Tuning Chart**
   
   Visualisasi pengaruh variasi nilai $K$ terhadap performa model KNN.
   ![KNN Tuning](reports/knn_k_tuning.png)

---

## 6. KESIMPULAN & SARAN

### A. Kesimpulan
1. Kedua model klasifikasi yang diimplementasikan dari awal (*from scratch*) tanpa pustaka eksternal machine learning berhasil memprediksi cuaca harian Kelapa Gading dengan kinerja yang sangat memuaskan (akurasi $>81\%$).
2. Model **KNN dengan $K = 9$** menghasilkan performa prediksi terbaik secara keseluruhan dengan Akurasi **82,10%** dan F1-Score **84,57%**. Namun, model ini membutuhkan memori dan daya komputasi waktu prediksi yang signifikan karena kompleksitas pencariannya yang bertumpu pada seluruh data latih ($O(N)$ per sampel uji).
3. Model **Gaussian Naive Bayes** menawarkan performa yang hampir setara (Akurasi **81,66%** dan F1-Score **84,28%**) namun dengan **kecepatan prediksi yang jauh lebih cepat (32x lebih cepat)**. Naive Bayes adalah pilihan yang sangat optimal untuk sistem produksi yang membutuhkan respons cepat atau dijalankan pada perangkat dengan sumber daya komputasi terbatas.

### B. Saran Pengembangan
1. **Seleksi Fitur (Feature Selection)**: Disarankan melakukan analisis korelasi fitur untuk mereduksi dimensi fitur dari 21 kolom menjadi kolom-kolom yang paling berpengaruh saja (seperti tutupan awan dan kelembapan). Reduksi dimensi ini akan sangat mempercepat waktu prediksi KNN.
2. **Kombinasi Jarak Lain pada KNN**: Mengeksplorasi penggunaan jarak Manhattan atau jarak Minkowski selain Euclidean untuk melihat efeknya terhadap kinerja KNN pada ruang dimensi tinggi.
3. **Pembobotan Jarak pada KNN**: Menerapkan metode *weighted KNN* (di mana voting tetangga terdekat diberi bobot berbanding terbalik dengan jaraknya) untuk mengatasi bias persebaran kelas.
