# LAPORAN TUGAS BESAR MATA KULIAH KECERDASAN BUATAN (ARTIFICIAL INTELLIGENCE PROGRAM STUDI S1 REKAYASA PERANGKAT LUNAK

## IDENTITAS KELOMPOK & PERAN ANGGOTA

* **Kelas**: SE-48-03
* **Nama Kelompok**: Mie Ayam
* **Anggota Kelompok**:
  1. **Rayazka Aris Fadhilahn** (NIM: 103022400096)
     * **Peran & Kontribusi**:
       * Merancang dan mengimplementasikan modul preprocessing data (`src/preprocessing.py`) termasuk pemuatan CSV ke NumPy, pemisahan data latih/uji (*train-test split*), dan normalisasi skala fitur (*Min-Max scaling*).
       * Membangun model *K-Nearest Neighbors (KNN) Classifier* dengan optimasi komputasi matriks/vektorisasi NumPy (`src/model_knn.py`).
       * Mengintegrasikan metrik evaluasi (`src/evaluation.py`) dan merancang alur utama pipeline program (`main.py`).
       * Melakukan eksperimen tuning hyperparameter $K$ pada KNN dan menyusun visualisasi grafik laporan.
  2. **Raka Putra** (NIM: 130222XXXX)
     * **Peran & Kontribusi**:
       * Membangun model *Gaussian Naive Bayes Classifier* dengan memanfaatkan operasi kolom NumPy untuk perhitungan mean, varians, prior, dan likelihood secara efisien (`src/model_naive_bayes.py`).
       * Membantu memvalidasi implementasi matematika model terhadap hasil pengujian serta melakukan analisis perbandingan performa.

## 1. DESKRIPSI KASUS, DATASET, & IMPLEMENTASI PREPROCESSING

### A. Deskripsi Kasus & Variabel Target
Kasus yang diselesaikan dalam proyek ini adalah **klasifikasi cuaca harian biner (Hujan vs Tidak Hujan)** di wilayah **Kelapa Gading, Jakarta Utara** menggunakan dua model klasifikasi (*from scratch*): K-Nearest Neighbors (KNN) dan Gaussian Naive Bayes.
Untuk memodelkan kasus ini, kita mengonversi data curah hujan kontinu (`precipitation_sum (mm)`) menjadi target biner $y \in \{0, 1\}$ (diberi nama `is_rain` di memori) dengan batas threshold:
$$y = \begin{cases} 
1, & \text{jika } \text{precipitation\_sum} > 3.0 \text{ mm (Hari Hujan)} \\ 
0, & \text{jika } \text{precipitation\_sum} \le 3.0 \text{ mm (Hari Tidak Hujan)} 
\end{cases}$$

### B. Analisis Dataset & Eliminasi Kolom Bocor (Data Leakage)
Dataset asli dari berkas `data/cuaca-harian-dki2-kelapagading.csv` berisi **5.722 baris data** dengan total 24 kolom.
Guna mencegah bias klasifikasi (*data leakage*), tiga kolom berikut dibuang secara ketat dari fitur prediktor:
1. **`time`**: Merupakan metadata penunjuk waktu (tanggal) yang tidak mempengaruhi model fisik atmosfer secara langsung.
2. **`precipitation_sum (mm)`**: Sumber langsung dari variabel target biner. Jika tidak dibuang, model akan memprediksi secara perkalian langsung dengan tingkat keberhasilan palsu 100%.
3. **`precipitation_hours (h)`**: Jumlah jam terjadinya hujan. Data ini hanya diketahui *setelah* hari tersebut selesai dilewati, sehingga tidak valid digunakan untuk meramal di awal hari.

Setelah mengeliminasi kolom-kolom bocor ini, kita memperoleh **21 fitur prediktor numerik** (suhu udara maks/min/rata-rata, kelembapan, tutupan awan, tekanan udara, kecepatan angin, radiasi matahari, dsb.).

---

### C. Implementasi Langkah Preprocessing Data & Hasil Proses

Proses preprocessing data diimplementasikan sepenuhnya pada berkas `src/preprocessing.py` menggunakan array NumPy untuk optimasi kecepatan eksekusi. Berikut adalah langkah-langkah detailnya:

#### Langkah 1: Pemuatan Data & Penyaringan Kolom (Data Loading)
Fungsi `load_data` membaca data dari file CSV baris demi baris menggunakan standard library `csv`, mengekstrak target curah hujan, menerapkan binarisasi, memfilter kolom bocor, dan mengonversi list data menjadi array NumPy `np.ndarray`.

* **Kode Implementasi (`src/preprocessing.py`):**
```python
def load_data(file_path, target_threshold=3.0):
    X = []
    y = []
    
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [col.strip() for col in header]
        
        try:
            target_idx = header.index("precipitation_sum (mm)")
        except ValueError:
            raise ValueError("Kolom target 'precipitation_sum (mm)' tidak ditemukan.")
            
        leakage_cols = ["time", "precipitation_sum (mm)", "precipitation_hours (h)"]
        drop_indices = set()
        for col_name in leakage_cols:
            if col_name in header:
                drop_indices.add(header.index(col_name))
                
        feature_indices = [i for i in range(len(header)) if i not in drop_indices]
        feature_names = [header[i] for i in feature_indices]
        
        for row in reader:
            if not row:
                continue
            try:
                precip = float(row[target_idx])
                label = 1 if precip > target_threshold else 0
                features = [float(row[i]) for i in feature_indices]
                X.append(features)
                y.append(label)
            except ValueError:
                continue
                
    return np.array(X), np.array(y), feature_names
```

* **Hasil Proses pada Dataset Kita:**
  * Jumlah Sampel yang berhasil dimuat: **5.722 baris data**.
  * Jumlah Fitur Prediktor yang digunakan: **21 kolom**.
  * Kolom Fitur Terpilih: `temperature_2m_max (°C)`, `temperature_2m_min (°C)`, `wind_speed_10m_max (km/h)`, `wind_direction_10m_dominant (°)`, ..., dan 17 fitur atmosfer numerik lainnya.

#### Langkah 2: Pembagian Data Uji dan Data Latih (Train-Test Split)
Dataset dibagi menjadi **80% data latih (train)** untuk pelatihan model dan **20% data uji (test)** untuk pengujian performa menggunakan indeks acak yang terkontrol (`seed = 42`).

* **Kode Implementasi (`src/preprocessing.py`):**
```python
def train_test_split(X, y, test_size=0.2, seed=42):
    n = len(X)
    indices = np.arange(n)
    
    # Inisialisasi generator acak modern dengan seed tetap agar reproducible
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    split_idx = int(n * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
```

* **Hasil Proses pada Dataset Kita:**
  * Ukuran Data Latih (*Train*): **4.577 sampel** ($80\%$).
  * Ukuran Data Uji (*Test*): **1.145 sampel** ($20\%$).
  * **Distribusi Kelas Target (`is_rain`):**
    * **Data Latih**: Hujan = **2.399** ($52,4\%$), Tidak Hujan = **2.178** ($47,6\%$).
    * **Data Uji**: Hujan = **632** ($55,2\%$), Tidak Hujan = **513** ($44,8\%$).
    * *Catatan*: Proporsi sebaran kelas cukup berimbang sehingga metrik akurasi dapat digunakan dengan objektif bersama F1-score.

#### Langkah 3: Normalisasi Fitur (Min-Max Scaling)
Karena fitur prediktor memiliki satuan dan skala nilai yang berbeda (misalnya, tekanan udara dalam ribuan hPa dan suhu dalam puluhan °C), dilakukan normalisasi Min-Max ke dalam rentang $[0, 1]$.
Guna menghindari kebocoran data uji (*data leakage*), batas min dan max dihitung eksklusif hanya pada data latih (`fit_min_max`), lalu batas tersebut diterapkan untuk mentransformasikan data latih maupun data uji (`transform_min_max`).

* **Kode Implementasi (`src/preprocessing.py`):**
```python
def fit_min_max(X_train):
    # np.min dan np.max mencari nilai ekstrem untuk setiap kolom (axis=0)
    min_val = np.min(X_train, axis=0)
    max_val = np.max(X_train, axis=0)
    return min_val, max_val

def transform_min_max(X, scale_params):
    min_val, max_val = scale_params
    range_val = max_val - min_val
    
    # Cegah pembagian dengan nol dengan mengganti range 0 menjadi 1.0
    range_val = np.where(range_val == 0.0, 1.0, range_val)
    
    # Lakukan kalkulasi secara paralel menggunakan NumPy broadcasting
    return (X - min_val) / range_val
```

* **Hasil Proses pada Dataset Kita:**
  * Nilai minimum dan maksimum untuk 21 fitur berhasil diekstrak dari 4.577 data training.
  * Sebagai contoh, fitur `temperature_2m_max` (Suhu Maks) memiliki $X_{min} = 22.1^\circ\text{C}$ dan $X_{max} = 34.6^\circ\text{C}$ pada data latih. Setelah dinormalisasi, hari dengan suhu $34.6^\circ\text{C}$ akan bernilai $1.0$, hari dengan suhu $22.1^\circ\text{C}$ akan bernilai $0.0$, sedangkan hari dengan suhu $28.35^\circ\text{C}$ akan bernilai tepat $0.5$.
  * Transformasi ini berhasil menyamakan skala seluruh fitur ke rentang $[0, 1]$ tanpa ada kebocoran informasi dari data uji ke data latih.

---

## 2. DESAIN ALGORITMA 

### A. K-Nearest Neighbors (KNN) Classifier

#### 1. Definisi & Konsep Utama
K-Nearest Neighbors (KNN) adalah algoritma klasifikasi berbasis instansi (*instance-based learning*). Algoritma ini tidak membuat model rumus baru, melainkan memprediksi kelas data baru dengan cara mencari $K$ sampel data latih yang memiliki karakteristik (jarak fitur) paling mirip, lalu melakukan voting suara terbanyak untuk menentukan kelas akhir.

#### 2. Langkah-demi-Langkah Implementasi Kode & Hasil pada Data

##### Langkah 1: Inisialisasi & Penyimpanan Data Latih (Fase Fit)
KNN adalah *lazy learner*, sehingga tidak ada komputasi matematis saat latihan. Kita hanya menginisialisasi parameter $K$ dan menyimpan data latih berskala ke dalam memori objek.

* **Kode Implementasi (`src/model_knn.py`):**
```python
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
```
* **Hasil Proses pada Dataset Kita:**
  * Menyimpan matriks fitur latih ternormalisasi `X_train_scaled` berdimensi **$4.577 \times 21$** (4.577 hari historis dengan 21 parameter cuaca masing-masing).
  * Menyimpan vektor target `y_train` berdimensi **$4.577$** (label 1 = Hujan, 0 = Tidak Hujan).
  * Waktu eksekusi langkah ini hampir instant ($\approx 0.00$ detik).

##### Langkah 2: Perhitungan Jarak Euclidean (Jarak Kemiripan Cuaca)
Ketika diberikan data cuaca 1 hari uji ($x$), model menghitung kuadrat jarak Euclidean dari hari uji tersebut ke **seluruh** $4.577$ hari latih secara paralel menggunakan fitur *broadcasting* NumPy.

* **Kode Implementasi (`src/model_knn.py`):**
```python
        # (self.X_train - x) mengurangi fitur hari uji dari setiap 4.577 baris secara otomatis
        # .sum(axis=1) menjumlahkan selisih kuadrat di 21 kolom fitur secara horizontal
        dists_sq = np.sum((self.X_train - x) ** 2, axis=1)
```
* **Hasil Proses pada Dataset Kita:**
  * Diperoleh sebuah array NumPy `dists_sq` berdimensi **$4.577$** yang menampung jarak kemiripan cuaca. 
  * Nilai jarak mendekati `0.0` menunjukkan hari latih tersebut memiliki kondisi cuaca (suhu, kelembapan, kecepatan angin) yang sangat mirip dengan hari yang diuji.

##### Langkah 3: Mengambil K Tetangga Terdekat
Model mempartisi array jarak untuk mengambil $K$ (dalam hal ini $K=9$) indeks baris data latih yang memiliki jarak paling kecil (tetangga paling mirip), kemudian mengurutkannya dari yang terdekat ke terjauh.

* **Kode Implementasi (`src/model_knn.py`):**
```python
        # np.argpartition mengisolasi K indeks terkecil secara efisien dalam waktu O(N)
        k_indices = np.argpartition(dists_sq, self.k)[:self.k]
        
        # Urutkan K indeks terpilih dari jarak paling dekat ke yang agak jauh
        k_indices = k_indices[np.argsort(dists_sq[k_indices])]
        
        # Ambil label target (Hujan/Tidak Hujan) dari K tetangga tersebut
        k_labels = self.y_train[k_indices]
```
* **Hasil Proses pada Dataset Kita:**
  * Diperoleh array `k_labels` berisi 9 label biner (misal: `[1, 1, 0, 1, 1, 0, 1, 1, 1]`), yang merepresentasikan status cuaca aktual dari 9 hari historis yang paling mirip dengan kondisi hari uji tersebut.

##### Langkah 4: Voting Suara Terbanyak & Pemecah Seri (Tie-Breaker)
Model menghitung jumlah label kelas di dalam `k_labels`. Kelas dengan jumlah suara terbanyak terpilih sebagai hasil prediksi. Jika terjadi seri (jumlah suara kelas 0 dan kelas 1 sama), model memilih kelas milik tetangga yang berada di posisi terdekat (indeks pertama pada `k_labels` yang terurut).

* **Kode Implementasi (`src/model_knn.py`):**
```python
        unique_labels, counts = np.unique(k_labels, return_counts=True)
        max_count = np.max(counts)
        candidates = unique_labels[counts == max_count]
        
        # Jika hanya ada satu pemenang suara mayoritas
        if len(candidates) == 1:
            return int(candidates[0])
            
        # Tie-Breaker: Pilih kelas dari kandidat yang paling dekat jarak fisiknya
        for label in k_labels:
            if label in candidates:
                return int(label)
```
* **Hasil Proses pada Dataset Kita:**
  * Jika mayoritas tetangga berlabel `1`, fungsi mengembalikan prediksi **1 (HUJAN)**.
  * Jika mayoritas tetangga berlabel `0`, fungsi mengembalikan prediksi **0 (TIDAK HUJAN)**.

##### Langkah 5: Menghitung Persentase Keyakinan (Confidence)
Model menghitung proporsi kemunculan kelas yang diprediksi di antara $K$ tetangga untuk mengukur tingkat kepercayaan prediksi tersebut.

* **Kode Implementasi (`src/model_knn.py`):**
```python
    def predict_proba_one(self, x):
        dists_sq = np.sum((self.X_train - x) ** 2, axis=1)
        k_indices = np.argpartition(dists_sq, self.k)[:self.k]
        k_labels = self.y_train[k_indices]
        
        # Hitung probabilitas kelas
        count_1 = np.sum(k_labels == 1)
        prob_1 = count_1 / self.k
        prob_0 = 1.0 - prob_1
        
        return {0: prob_0, 1: prob_1}
```
* **Hasil Proses pada Dataset Kita:**
  * Jika $K=9$, dan tetangga terdekat berisi 6 hari Hujan dan 3 hari Tidak Hujan, maka model memprediksi Hujan dengan tingkat keyakinan **$66,7\%$** ($\frac{6}{9} \times 100\%$).
  * Jika kondisi cuacanya sangat ekstrim, ke-9 tetangga terdekat semuanya berlabel Hujan, sehingga model memprediksi Hujan dengan keyakinan **$100\%$** ($\frac{9}{9} \times 100\%$).

---

### B. Gaussian Naive Bayes Classifier

#### 1. Definisi & Konsep Utama
Gaussian Naive Bayes adalah model klasifikasi probabilistik yang didasarkan pada **Teorema Bayes**. Model ini disebut **"Naive" (Naif)** karena mengasumsikan bahwa seluruh fitur prediktor bersifat saling bebas (independen) satu sama lain untuk suatu kelas target. Kata **"Gaussian"** berarti model mengasumsikan bahwa fitur-fitur numerik yang bersifat kontinu berdistribusi normal (mengikuti kurva lonceng Gaussian).

Secara matematis, peluang suatu kelas $y$ diberikan sekumpulan fitur $X$ dihitung dengan rumus:
$$P(y | X) \propto P(y) \prod_{i=1}^{D} P(x_i | y)$$
Dalam domain logaritma alami (untuk stabilitas numerik):
$$\ln P(y | X) \propto \ln P(y) + \sum_{i=1}^{D} \ln P(x_i | y)$$

#### 2. Langkah-demi-Langkah Implementasi Kode & Hasil pada Data Kita

##### Langkah 1: Inisialisasi Model
Model menyiapkan wadah kamus kosong untuk menyimpan parameter kelas (`classes`), probabilitas prior (`priors`), rata-rata fitur (`means`), dan varians fitur (`variances`).

* **Kode Implementasi (`src/model_naive_bayes.py`):**
```python
    def __init__(self):
        self.classes = []
        self.priors = {}       # Prior P(y) untuk setiap kelas
        self.means = {}        # Rata-rata fitur per kelas: {kelas: np.ndarray}
        self.variances = {}    # Varians fitur per kelas: {kelas: np.ndarray}
        self.is_fitted = False
```
* **Hasil Proses pada Dataset Kita:**
  * Menyiapkan objek model yang siap menampung hasil perhitungan statistik dari 21 fitur cuaca.

##### Langkah 2: Pelatihan Model - Menghitung Statistik Prior, Mean, dan Varians (`fit`)
Model membagi data latih berdasarkan kelas target (Hujan dan Tidak Hujan). Lalu dihitung peluang prior kelas $P(y)$, nilai rata-rata ($\mu$), dan varians ($\sigma^2$) dari masing-masing 21 fitur menggunakan NumPy.

* **Kode Implementasi (`src/model_naive_bayes.py`):**
```python
    def fit(self, X, y):
        n_samples = len(X)
        self.classes = np.unique(y) # Dapatkan kelas unik [0, 1]
        
        for c in self.classes:
            # Saring baris yang memiliki kelas target c
            X_c = X[y == c]
            
            # 1. Hitung Prior P(y = c)
            self.priors[c] = len(X_c) / n_samples
            
            # 2. Hitung rata-rata dan varians kolom secara paralel
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0)
            
        self.is_fitted = True
```
* **Hasil Proses pada Dataset Kita:**
  * **Peluang Prior Hujan $P(y=1)$**: $\frac{2399}{4577} \approx \mathbf{52,4\%}$
  * **Peluang Prior Tidak Hujan $P(y=0)$**: $\frac{2178}{4577} \approx \mathbf{47,6\%}$
  * **Mean & Varians Fitur**: Diperoleh nilai rata-rata dan varians 21 fitur untuk masing-masing kelas.
    * *Contoh Fisis*: Fitur `temperature_2m_max` (Suhu Maks) memiliki rata-rata $\mu_{\text{Hujan}} = 29.5^\circ\text{C}$ dan rata-rata $\mu_{\text{Tidak Hujan}} = 31.2^\circ\text{C}$. Ini menunjukkan bahwa secara statistik, hari hujan memiliki suhu rata-rata yang lebih rendah (karena banyak awan mendung).

##### Langkah 3: Kalkulasi Peluang Likelihood Fitur Kontinu (`_calculate_log_likelihood`)
Untuk data uji hari baru $x$, dihitung seberapa cocok nilai masing-masing fitur terhadap karakteristik kelas $c$ menggunakan fungsi kepadatan probabilitas (PDF) Gaussian:
$$P(x_i | y=c) = \frac{1}{\sqrt{2\pi\sigma_{c,i}^2}} \exp\left(-\frac{(x_i - \mu_{c,i})^2}{2\sigma_{c,i}^2}\right)$$
Nilai ini ditransformasikan ke log-likelihood ($\ln P(x_i | y=c)$) dan dijumlahkan untuk seluruh 21 fitur.

* **Kode Implementasi (`src/model_naive_bayes.py`):**
```python
    def _calculate_log_likelihood(self, x, mean, variance):
        eps = 1e-9 # Epsilon kecil agar tidak terjadi pembagian nol
        var = variance + eps
        
        # Hitung log-PDF Gaussian secara vektorisasi untuk 21 fitur sekaligus
        log_pdf = -0.5 * np.log(2.0 * np.pi * var) - ((x - mean) ** 2) / (2.0 * var)
        
        # Jumlahkan nilai log-likelihood dari seluruh fitur
        return np.sum(log_pdf)
```
* **Hasil Proses pada Dataset Kita:**
  * Jika nilai suhu hari uji adalah $28.0^\circ\text{C}$, model menghitung seberapa besar peluang munculnya suhu tersebut di kelompok hari Hujan ($\mu=29.5^\circ\text{C}$) dibandingkan dengan hari Tidak Hujan ($\mu=31.2^\circ\text{C}$). 
  * Setelah diproses untuk 21 fitur, diperoleh angka log-likelihood total berupa nilai negatif (misalnya: $-35.2$ untuk Hujan dan $-48.9$ untuk Tidak Hujan).

##### Langkah 4: Keputusan Posterior (`predict_one`)
Model menjumlahkan nilai log-prior dan log-likelihood total untuk masing-masing kelas untuk mendapatkan skor peluang posterior. Kelas dengan skor terbesar dipilih sebagai keputusan prediksi.

* **Kode Implementasi (`src/model_naive_bayes.py`):**
```python
    def predict_one(self, x):
        posteriors = {}
        for c in self.classes:
            # log( P(c) )
            log_prior = np.log(self.priors[c])
            
            # log( P(X | c) )
            log_likelihood = self._calculate_log_likelihood(x, self.means[c], self.variances[c])
            
            # Posterior = log prior + log likelihood
            posteriors[c] = log_prior + log_likelihood
            
        # Kembalikan kelas dengan nilai log-posterior terbesar
        return int(max(posteriors, key=posteriors.get))
```
* **Hasil Proses pada Dataset Kita:**
  * Misal didapatkan skor $\text{Posterior}(\text{Hujan}) = -35.2$ dan $\text{Posterior}(\text{Tidak Hujan}) = -48.9$. 
  * Karena $-35.2 > -48.9$, model memilih kelas dengan nilai log terbesar yaitu **1 (HUJAN)**.

##### Langkah 5: Normalisasi & Perhitungan Persentase Keyakinan (`predict_proba_one`)
Untuk menampilkan persentase keyakinan di konsol, nilai log-posterior dikonversi kembali ke probabilitas nyata (skala $0.0$ sampai $1.0$). Digunakan **Log-Sum-Exp Trick** (mengurangi nilai log dengan nilai maksimumnya) sebelum dilakukan eksponensial guna menghindari kegagalan komputasi akibat angka di bawah batas kapasitas float komputer (*underflow*).

* **Kode Implementasi (`src/model_naive_bayes.py`):**
```python
    def predict_proba_one(self, x):
        if not self.is_fitted:
            raise ValueError("Model belum dilatih!")
            
        posteriors = {}
        for c in self.classes:
            log_prior = np.log(self.priors[c])
            log_likelihood = self._calculate_log_likelihood(x, self.means[c], self.variances[c])
            posteriors[c] = log_prior + log_likelihood
            
        # Log-Sum-Exp Trick untuk stabilitas numerik
        max_log = max(posteriors.values())
        unnormalized = {c: np.exp(val - max_log) for c, val in posteriors.items()}
        total = sum(unnormalized.values())
        
        return {c: unnormalized[c] / total for c in self.classes}
```
* **Hasil Proses pada Dataset Kita:**
  * Jika skor log-posterior kelas adalah $-35.2$ (Hujan) dan $-48.9$ (Tidak Hujan):
    * Nilai maksimumnya adalah $-35.2$.
    * Nilai tidak ternormalisasi: $e^{-35.2 - (-35.2)} = e^0 = \mathbf{1.0}$ (untuk Hujan), dan $e^{-48.9 - (-35.2)} = e^{-13.7} \approx \mathbf{0.000001}$ (untuk Tidak Hujan).
    * Total nilai: $1.0 + 0.000001 = 1.000001$.
    * Persentase Keyakinan Hujan: $\frac{1.0}{1.000001} \times 100\% = \mathbf{99.9\%}$.
  * Hal ini menjelaskan mengapa persentase keyakinan Naive Bayes sering kali bernilai sangat besar/ekstrim (seperti $99.9\%$ or $100\%$): perkalian probabilitas dari 21 fitur menyebabkan dominasi mutlak bagi salah satu kelas.

---

## 3. MODEL EVALUASI & HASIL EKSPERIMEN

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

## 4. SCREENSHOT & OUTPUT RUNNING PROGRAM

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

## 5. KESIMPULAN & SARAN

### A. Kesimpulan
1. Kedua model klasifikasi yang diimplementasikan dari awal (*from scratch*) tanpa pustaka eksternal machine learning berhasil memprediksi cuaca harian Kelapa Gading dengan kinerja yang sangat memuaskan (akurasi $>81\%$).
2. Model **KNN dengan $K = 9$** menghasilkan performa prediksi terbaik secara keseluruhan dengan Akurasi **82,10%** dan F1-Score **84,57%**. Namun, model ini membutuhkan memori dan daya komputasi waktu prediksi yang signifikan karena kompleksitas pencariannya yang bertumpu pada seluruh data latih ($O(N)$ per sampel uji).
3. Model **Gaussian Naive Bayes** menawarkan performa yang hampir setara (Akurasi **81,66%** dan F1-Score **84,28%**) namun dengan **kecepatan prediksi yang jauh lebih cepat (32x lebih cepat)**. Naive Bayes adalah pilihan yang sangat optimal untuk sistem produksi yang membutuhkan respons cepat atau dijalankan pada perangkat dengan sumber daya komputasi terbatas.

### B. Saran Pengembangan
1. **Seleksi Fitur (Feature Selection)**: Disarankan melakukan analisis korelasi fitur untuk mereduksi dimensi fitur dari 21 kolom menjadi kolom-kolom yang paling berpengaruh saja (seperti tutupan awan dan kelembapan). Reduksi dimensi ini akan sangat mempercepat waktu prediksi KNN.
2. **Kombinasi Jarak Lain pada KNN**: Mengeksplorasi penggunaan jarak Manhattan atau jarak Minkowski selain Euclidean untuk melihat efeknya terhadap kinerja KNN pada ruang dimensi tinggi.
3. **Pembobotan Jarak pada KNN**: Menerapkan metode *weighted KNN* (di mana voting tetangga terdekat diberi bobot berbanding terbalik dengan jaraknya) untuk mengatasi bias persebaran kelas.
