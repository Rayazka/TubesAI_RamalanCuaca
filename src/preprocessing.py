import csv
import random
import numpy as np

def load_data(file_path, target_threshold=3.0):
    """
    Membaca dataset cuaca dari berkas CSV, memisahkan fitur dan target,
    serta membuang kolom kebocoran data. Hasil akhir dikonversi ke array NumPy.
    
    Parameter:
        file_path (str): Jalur berkas CSV dataset.
        target_threshold (float): Batas curah hujan (mm) untuk pelabelan biner Hujan (1).
        
    Return:
        X (np.ndarray): Matriks fitur prediktor berdimensi (n_samples, n_features).
        y (np.ndarray): Vektor label target biner (n_samples,).
        feature_names (list): Daftar nama kolom fitur prediktor.
    """
    X = []
    y = []
    
    # Membaca berkas CSV menggunakan standard library csv
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Bersihkan whitespace di sekitar nama kolom
        header = [col.strip() for col in header]
        
        # Dapatkan indeks kolom target curah hujan
        try:
            target_idx = header.index("precipitation_sum (mm)")
        except ValueError:
            raise ValueError("Kolom target 'precipitation_sum (mm)' tidak ditemukan.")
            
        # Kolom kebocoran data (leakage) yang harus dieliminasi
        leakage_cols = ["time", "precipitation_sum (mm)", "precipitation_hours (h)"]
        drop_indices = set()
        for col_name in leakage_cols:
            if col_name in header:
                drop_indices.add(header.index(col_name))
                
        # Dapatkan indeks dan nama fitur yang digunakan sebagai prediktor
        feature_indices = [i for i in range(len(header)) if i not in drop_indices]
        feature_names = [header[i] for i in feature_indices]
        
        # Iterasi membaca baris data
        for row in reader:
            if not row:
                continue
            try:
                # Label biner curah hujan
                precip = float(row[target_idx])
                label = 1 if precip > target_threshold else 0
                
                # Fitur numerik
                features = [float(row[i]) for i in feature_indices]
                
                X.append(features)
                y.append(label)
            except ValueError:
                continue
                
    # Mengonversi list Python menjadi array NumPy agar komputasi lebih cepat
    return np.array(X), np.array(y), feature_names

def fit_min_max(X_train):
    """
    Menghitung batas minimum dan maksimum untuk setiap fitur pada data training menggunakan NumPy.
    
    Parameter:
        X_train (np.ndarray): Array dua dimensi data training.
        
    Return:
        min_val (np.ndarray): Nilai minimum tiap fitur (kolom).
        max_val (np.ndarray): Nilai maksimum tiap fitur (kolom).
    """
    # np.min(..., axis=0) mencari nilai minimal untuk setiap kolom
    min_val = np.min(X_train, axis=0)
    max_val = np.max(X_train, axis=0)
    return min_val, max_val

def transform_min_max(X, scale_params):
    """
    Menormalisasikan matriks fitur menggunakan NumPy secara vektorisasi tanpa loop.
    Rumus: (X - min) / (max - min)
    
    Parameter:
        X (np.ndarray): Array dua dimensi yang akan dinormalisasi.
        scale_params (tuple): Pasangan (min_val, max_val) hasil dari fit_min_max.
        
    Return:
        X_scaled (np.ndarray): Array hasil normalisasi berskala [0, 1].
    """
    min_val, max_val = scale_params
    
    # Hitung selisih rentang nilai (max - min)
    range_val = max_val - min_val
    
    # Cegah pembagian dengan nol dengan mengganti range bernilai 0 menjadi 1.0
    range_val = np.where(range_val == 0.0, 1.0, range_val)
    
    # Lakukan kalkulasi secara paralel untuk semua elemen matriks (Broadcasting)
    return (X - min_val) / range_val

def train_test_split(X, y, test_size=0.2, seed=42):
    """
    Membagi dataset menjadi set training dan testing secara acak berbasis NumPy.
    
    Parameter:
        X (np.ndarray): Array fitur lengkap.
        y (np.ndarray): Array label target lengkap.
        test_size (float): Proporsi ukuran data uji (uji / total).
        seed (int): Seed generator acak agar hasil pengacakan konsisten.
        
    Return:
        X_train, X_test, y_train, y_test (semuanya bertipe np.ndarray)
    """
    n = len(X)
    indices = np.arange(n)
    
    # Inisialisasi generator acak modern NumPy dengan seed tetap
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    # Tentukan indeks pembatas pembagian
    split_idx = int(n * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Saring data latih dan uji menggunakan boolean/fancy indexing NumPy
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
