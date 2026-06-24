import time
# Mengimpor modul-modul custom yang telah kita bangun secara manual di folder src dengan optimasi NumPy
from src.preprocessing import load_data, fit_min_max, transform_min_max, train_test_split
from src.model_knn import KNNClassifier
from src.model_naive_bayes import GaussianNaiveBayes
from src.evaluation import calculate_metrics, print_metrics
# Mengimpor modul visualisasi diagram baru
from src.visualization import (
    plot_confusion_matrices,
    plot_metrics_comparison,
    plot_time_comparison,
    plot_knn_tuning
)

def predict_and_display_days(X_test, X_test_scaled, y_test, knn, gnb, feature_names, num_days=7):
    """
    Memprediksi sejumlah hari pertama dari data uji dan menampilkan hasil prediksi
    beserta persentase keyakinan (probabilitas) untuk masing-masing model ke konsol.
    """
    print("\n==================================================")
    print(f"    HASIL PREDIKSI UNTUK {num_days} HARI PERTAMA (DATA UJI) ")
    print("==================================================")
    
    # Ambil sampel sebanyak num_days
    samples_raw = X_test[:num_days]
    samples_scaled = X_test_scaled[:num_days]
    actual_labels = y_test[:num_days]
    
    # Prediksi kelas dan hitung probabilitas
    knn_preds = knn.predict(samples_scaled)
    knn_probas = knn.predict_proba(samples_scaled)
    
    gnb_preds = gnb.predict(samples_raw)
    gnb_probas = gnb.predict_proba(samples_raw)
    
    for i in range(num_days):
        print(f"\nHari ke-{i+1}:")
        
        # Tampilkan beberapa fitur cuaca penting sebagai info pendukung
        # 'temperature_2m_mean (°C)' -> index 5
        # 'relative_humidity_2m_mean (%)' -> index 6
        # 'cloud_cover_mean (%)' -> index 7
        # 'wind_speed_10m_mean (km/h)' -> index 16
        temp = samples_raw[i][5]
        humid = samples_raw[i][6]
        cloud = samples_raw[i][7]
        wind = samples_raw[i][16]
        print(f"      [Info Cuaca] Rata-rata Suhu: {temp:.1f}°C | Kelembapan: {humid:.1f}% | Awan: {cloud:.1f}% | Kec. Angin: {wind:.1f} km/h")
        
        # Label Aktual
        actual_str = "HUJAN" if actual_labels[i] == 1 else "TIDAK HUJAN"
        print(f"      [Aktual]     : {actual_str}")
        
        # Hasil KNN
        knn_pred = knn_preds[i]
        knn_pred_str = "HUJAN" if knn_pred == 1 else "TIDAK HUJAN"
        knn_conf = knn_probas[i][knn_pred] * 100
        print(f"      [Model KNN]  : {knn_pred_str} (Persentase Keyakinan: {knn_conf:.1f}%)")
        
        # Hasil Naive Bayes
        gnb_pred = gnb_preds[i]
        gnb_pred_str = "HUJAN" if gnb_pred == 1 else "TIDAK HUJAN"
        gnb_conf = gnb_probas[i][gnb_pred] * 100
        print(f"      [Model GNB]  : {gnb_pred_str} (Persentase Keyakinan: {gnb_conf:.1f}%)")
        
    print("\n==================================================")

def main():
    print("==================================================")
    print("   MEMULAI PIPELINE PREDIKSI CUACA KELAPA GADING ")
    print("==================================================")
    
    # 1. Langkah Pemuatan Data (Data Loading)
    data_path = "data/cuaca-harian-dki2-kelapagading.csv"
    print(f"\n[1/5] Memuat data dari berkas: {data_path}...")
    try:
        # Pemuatan data dengan batas threshold curah hujan > 3.0 mm untuk kelas Hujan
        # Output X dan y berupa array NumPy
        X, y, feature_names = load_data(data_path, target_threshold=3.0)
        print(f"      -> Berhasil memuat {len(X)} sampel data dengan {len(feature_names)} kolom fitur prediktor.")
        print(f"      -> Sampel Fitur: {', '.join(feature_names[:4])} ... (+ {len(feature_names)-4} fitur lainnya)")
    except Exception as e:
        print(f"      -> ERROR gagal membaca data: {e}")
        return

    # 2. Langkah Pembagian Data (Train-Test Split)
    # Memisahkan data dengan porsi 80% untuk latihan (train) dan 20% untuk pengujian (test)
    print("\n[2/5] Melakukan Train-Test Split (Rasio 80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    print(f"      -> Jumlah sampel Data Latih (Train): {len(X_train)} sampel")
    print(f"      -> Jumlah sampel Data Uji (Test)  : {len(X_test)} sampel")
    
    # Menghitung frekuensi kemunculan kelas target untuk memantau keseimbangan kelas
    train_rain = sum(y_train)
    train_dry = len(y_train) - train_rain
    test_rain = sum(y_test)
    test_dry = len(y_test) - test_rain
    print(f"      -> Distribusi Kelas Train  : Hujan = {train_rain} ({train_rain/len(y_train)*100:.1f}%), Tidak Hujan = {train_dry} ({train_dry/len(y_train)*100:.1f}%)")
    print(f"      -> Distribusi Kelas Test   : Hujan = {test_rain} ({test_rain/len(y_test)*100:.1f}%), Tidak Hujan = {test_dry} ({test_dry/len(y_test)*100:.1f}%)")

    # 3. Langkah Normalisasi Fitur (Feature Scaling)
    # Melakukan penyetaraan skala fitur menggunakan NumPy ke dalam rentang [0, 1] secara paralel
    print("\n[3/5] Melakukan Normalisasi Fitur (Min-Max Scaling [0,1]) secara manual dengan NumPy...")
    # Hitung nilai min-max berdasarkan X_train
    scale_params = fit_min_max(X_train)
    # Transformasikan X_train dan X_test secara vektorisasi (Broadcasting)
    X_train_scaled = transform_min_max(X_train, scale_params)
    X_test_scaled = transform_min_max(X_test, scale_params)
    print("      -> Normalisasi selesai tanpa terjadi kebocoran data uji (data leakage).")

    # 4. Langkah Eksperimen & Evaluasi Model KNN Classifier
    # Menguji kinerja KNN dengan beberapa variasi nilai K untuk mencari performa optimal
    print("\n[4/5] Melatih dan Menguji Model KNN Classifier (From Scratch - Optimasi NumPy)...")
    k_values = [1, 3, 5, 7, 9]
    accuracies = []
    f1_scores = []
    
    best_k = 5
    best_f1 = 0.0
    best_metrics = None
    best_pred_time = 0.0
    
    for k in k_values:
        print(f"      -> Menguji KNN dengan K = {k}...")
        knn = KNNClassifier(k=k)
        
        # Latih model (menyimpan data latih berskala)
        start_train = time.time()
        knn.fit(X_train_scaled, y_train)
        end_train = time.time()
        
        # Prediksi label menggunakan kalkulasi jarak Euclidean NumPy terakselerasi
        start_pred = time.time()
        y_pred = knn.predict(X_test_scaled)
        end_pred = time.time()
        
        # Hitung metrik evaluasi hasil prediksi
        metrics = calculate_metrics(y_test, y_pred)
        train_time = end_train - start_train
        pred_time = end_pred - start_pred
        
        accuracies.append(metrics['accuracy'])
        f1_scores.append(metrics['f1_score'])
        
        print(f"         Akurasi: {metrics['accuracy']*100:.2f}% | F1-Score: {metrics['f1_score']*100:.2f}% | Waktu Prediksi: {pred_time:.3f} detik")
        
        # Melacak nilai K terbaik yang menghasilkan F1-score tertinggi
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_k = k
            best_metrics = metrics
            best_pred_time = pred_time
            
    print(f"\n      => Hasil Terbaik KNN diperoleh pada nilai K = {best_k}")
    # Cetak laporan metrik lengkap untuk KNN terbaik
    print_metrics(best_metrics, title=f"KNN Classifier Terbaik (K = {best_k})")
    print(f"      => Waktu Pengujian KNN: {best_pred_time:.3f} detik")

    # 5. Langkah Eksperimen & Evaluasi Model Gaussian Naive Bayes
    print("\n[5/5] Melatih dan Menguji Model Gaussian Naive Bayes (From Scratch - Optimasi NumPy)...")
    gnb = GaussianNaiveBayes()
    
    # Latih model Naive Bayes (menghitung prior, mean, dan varians dari data latih dengan NumPy)
    start_train_nb = time.time()
    gnb.fit(X_train, y_train)
    end_train_nb = time.time()
    
    # Prediksi label menggunakan kalkulasi log-probabilitas posterior terakselerasi
    start_pred_nb = time.time()
    y_pred_nb = gnb.predict(X_test)
    end_pred_nb = time.time()
    
    nb_train_time = end_train_nb - start_train_nb
    nb_pred_time = end_pred_nb - start_pred_nb
    
    # Hitung dan cetak metrik evaluasi hasil prediksi Naive Bayes
    metrics_nb = calculate_metrics(y_test, y_pred_nb)
    print_metrics(metrics_nb, title="Gaussian Naive Bayes")
    print(f"      => Waktu Latih NB  : {nb_train_time:.6f} detik")
    print(f"      => Waktu Prediksi NB: {nb_pred_time:.6f} detik")
    
    # 6. Pembuatan dan Penyimpanan Diagram/Visualisasi
    print("\n[Visualisasi] Menggambar dan menyimpan grafik laporan ke folder 'reports/'...")
    try:
        # A. Plot Confusion Matrices
        plot_confusion_matrices(
            best_metrics['confusion_matrix'], 
            metrics_nb['confusion_matrix']
        )
        
        # B. Plot Metrics Comparison Chart
        plot_metrics_comparison(best_metrics, metrics_nb)
        
        # C. Plot Computation Time Chart
        plot_time_comparison(best_pred_time, (nb_train_time, nb_pred_time))
        
        # D. Plot KNN Hyperparameter Tuning Chart
        plot_knn_tuning(k_values, accuracies, f1_scores)
        
        print("      -> Semua diagram visualisasi berhasil disimpan di folder 'reports/'.")
    except Exception as e:
        print(f"      -> ERROR gagal menggambar visualisasi: {e}")
    
    # 7. Memprediksi sejumlah hari pertama dari data uji (Default: 7 hari)
    # Buat instansiasi KNN terbaik dan latih kembali untuk pengujian sampel
    best_knn = KNNClassifier(k=best_k)
    best_knn.fit(X_train_scaled, y_train)
    predict_and_display_days(X_test, X_test_scaled, y_test, best_knn, gnb, feature_names, num_days=7)

if __name__ == "__main__":
    main()
