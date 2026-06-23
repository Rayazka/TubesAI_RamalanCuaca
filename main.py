import time
from src.preprocessing import load_data, fit_min_max, transform_min_max, train_test_split
from src.model_knn import KNNClassifier
from src.model_naive_bayes import GaussianNaiveBayes
from src.evaluation import calculate_metrics, print_metrics

def main():
    print("==================================================")
    print("   MEMULAI PIPELINE PREDIKSI CUACA KELAPA GADING ")
    print("==================================================")
    
    # 1. Load Data
    data_path = "data/cuaca-harian-dki2-kelapagading.csv"
    print(f"\n[1/5] Memuat data dari: {data_path}...")
    try:
        X, y, feature_names = load_data(data_path, target_threshold=3.0)
        print(f"      -> Berhasil memuat {len(X)} sampel dengan {len(feature_names)} fitur.")
        print(f"      -> Fitur yang digunakan: {', '.join(feature_names[:4])} ... (+ {len(feature_names)-4} fitur lainnya)")
    except Exception as e:
        print(f"      -> ERROR memuat data: {e}")
        return

    # 2. Train-Test Split
    print("\n[2/5] Melakukan Train-Test Split (Rasio 80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    print(f"      -> Jumlah Data Latih (Train): {len(X_train)} sampel")
    print(f"      -> Jumlah Data Uji (Test)  : {len(X_test)} sampel")
    
    # Check class distribution
    train_rain = sum(y_train)
    train_dry = len(y_train) - train_rain
    test_rain = sum(y_test)
    test_dry = len(y_test) - test_rain
    print(f"      -> Distribusi Kelas Train  : Hujan = {train_rain} ({train_rain/len(y_train)*100:.1f}%), Tidak Hujan = {train_dry} ({train_dry/len(y_train)*100:.1f}%)")
    print(f"      -> Distribusi Kelas Test   : Hujan = {test_rain} ({test_rain/len(y_test)*100:.1f}%), Tidak Hujan = {test_dry} ({test_dry/len(y_test)*100:.1f}%)")

    # 3. Min-Max Scaling (Fit on Train, Transform both Train & Test)
    print("\n[3/5] Melakukan Normalisasi Fitur (Min-Max Scaling [0,1]) secara manual...")
    scale_params = fit_min_max(X_train)
    X_train_scaled = transform_min_max(X_train, scale_params)
    X_test_scaled = transform_min_max(X_test, scale_params)
    print("      -> Normalisasi selesai tanpa kebocoran data (data leakage).")

    # 4. Evaluasi KNN untuk berbagai nilai K
    print("\n[4/5] Melatih dan Menguji Model KNN Classifier (From Scratch)...")
    k_values = [1, 3, 5, 7, 9]
    best_k = 5
    best_f1 = 0.0
    
    for k in k_values:
        print(f"      -> Menguji KNN dengan K = {k}...")
        knn = KNNClassifier(k=k)
        
        # Fit (hanya menyimpan data latih)
        start_train = time.time()
        knn.fit(X_train_scaled, y_train)
        end_train = time.time()
        
        # Predict
        start_pred = time.time()
        y_pred = knn.predict(X_test_scaled)
        end_pred = time.time()
        
        # Evaluate
        metrics = calculate_metrics(y_test, y_pred)
        train_time = end_train - start_train
        pred_time = end_pred - start_pred
        
        print(f"         Akurasi: {metrics['accuracy']*100:.2f}% | F1-Score: {metrics['f1_score']*100:.2f}% | Waktu Prediksi: {pred_time:.3f}s")
        
        # Keep track of the best model based on F1-score
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_k = k
            best_metrics = metrics
            best_pred_time = pred_time
            
    print(f"\n      => Hasil Terbaik KNN diperoleh pada K = {best_k}")
    print_metrics(best_metrics, title=f"KNN Classifier Terbaik (K = {best_k})")
    print(f"      => Waktu Pengujian KNN: {best_pred_time:.3f} detik")

    # 5. Memanggil Model Naive Bayes (Sebagai Integrasi Awal / Boilerplate)
    print("\n[5/5] Memanggil Model Gaussian Naive Bayes (Placeholder/Boilerplate)...")
    gnb = GaussianNaiveBayes()
    
    start_train_nb = time.time()
    gnb.fit(X_train, y_train) # Naive Bayes tidak wajib discaling (bisa pakai raw X_train)
    end_train_nb = time.time()
    
    start_pred_nb = time.time()
    y_pred_nb = gnb.predict(X_test)
    end_pred_nb = time.time()
    
    metrics_nb = calculate_metrics(y_test, y_pred_nb)
    print_metrics(metrics_nb, title="Gaussian Naive Bayes (Templat)")
    print(f"      => Waktu Latih NB  : {end_train_nb - start_train_nb:.6f} detik")
    print(f"      => Waktu Prediksi NB: {end_pred_nb - start_pred_nb:.6f} detik")
    print("\n      *Catatan: Model Naive Bayes saat ini mengembalikan prediksi templat (default kelas pertama).")
    print("       Rekan kelompok Anda dapat langsung mengimplementasikannya di 'src/model_naive_bayes.py'")
    print("       dan hasil perbandingannya akan langsung terupdate di sini.")

if __name__ == "__main__":
    main()
