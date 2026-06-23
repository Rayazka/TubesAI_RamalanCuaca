import os
import matplotlib.pyplot as plt
import numpy as np

# Konfigurasi estetika grafik agar terlihat modern dan premium
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['figure.facecolor'] = '#fdfdfd' # Background figure putih bersih
plt.rcParams['axes.facecolor'] = '#ffffff' # Background plot putih
plt.rcParams['axes.edgecolor'] = '#e0e0e0' # Border plot abu-abu tipis
plt.rcParams['grid.color'] = '#f0f0f0'     # Garis grid abu-abu sangat tipis

def create_reports_dir():
    """Membuat folder 'reports' jika belum ada untuk menyimpan file gambar."""
    if not os.path.exists('reports'):
        os.makedirs('reports')

def plot_confusion_matrices(knn_matrix, nb_matrix, save_path='reports/confusion_matrices.png'):
    """
    Menggambar Heatmap Confusion Matrix untuk KNN dan Naive Bayes secara berdampingan.
    
    Parameter:
        knn_matrix (tuple): (tp, tn, fp, fn) untuk model KNN.
        nb_matrix (tuple): (tp, tn, fp, fn) untuk model Naive Bayes.
        save_path (str): Jalur penyimpanan gambar.
    """
    create_reports_dir()
    
    # Membongkar tuple matriks kebingungan
    tp_knn, tn_knn, fp_knn, fn_knn = knn_matrix
    tp_nb, tn_nb, fp_nb, fn_nb = nb_matrix
    
    # Matriks 2D untuk heatmap [[TP, FN], [FP, TN]]
    knn_arr = np.array([[tp_knn, fn_knn], [fp_knn, tn_knn]])
    nb_arr = np.array([[tp_nb, fn_nb], [fp_nb, tn_nb]])
    
    # Membuat figure dengan 1 baris dan 2 kolom subplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
    
    matrices = [knn_arr, nb_arr]
    titles = ['Confusion Matrix: KNN (K=9)', 'Confusion Matrix: Naive Bayes']
    colormaps = [plt.cm.Blues, plt.cm.Greens] # Biru untuk KNN, Hijau untuk Naive Bayes
    
    labels = ['Hujan', 'Tidak Hujan']
    
    for i, ax in enumerate(axes):
        matrix = matrices[i]
        im = ax.imshow(matrix, interpolation='nearest', cmap=colormaps[i])
        ax.set_title(titles[i], fontsize=13, fontweight='bold', pad=15, color='#2c3e50')
        
        # Pengaturan label aksis
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels, fontsize=10, rotation=90, va='center')
        
        ax.set_xlabel('Prediksi Model', fontsize=11, labelpad=10)
        ax.set_ylabel('Aktual Sebenarnya', fontsize=11, labelpad=10)
        
        # Menuliskan angka frekuensi data di dalam setiap sel heatmap
        thresh = matrix.max() / 2.0
        for x in range(2):
            for y in range(2):
                val = matrix[x, y]
                text_color = "white" if val > thresh else "black"
                
                # Memberikan label keterangan jenis prediksi
                if x == 0 and y == 0:
                    lbl = f"True Positive\n({val})"
                elif x == 0 and y == 1:
                    lbl = f"False Negative\n({val})"
                elif x == 1 and y == 0:
                    lbl = f"False Positive\n({val})"
                else:
                    lbl = f"True Negative\n({val})"
                    
                ax.text(y, x, lbl, ha="center", va="center", color=text_color, 
                        fontsize=10, fontweight='bold')
                
        # Menghapus dekorasi yang tidak penting
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Visualisasi] Confusion Matrix Heatmap berhasil disimpan ke: {save_path}")

def plot_metrics_comparison(knn_metrics, nb_metrics, save_path='reports/metrics_comparison.png'):
    """
    Menggambar diagram batang perbandingan akurasi, presisi, recall, dan F1-score.
    
    Parameter:
        knn_metrics (dict): Metrik evaluasi model KNN.
        nb_metrics (dict): Metrik evaluasi model Naive Bayes.
        save_path (str): Jalur penyimpanan gambar.
    """
    create_reports_dir()
    
    categories = ['Akurasi', 'Presisi', 'Recall', 'F1-Score']
    
    # Mengambil nilai metrik dan mengubahnya ke persentase
    knn_vals = [
        knn_metrics['accuracy'] * 100,
        knn_metrics['precision'] * 100,
        knn_metrics['recall'] * 100,
        knn_metrics['f1_score'] * 100
    ]
    
    nb_vals = [
        nb_metrics['accuracy'] * 100,
        nb_metrics['precision'] * 100,
        nb_metrics['recall'] * 100,
        nb_metrics['f1_score'] * 100
    ]
    
    x = np.arange(len(categories))
    width = 0.35 # Lebar batang
    
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    # Plot batang (Steel Blue untuk KNN, Emerald Green untuk Naive Bayes)
    rects_knn = ax.bar(x - width/2, knn_vals, width, label='KNN (K=9)', color='#3498db', edgecolor='#2980b9', alpha=0.9)
    rects_nb = ax.bar(x + width/2, nb_vals, width, label='Gaussian Naive Bayes', color='#2ecc71', edgecolor='#27ae60', alpha=0.9)
    
    # Pengaturan judul dan label aksis
    ax.set_title('Perbandingan Performa Klasifikasi Cuaca (%)', fontsize=14, fontweight='bold', pad=20, color='#2c3e50')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylabel('Skor Persentase (%)', fontsize=11)
    ax.set_ylim(0, 110) # Rentang Y dilebihkan agar muat label angka di atas batang
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#dee2e6')
    
    # Fungsi penambah label angka persentase di atas setiap batang diagram
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color='#34495e')
            
    autolabel(rects_knn)
    autolabel(rects_nb)
    
    # Merapikan tampilan border aksis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Visualisasi] Perbandingan metrik berhasil disimpan ke: {save_path}")

def plot_time_comparison(knn_time, nb_times, save_path='reports/time_comparison.png'):
    """
    Menggambar diagram batang perbandingan waktu latih dan waktu prediksi (skala log).
    
    Parameter:
        knn_time (float): Waktu pengujian (prediksi) KNN.
        nb_times (tuple): (train_time, pred_time) untuk Naive Bayes.
        save_path (str): Jalur penyimpanan gambar.
    """
    create_reports_dir()
    
    train_nb, pred_nb = nb_times
    
    # KNN tidak memiliki waktu latih aktif (Lazy learner, 0 detik)
    categories = ['Waktu Latih (Training)', 'Waktu Uji (Predicting)']
    knn_times = [0.000001, knn_time] # Tambahkan angka mikro sebagai pengganti 0 agar tidak error skala log
    nb_times_list = [train_nb, pred_nb]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    
    # Gunakan skala logaritmik karena perbedaan waktu NB vs KNN sangat ekstrem (milidetik vs puluhan detik)
    ax.set_yscale('log')
    
    rects_knn = ax.bar(x - width/2, knn_times, width, label='KNN (K=9)', color='#34495e', edgecolor='#2c3e50', alpha=0.85)
    rects_nb = ax.bar(x + width/2, nb_times_list, width, label='Gaussian Naive Bayes', color='#e74c3c', edgecolor='#c0392b', alpha=0.85)
    
    ax.set_title('Perbandingan Waktu Komputasi (Detik - Skala Log)', fontsize=14, fontweight='bold', pad=20, color='#2c3e50')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylabel('Waktu (Detik) - Skala Log', fontsize=11)
    
    ax.grid(axis='y', which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#dee2e6')
    
    # Label nilai asli di atas diagram batang
    def label_time(rects, is_knn=False):
        for idx, rect in enumerate(rects):
            height = rect.get_height()
            # Tuliskan teks khusus untuk training KNN
            if is_knn and idx == 0:
                lbl_text = "0.0s (Lazy)"
            else:
                lbl_text = f"{height:.5f}s"
                
            ax.annotate(lbl_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2c3e50')
            
    label_time(rects_knn, is_knn=True)
    label_time(rects_nb, is_knn=False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Visualisasi] Perbandingan waktu komputasi berhasil disimpan ke: {save_path}")

def plot_knn_tuning(k_values, accuracies, f1_scores, save_path='reports/knn_k_tuning.png'):
    """
    Menggambar grafik pengaruh nilai hyperparameter K terhadap akurasi dan F1-Score KNN.
    
    Parameter:
        k_values (list): Nilai K yang diuji.
        accuracies (list): Skor akurasi pada tiap K (dalam desimal 0.0 - 1.0).
        f1_scores (list): Skor F1 pada tiap K (dalam desimal 0.0 - 1.0).
        save_path (str): Jalur penyimpanan gambar.
    """
    create_reports_dir()
    
    # Ubah data ke persentase
    acc_pct = [val * 100 for val in accuracies]
    f1_pct = [val * 100 for val in f1_scores]
    
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    # Menggambar garis plot dengan marker
    ax.plot(k_values, acc_pct, marker='o', linewidth=2.5, markersize=8, color='#3498db', label='Akurasi (%)')
    ax.plot(k_values, f1_pct, marker='s', linewidth=2.5, markersize=8, color='#e67e22', label='F1-Score (%)')
    
    # Judul dan dekorasi
    ax.set_title('Pengaruh Hyperparameter K terhadap Performa KNN', fontsize=14, fontweight='bold', pad=20, color='#2c3e50')
    ax.set_xlabel('Nilai Tetangga Terdekat (K)', fontsize=11, labelpad=8)
    ax.set_ylabel('Skor Performa (%)', fontsize=11, labelpad=8)
    
    ax.set_xticks(k_values)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#dee2e6')
    
    # Menambahkan label teks angka di setiap titik data plot untuk kejelasan
    for i, k in enumerate(k_values):
        ax.annotate(f"{acc_pct[i]:.2f}%", (k, acc_pct[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, fontweight='bold', color='#2980b9')
        ax.annotate(f"{f1_pct[i]:.2f}%", (k, f1_pct[i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, fontweight='bold', color='#d35400')
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Visualisasi] Grafik tuning parameter K KNN berhasil disimpan ke: {save_path}")
