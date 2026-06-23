def calculate_confusion_matrix(y_true, y_pred):
    """
    Menghitung komponen matriks kebingungan (Confusion Matrix):
    - True Positive (TP): Aktual Hujan, Prediksi Hujan
    - True Negative (TN): Aktual Tidak Hujan, Prediksi Tidak Hujan
    - False Positive (FP): Aktual Tidak Hujan, Prediksi Hujan
    - False Negative (FN): Aktual Hujan, Prediksi Tidak Hujan
    
    Parameter:
        y_true (list): Nilai target aktual/sebenarnya.
        y_pred (list): Nilai hasil prediksi model.
        
    Return:
        tuple: (tp, tn, fp, fn) yang berisi frekuensi hitung masing-masing kondisi.
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    # Mencocokkan nilai aktual dan prediksi pada baris yang sama
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
            
    return tp, tn, fp, fn

def calculate_metrics(y_true, y_pred):
    """
    Menghitung metrik performa klasifikasi dari awal hingga akhir, termasuk:
    - Accuracy: Seberapa banyak prediksi yang tepat dari total seluruh data.
    - Precision: Dari semua prediksi kelas Hujan, berapa yang benar-benar Hujan.
    - Recall (Sensitivity): Dari seluruh hari yang aktualnya Hujan, berapa banyak yang berhasil dideteksi.
    - F1-Score: Rata-rata harmonik antara Precision dan Recall (baik untuk evaluasi jika data sedikit tidak seimbang).
    
    Parameter:
        y_true (list): Target aktual.
        y_pred (list): Hasil prediksi.
        
    Return:
        dict: Kamus berisi nilai metrik evaluasi numerik.
    """
    total = len(y_true)
    if total == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "confusion_matrix": (0, 0, 0, 0)
        }
        
    # Ambil nilai matriks kebingungan
    tp, tn, fp, fn = calculate_confusion_matrix(y_true, y_pred)
    
    # 1. Akurasi = (TP + TN) / Total
    accuracy = (tp + tn) / total
    
    # 2. Presisi = TP / (TP + FP) (Gunakan pengaman jika penyebutnya 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # 3. Recall = TP / (TP + FN) (Gunakan pengaman jika penyebutnya 0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # 4. F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": (tp, tn, fp, fn)
    }

def print_metrics(metrics, title="Evaluation Results"):
    """
    Mencetak laporan hasil metrik evaluasi dan tampilan visual Confusion Matrix
    ke layar terminal dengan format yang rapi dan mudah dibaca.
    
    Parameter:
        metrics (dict): Hasil kamus metrik dari calculate_metrics.
        title (str): Judul judul laporan (misal nama model yang diuji).
    """
    tp, tn, fp, fn = metrics["confusion_matrix"]
    
    print(f"\n==============================================")
    print(f"  {title.upper()}")
    print(f"==============================================")
    print(f" Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f" Precision : {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f" Recall    : {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f" F1-Score  : {metrics['f1_score']:.4f} ({metrics['f1_score'] * 100:.2f}%)")
    print(f"----------------------------------------------")
    print(f" Confusion Matrix:")
    print(f"                    Predicted Hujan    Predicted Tidak Hujan")
    print(f" Actual Hujan            {tp:<15}    {fn:<15}")
    print(f" Actual Tidak Hujan      {fp:<15}    {tn:<15}")
    print(f"==============================================")
