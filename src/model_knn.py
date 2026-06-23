import numpy as np

class KNNClassifier:
    """
    Model K-Nearest Neighbors Classifier yang diimplementasikan
    dengan optimasi komputasi matriks menggunakan NumPy (Broadcasting).
    """
    def __init__(self, k=5):
        """
        Inisialisasi model KNN dengan menentukan jumlah tetangga terdekat (K).
        
        Parameter:
            k (int): Jumlah tetangga terdekat yang digunakan untuk proses voting.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Menyimpan data training berupa array NumPy ke dalam memori.
        
        Parameter:
            X (np.ndarray): Fitur data training berdimensi (n_samples, n_features).
            y (np.ndarray): Target label data training berdimensi (n_samples,).
        """
        self.X_train = X
        self.y_train = y

    def predict_one(self, x):
        """
        Memprediksi kelas untuk satu vektor sampel uji menggunakan operasi vektor NumPy.
        
        Langkah-langkah:
        1. Hitung kuadrat jarak Euclidean dari x ke seluruh baris X_train secara paralel.
        2. Cari K indeks tetangga terdekat dengan partisi dan sorting NumPy.
        3. Lakukan voting mayoritas kelas tetangga terdekat.
        
        Parameter:
            x (np.ndarray): Vektor fitur data uji tunggal (n_features,).
            
        Return:
            int: Label kelas hasil prediksi (0 atau 1).
        """
        # 1. Menghitung kuadrat jarak Euclidean secara paralel menggunakan Broadcasting
        # (self.X_train - x) mengurangi x dari setiap baris di X_train secara otomatis
        # .sum(axis=1) menjumlahkan selisih kuadrat sepanjang kolom fitur (horizontal)
        dists_sq = np.sum((self.X_train - x) ** 2, axis=1)
        
        # 2. Mengambil K indeks data dengan jarak kuadrat terkecil
        # np.argpartition mempartisi K data terkecil secara efisien dalam kompleksitas O(N)
        k_indices = np.argpartition(dists_sq, self.k)[:self.k]
        
        # Urutkan K data terpilih tersebut berdasarkan jarak sebenarnya dari yang terdekat
        k_indices = k_indices[np.argsort(dists_sq[k_indices])]
        
        # Dapatkan label kelas dari K tetangga terdekat
        k_labels = self.y_train[k_indices]
        
        # 3. Hitung jumlah voting kelas menggunakan operasi NumPy
        unique_labels, counts = np.unique(k_labels, return_counts=True)
        max_count = np.max(counts)
        
        # Cari label kelas yang mendapatkan jumlah suara terbanyak (bisa lebih dari satu/seri)
        candidates = unique_labels[counts == max_count]
        
        # Jika hanya ada satu pemenang mutlak
        if len(candidates) == 1:
            return int(candidates[0])
            
        # Pemecah Seri (Tie-Breaker):
        # Pilih kelas dari kandidat terdekat pertama yang kita temui di k_labels
        # (karena k_labels sudah terurut dari jarak paling dekat ke jauh)
        for label in k_labels:
            if label in candidates:
                return int(label)
                
        return int(candidates[0])

    def predict(self, X):
        """
        Memprediksi kelas untuk seluruh baris matriks uji secara efisien.
        
        Parameter:
            X (np.ndarray): Matriks fitur data uji berdimensi (n_test_samples, n_features).
            
        Return:
            list: Daftar label kelas hasil prediksi.
        """
        predictions = []
        # Iterasi baris demi baris pada array NumPy
        for row in X:
            pred = self.predict_one(row)
            predictions.append(pred)
        return predictions
