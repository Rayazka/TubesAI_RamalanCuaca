import numpy as np

class GaussianNaiveBayes:
    """
    Model Gaussian Naive Bayes Classifier yang diimplementasikan secara manual
    dengan memanfaatkan operasi vektor dan matriks NumPy untuk performa tinggi.
    """
    def __init__(self):
        self.classes = []
        self.priors = {}       # Prior P(y) untuk setiap kelas
        self.means = {}        # Rata-rata fitur per kelas: {kelas: np.ndarray}
        self.variances = {}    # Varians fitur per kelas: {kelas: np.ndarray}
        self.is_fitted = False

    def fit(self, X, y):
        """
        Melatih model Naive Bayes menggunakan operasi rata-rata dan varians kolom NumPy.
        
        Parameter:
            X (np.ndarray): Fitur data training berdimensi (n_samples, n_features).
            y (np.ndarray): Target label data training berdimensi (n_samples,).
        """
        n_samples = len(X)
        self.classes = np.unique(y) # Dapatkan semua kelas unik secara otomatis
        
        for c in self.classes:
            # Saring baris yang memiliki kelas target c (boolean indexing)
            X_c = X[y == c]
            
            # 1. Hitung Prior P(y = c)
            self.priors[c] = len(X_c) / n_samples
            
            # 2. Hitung rata-rata dan varians kolom (axis=0) secara paralel menggunakan NumPy
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0)
            
        self.is_fitted = True
        print("[Naive Bayes] Model berhasil dilatih secara efisien dengan NumPy.")

    def _calculate_log_likelihood(self, x, mean, variance):
        """
        Menghitung penjumlahan log-likelihood P(X | y) secara vektor untuk semua fitur sekaligus.
        
        Rumus Log-PDF Gaussian:
        log(P(x_i | y)) = -0.5 * log(2 * pi * variance) - ((x_i - mean)^2 / (2 * variance))
        
        Parameter:
            x (np.ndarray): Vektor fitur data uji tunggal (n_features,).
            mean (np.ndarray): Vektor rata-rata fitur pada kelas (n_features,).
            variance (np.ndarray): Vektor varians fitur pada kelas (n_features,).
            
        Return:
            float: Total penjumlahan log-likelihood untuk seluruh fitur.
        """
        # Epsilon kecil agar tidak terjadi pembagian dengan nol
        eps = 1e-9
        var = variance + eps
        
        # Hitung log probabilitas densitas Gaussian untuk seluruh kolom fitur sekaligus
        log_pdf = -0.5 * np.log(2.0 * np.pi * var) - ((x - mean) ** 2) / (2.0 * var)
        
        # Jumlahkan log-likelihood dari semua fitur (menggantikan loop perkalian PDF)
        return np.sum(log_pdf)

    def predict_one(self, x):
        """
        Memprediksi kelas untuk satu vektor sampel uji.
        P(y | X) dihitung melalui penjumlahan log prior dan log likelihood.
        
        Parameter:
            x (np.ndarray): Vektor fitur data uji tunggal.
            
        Return:
            int: Label kelas hasil prediksi.
        """
        if not self.is_fitted:
            raise ValueError("Model belum dilatih!")
            
        posteriors = {}
        for c in self.classes:
            # log( P(c) )
            log_prior = math_log = np.log(self.priors[c])
            
            # Menghitung sum( log( P(x_i | c) ) ) secara vektor
            log_likelihood = self._calculate_log_likelihood(x, self.means[c], self.variances[c])
            
            # Posterior = log prior + log likelihood
            posteriors[c] = log_prior + log_likelihood
            
        # Kembalikan kelas dengan probabilitas log-posterior terbesar
        return int(max(posteriors, key=posteriors.get))

    def predict(self, X):
        """
        Memprediksi kelas untuk seluruh matriks data uji.
        
        Parameter:
            X (np.ndarray): Matriks fitur data uji berdimensi (n_samples, n_features).
            
        Return:
            list: Daftar label kelas hasil prediksi.
        """
        return [self.predict_one(row) for row in X]
