class RidgeRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha  # regularizasyon parametresi
        self.max_iter = max_iter  # maksimum iterasyon sayısı
        self.tol = tol  # tolerans değeri
        self.weights = None
        self.bias = None
    
    def _normalize(self, X):
        # Özellikleri normalize et
        means = [sum(col) / len(col) for col in zip(*X)]
        stds = [sum((x - m) ** 2 for x in col) ** 0.5 / len(col) ** 0.5 
               for col, m in zip(zip(*X), means)]
        X_norm = [[(x - m) / s if s != 0 else 0 
                   for x, m, s in zip(row, means, stds)] 
                  for row in X]
        return X_norm, means, stds

    def fit(self, X, y):
        # Veriyi normalize et
        X_norm, self.means, self.stds = self._normalize(X)
        n_samples = len(X)
        n_features = len(X[0])
        
        # Ağırlıkları ve bias'ı başlat
        self.weights = [0.0] * n_features
        self.bias = 0.0
        
        # Gradyan inişi
        for _ in range(self.max_iter):
            weights_old = self.weights.copy()
            
            # Her örnek için gradyanları hesapla
            for i in range(n_samples):
                # Tahmin
                y_pred = sum(w * x for w, x in zip(self.weights, X_norm[i])) + self.bias
                
                # Hata
                error = y_pred - y[i]
                
                # Ağırlıkları güncelle
                for j in range(n_features):
                    self.weights[j] -= (2 * error * X_norm[i][j] + 
                                      2 * self.alpha * self.weights[j]) / n_samples
                
                # Bias'ı güncelle
                self.bias -= 2 * error / n_samples
            
            # Yakınsama kontrolü
            if all(abs(w - w_old) < self.tol 
                  for w, w_old in zip(self.weights, weights_old)):
                break
    
    def predict(self, X):
        # Test verisini normalize et
        X_norm = [[(x - m) / s if s != 0 else 0 
                   for x, m, s in zip(row, self.means, self.stds)] 
                  for row in X]
        
        # Tahminleri hesapla
        predictions = []
        for x in X_norm:
            y_pred = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
            predictions.append(y_pred)
        
        return predictions
