class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.coefficients = None
    
    def _create_polynomial_features(self, X):
        # Çoklu özellikler için polinomial özellikleri oluştur
        n_samples = len(X)
        n_features = len(X[0])
        
        # Başlangıçta bias terimi (1'ler)
        X_poly = [[1] * n_samples]
        
        # Her özellik için lineer terimler
        for j in range(n_features):
            X_poly.append([x[j] for x in X])
        
        # Etkileşim terimleri ve yüksek dereceli terimler
        if self.degree >= 2:
            for d in range(2, self.degree + 1):
                for j in range(n_features):
                    X_poly.append([x[j] ** d for x in X])
                    
                    # Özellikler arası etkileşimler
                    for k in range(j + 1, n_features):
                        X_poly.append([x[j] * x[k] for x in X])
        
        # Dikey matristen yatay matrise çevir
        return [[X_poly[j][i] for j in range(len(X_poly))] 
                for i in range(n_samples)]
    
    def _matrix_multiply(self, A, B):
        # İki matrisin çarpımı
        result = [[sum(a * b for a, b in zip(row, col))
                  for col in zip(*B)]
                 for row in A]
        return result
    
    def _matrix_transpose(self, A):
        # Matrisin transpozunu al
        return list(map(list, zip(*A)))
    
    def _solve_equation(self, A, b):
        n = len(A)
        
        # Genişletilmiş matris [A|b]
        M = [row[:] + [b[i][0]] for i, row in enumerate(A)]
        
        # İleri eliminasyon
        for i in range(n):
            pivot = M[i][i]
            if abs(pivot) < 1e-10:
                # Pivot çok küçükse, diagonal elemanı artır
                M[i][i] += 1e-10
                pivot = M[i][i]
            
            for j in range(i + 1, n):
                factor = M[j][i] / pivot
                for k in range(i, n + 1):
                    M[j][k] -= factor * M[i][k]
        
        # Geri yerine koyma
        x = [0] * n
        for i in range(n - 1, -1, -1):
            x[i] = M[i][n]
            for j in range(i + 1, n):
                x[i] -= M[i][j] * x[j]
            x[i] /= M[i][i]
        
        return [[xi] for xi in x]
    
    def fit(self, X, y):
        # Polinomial özellikleri oluştur
        X_poly = self._create_polynomial_features(X)
        
        # Normal denklem yöntemi
        X_t = self._matrix_transpose(X_poly)
        X_t_X = self._matrix_multiply(X_t, X_poly)
        X_t_y = self._matrix_multiply(X_t, [[yi] for yi in y])
        # Katsayıları hesapla
        self.coefficients = self._solve_equation(X_t_X, X_t_y)
        
        
        return self
    
    def predict(self, X):
        # Tahmin için X'i polinom özelliklerine dönüştür
        X_poly = self._create_polynomial_features(X)
        
        # Tahminleri hesapla
        predictions = self._matrix_multiply(X_poly, self.coefficients)
        return [pred[0] for pred in predictions]

