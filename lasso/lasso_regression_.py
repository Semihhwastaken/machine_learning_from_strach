class LassoRegression:
    def __init__(self, learning_rate=0.01, lambda_param=1.0, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # Örnek sayısı ve özellik sayısı
        n_samples, n_features = len(X), len(X[0])
        
        # Ağırlıkları ve bias'ı başlat
        self.weights = [0] * n_features
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Tahmin
            y_pred = self._predict(X)
            
            # Gradyanları hesapla
            dw = [0] * n_features
            db = 0
            
            for i in range(n_samples):
                error = y_pred[i] - y[i]
                # Bias gradyanı
                db += error
                
                # Ağırlık gradyanları
                for j in range(n_features):
                    dw[j] += error * X[i][j]
            
            # Gradyanları ortala
            db /= n_samples
            for j in range(n_features):
                dw[j] = dw[j] / n_samples + self.lambda_param * self._sign(self.weights[j])
            
            # Ağırlıkları güncelle
            self.bias -= self.learning_rate * db
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * dw[j]
    
    def _predict(self, X):
        predictions = []
        for xi in X:
            pred = self.bias
            for j in range(len(self.weights)):
                pred += self.weights[j] * xi[j]
            predictions.append(pred)
        return predictions
    
    def predict(self, X):
        return self._predict(X)
    
    def _sign(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        return 0
