import math

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.costs = []  # Eğitim sırasındaki maliyet değerlerini takip etmek için
        
    def sigmoid(self, z):
        if z < -20:
            return 0
        elif z > 20:
            return 1
        try:
            return 1 / (1 + math.exp(-z))
        except OverflowError:
            return 0 if z < 0 else 1
    
    def compute_cost(self, X, y, predictions):
        """
        Binary Cross-Entropy Loss hesapla:
        J = -1/m * Σ(y*log(h) + (1-y)*log(1-h))
        """
        m = len(y)
        cost = 0
        epsilon = 1e-15  # log(0) hatası almamak için küçük bir değer
        
        for i in range(m):
            # predictions'ı 0 ve 1'e çok yakın değerlerden kaçınmak için düzenle
            pred = max(min(predictions[i], 1 - epsilon), epsilon)
            if y[i] == 1:
                cost += -math.log(pred)
            else:
                cost += -math.log(1 - pred)
        
        return cost / m
    
    def fit(self, X, y):
        # Veriyi normalize et
        X_normalized = self._normalize_features(X)
        
        num_samples = len(X)
        num_features = len(X[0])
        
        self.weights = [0] * num_features
        self.bias = 0
        
        for iteration in range(self.num_iterations):
            # Forward propagation
            linear_pred = [sum(x_i * w_i for x_i, w_i in zip(x, self.weights)) + self.bias 
                         for x in X_normalized]
            predictions = [self.sigmoid(z) for z in linear_pred]
            
            # Cost hesapla
            cost = self.compute_cost(X_normalized, y, predictions)
            self.costs.append(cost)
            
            # Backward propagation
            dw = [0] * num_features
            db = 0
            
            for i in range(num_samples):
                difference = predictions[i] - y[i]
                for j in range(num_features):
                    dw[j] += X_normalized[i][j] * difference
                db += difference
            
            # Gradient descent güncelleme
            for j in range(num_features):
                self.weights[j] -= self.learning_rate * (dw[j] / num_samples)
            self.bias -= self.learning_rate * (db / num_samples)
            
            # Her 100 iterasyonda bir cost değerini yazdır
            if iteration % 100 == 0:
                print(f"Iterasyon {iteration}, Cost: {cost:.6f}")
    
    def predict(self, X):
        X_normalized = self._normalize_features(X)
        linear_pred = [sum(x_i * w_i for x_i, w_i in zip(x, self.weights)) + self.bias 
                      for x in X_normalized]
        predictions = [self.sigmoid(z) for z in linear_pred]
        return [1 if p >= 0.5 else 0 for p in predictions]
    
    def _normalize_features(self, X):
        X_normalized = []
        for feature_idx in range(len(X[0])):
            feature_values = [x[feature_idx] for x in X]
            min_val = min(feature_values)
            max_val = max(feature_values)
            range_val = max_val - min_val if max_val != min_val else 1
            
            normalized_feature = [(x - min_val) / range_val for x in feature_values]
            if not X_normalized:
                X_normalized = [[val] for val in normalized_feature]
            else:
                for i, val in enumerate(normalized_feature):
                    X_normalized[i].append(val)
        
        return X_normalized 