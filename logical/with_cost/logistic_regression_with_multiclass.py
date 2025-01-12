import math

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.costs = []  # Eğitim sırasındaki maliyet değerlerini takip etmek için
        
    def softmax(self, z):
        """
        Softmax fonksiyonu, çoklu sınıf tahminleri için kullanılır.
        """
        exp_z = [math.exp(i) for i in z]
        sum_exp_z = sum(exp_z)
        return [i / sum_exp_z for i in exp_z]
    
    def compute_cost(self, X, y, predictions):
        """
        Multiclass Cross-Entropy Loss hesapla:
        J = -1/m * Σ Σ(y*log(h))
        """
        m = len(y)
        cost = 0
        epsilon = 1e-15  # log(0) hatası almamak için küçük bir değer
        
        for i in range(m):
            for j in range(len(predictions[i])):
                pred = max(min(predictions[i][j], 1 - epsilon), epsilon)
                if y[i] == j:  # y[i] doğru sınıf
                    cost += -math.log(pred)
        
        return cost / m
    
    def fit(self, X, y):
        """
        Modeli eğit
        """
        # Bias terimi ekle
        X = [[1] + list(x) for x in X]
        
        # Sınıf sayısını belirle
        self.num_classes = len(set(y))
        
        # Ağırlıkları başlat
        num_features = len(X[0])
        self.weights = [[0] * num_features for _ in range(self.num_classes)]
        
        # Gradient descent
        self.costs = []
        for i in range(self.num_iterations):
            # Forward pass
            probas = self.predict_proba(X)
            
            # Gradient hesapla ve ağırlıkları güncelle
            for k in range(self.num_classes):
                for j in range(num_features):
                    gradient = 0
                    for xi, yi, pi in zip(X, y, probas):
                        gradient += xi[j] * (1.0 if yi == k else 0.0 - pi[k])
                    self.weights[k][j] += self.learning_rate * gradient / len(X)
            
            # Cost hesapla
            cost = 0
            for yi, pi in zip(y, probas):
                cost -= math.log(pi[yi] + 1e-15)  # Sayısal kararlılık için küçük değer ekle
            cost /= len(X)
            self.costs.append(cost)
            
            if i % 100 == 0:
                print(f"İterasyon {i}, Cost: {cost:.6f}")
    
    def predict(self, X):
        """
        X verisi için tahmin yap
        """
        predictions = self.predict_proba(X)
        # En yüksek olasılığa sahip sınıfı döndür
        return [self._get_max_index(p) for p in predictions]
    
    def _get_max_index(self, lst):
        """
        Bir listedeki en büyük değerin indeksini döndür
        """
        max_idx = 0
        max_val = lst[0]
        for i, val in enumerate(lst):
            if val > max_val:
                max_val = val
                max_idx = i
        return max_idx
    
    def predict_proba(self, X):
        """
        Her sınıf için olasılıkları hesapla
        """
        if not isinstance(X[0], list):  # Tek bir örnek gönderildiyse
            X = [X]
        
        # Bias terimi ekle
        X = [[1] + list(x) for x in X]
        
        probas = []
        for x in X:
            # Her örnek için sınıf skorlarını hesapla
            scores = []
            for w in self.weights:
                # Doğrusal skoru hesapla
                score = sum(xi * wi for xi, wi in zip(x, w))
                scores.append(score)
            
            # Sayısal kararlılık için max değeri çıkar
            max_score = max(scores)
            exp_scores = [math.exp(s - max_score) for s in scores]
            total_exp = sum(exp_scores)
            
            # Normalize edilmiş olasılıklar
            proba = [e / total_exp for e in exp_scores]
            probas.append(proba)
        
        return probas
    
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
