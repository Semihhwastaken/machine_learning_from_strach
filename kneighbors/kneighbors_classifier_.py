class KNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Eğitim verilerini saklar"""
        self.X_train = X
        self.y_train = y
    
    def _euclidean_distance(self, x1, x2):
        """İki nokta arasındaki Öklid mesafesini hesaplar"""
        distance = 0
        for i in range(len(x1)):
            distance += (x1[i] - x2[i]) ** 2
        return distance ** 0.5
    
    def _get_neighbors(self, test_sample):
        """Test örneği için en yakın k komşuyu bulur"""
        distances = []
        for i, train_sample in enumerate(self.X_train):
            dist = self._euclidean_distance(test_sample, train_sample)
            distances.append((dist, self.y_train[i]))
        
        # Mesafelere göre sırala ve en yakın k komşuyu al
        distances.sort(key=lambda x: x[0])
        return [d[1] for d in distances[:self.n_neighbors]]
    
    def predict(self, X):
        """Test örnekleri için tahminlerde bulunur"""
        predictions = []
        for test_sample in X:
            neighbors = self._get_neighbors(test_sample)
            # En sık görülen sınıfı bul
            prediction = max(set(neighbors), key=neighbors.count)
            predictions.append(prediction)
        return predictions

# Sınıflandırıcıyı oluştur
knn = KNeighborsClassifier(n_neighbors=3)

# Eğitim verileri
X_train = [[1, 2], [2, 3], [3, 1], [6, 5], [7, 8], [8, 7]]
y_train = [0, 0, 0, 1, 1, 1]

# Modeli eğit
knn.fit(X_train, y_train)

# Tahmin yap
X_test = [[5, 5]]
predictions = knn.predict(X_test)
print(predictions)  # Çıktı: [1]