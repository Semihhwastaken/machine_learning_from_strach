import pandas as pd
import numpy as np
from datetime import datetime

class MultipleRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        # Veriyi sakla (görselleştirme için)
        self._y_train = y
        
        # n_samples: örnek sayısı, n_features: özellik sayısı
        n_samples = len(X)
        n_features = len(X[0])
        
        # Ağırlıkları ve bias'ı sıfırla
        self.weights = [0] * n_features
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.iterations):
            # Tahmin
            y_pred = self._predict(X)
            
            # Gradyanları hesapla
            dw = [0] * n_features
            db = 0
            
            for i in range(n_samples):
                error = y_pred[i] - y[i]
                
                # Ağırlık gradyanları
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                
                # Bias gradyanı
                db += error
            
            # Ağırlıkları güncelle
            for j in range(n_features):
                self.weights[j] -= (self.learning_rate * dw[j]) / n_samples
            
            # Bias'ı güncelle
            self.bias -= (self.learning_rate * db) / n_samples
    
    def _predict(self, X):
        predictions = []
        for sample in X:
            prediction = self.bias
            for x_i, w_i in zip(sample, self.weights):
                prediction += x_i * w_i
            predictions.append(prediction)
        return predictions
    
    def predict(self, X):
        return self._predict(X)
    
    def plot_model(self, X):
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np
            
            # Tahmin yüzeyi için grid oluştur
            x1_min, x1_max = min(x[0] for x in X), max(x[0] for x in X)
            x2_min, x2_max = min(x[1] for x in X), max(x[1] for x in X)
            
            x1_grid = np.linspace(x1_min - 1, x1_max + 1, 20)
            x2_grid = np.linspace(x2_min - 1, x2_max + 1, 20)
            xx1, xx2 = np.meshgrid(x1_grid, x2_grid)
            
            # Tahminleri hesapla
            grid_points = [[x1, x2] for x1, x2 in zip(xx1.ravel(), xx2.ravel())]
            predictions = self._predict(grid_points)
            zz = np.array(predictions).reshape(xx1.shape)
            
            # Grafiği çiz
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            
            # Tahmin yüzeyini çiz
            surf = ax.plot_surface(xx1, xx2, zz, alpha=0.5, cmap='viridis')
            
            # Gerçek veri noktalarını çiz
            if hasattr(self, '_y_train'):
                ax.scatter([x[0] for x in X], [x[1] for x in X], self._y_train, 
                         color='red', marker='o', s=100)
            
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_zlabel('Y')
            plt.colorbar(surf)
            plt.title('Multiple Regression Model')
            plt.show()
            
        except ImportError:
            print("Görselleştirme için matplotlib kütüphanesi gereklidir.")

# Örnek kullanım
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from datetime import datetime
    
    # Veri setini oluştur
    data = []
    with open('C:\\Users\\semih\\OneDrive\\Masaüstü\\coding_challange\\machine_learning\\data.csv', 'r') as file:
        next(file)  # Başlık satırını atla
        for line in file:
            values = line.strip().split(',')
            if len(values) >= 5:  # Gerekli alanların varlığını kontrol et
                try:
                    price = float(values[1])        # Fiyat
                    bedrooms = float(values[2])     # Yatak odası
                    bathrooms = float(values[3])    # Banyo
                    sqft_living = float(values[4])  # Yaşam alanı
                    
                    # Sıfır olmayan değerleri ve mantıklı aralıktaki verileri al
                    if (price > 0 and sqft_living > 0 and 
                        bedrooms > 0 and bathrooms > 0):
                        data.append([
                            sqft_living,  # Yaşam alanı
                            bedrooms,     # Yatak odası sayısı
                            bathrooms,    # Banyo sayısı
                            price         # Fiyat (hedef değişken)
                        ])
                except ValueError:
                    continue
    
    # Numpy dizilerine dönüştür
    data = np.array(data)
    X = [[x[0], x[1], x[2]] for x in data]  # Yaşam alanı, yatak odası ve banyo
    y = [x[3] for x in data]                 # Fiyat
    
    # Verileri normalize et
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    y_mean = np.mean(y)
    y_std = np.std(y)
    
    X_normalized = [[(x[i] - X_mean[i]) / X_std[i] for i in range(len(x))] for x in X]
    y_normalized = [(y_i - y_mean) / y_std for y_i in y]
    
    # Model oluştur ve eğit
    model = MultipleRegression(learning_rate=0.01, iterations=1000)
    model.fit(X_normalized, y_normalized)
    
    # 2x2 subplot oluştur
    fig = plt.figure(figsize=(20, 15))
    
    # 3D Plot (Ana görselleştirme)
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Gerçek veri noktalarını çiz
    scatter1 = ax1.scatter([x[0] for x in X],    # Yaşam alanı
                         [x[1] for x in X],      # Yatak odası
                         y,                      # Fiyat
                         c=[x[2] for x in X],    # Banyo sayısı (renk)
                         cmap='viridis',
                         alpha=0.6,
                         label='Gerçek Veriler')
    
    # Tahmin yüzeyi
    x1_grid = np.linspace(min(x[0] for x in X), max(x[0] for x in X), 20)
    x2_grid = np.linspace(min(x[1] for x in X), max(x[1] for x in X), 20)
    xx1, xx2 = np.meshgrid(x1_grid, x2_grid)
    
    x3_mean = np.mean([x[2] for x in X])
    grid_points_normalized = [
        [(x1 - X_mean[0]) / X_std[0], 
         (x2 - X_mean[1]) / X_std[1],
         (x3_mean - X_mean[2]) / X_std[2]]
        for x1, x2 in zip(xx1.ravel(), xx2.ravel())
    ]
    
    predictions_normalized = model.predict(grid_points_normalized)
    predictions = [y_pred * y_std + y_mean for y_pred in predictions_normalized]
    zz = np.array(predictions).reshape(xx1.shape)
    
    surf = ax1.plot_surface(xx1, xx2, zz, alpha=0.2, cmap='viridis')
    
    ax1.set_xlabel('Yaşam Alanı (sqft)')
    ax1.set_ylabel('Yatak Odası Sayısı')
    ax1.set_zlabel('Fiyat ($)')
    ax1.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.colorbar(scatter1, ax=ax1, label='Banyo Sayısı')
    ax1.set_title('3D Görünüm')
    ax1.view_init(elev=20, azim=45)
    
    # Yaşam Alanı vs Fiyat Projeksiyonu
    ax2 = fig.add_subplot(222)
    scatter2 = ax2.scatter([x[0] for x in X], y, 
                         c=[x[2] for x in X],
                         cmap='viridis',
                         alpha=0.6)
    
    # Regresyon çizgisi için yeni tahminler
    x_range = np.linspace(min(x[0] for x in X), max(x[0] for x in X), 100)
    x_normalized = [(x - X_mean[0]) / X_std[0] for x in x_range]
    
    # Diğer özelliklerin ortalama değerlerini kullan
    x2_mean = np.mean([x[1] for x in X])  # Yatak odası ortalaması
    x3_mean = np.mean([x[2] for x in X])  # Banyo ortalaması
    
    # Tahmin için normalize edilmiş veri noktaları
    pred_points = [[x, 
                   (x2_mean - X_mean[1]) / X_std[1],
                   (x3_mean - X_mean[2]) / X_std[2]] for x in x_normalized]
    
    # Tahminleri hesapla ve geri dönüştür
    y_pred_line = model.predict(pred_points)
    y_pred_line = [y_pred * y_std + y_mean for y_pred in y_pred_line]
    
    ax2.plot(x_range, y_pred_line, 'r-', label='Regresyon', alpha=0.8)
    
    ax2.set_xlabel('Yaşam Alanı (sqft)')
    ax2.set_ylabel('Fiyat ($)')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.colorbar(scatter2, ax=ax2, label='Banyo Sayısı')
    ax2.set_title('Yaşam Alanı vs Fiyat')
    ax2.legend()
    
    # Yatak Odası vs Fiyat Projeksiyonu
    ax3 = fig.add_subplot(223)
    scatter3 = ax3.scatter([x[1] for x in X], y,
                         c=[x[2] for x in X],
                         cmap='viridis',
                         alpha=0.6)
    
    # Regresyon çizgisi için yeni tahminler
    x_range = np.linspace(min(x[1] for x in X), max(x[1] for x in X), 100)
    x_normalized = [(x - X_mean[1]) / X_std[1] for x in x_range]
    
    # Diğer özelliklerin ortalama değerlerini kullan
    x1_mean = np.mean([x[0] for x in X])  # Yaşam alanı ortalaması
    
    # Tahmin için normalize edilmiş veri noktaları
    pred_points = [[(x1_mean - X_mean[0]) / X_std[0],
                   x,
                   (x3_mean - X_mean[2]) / X_std[2]] for x in x_normalized]
    
    # Tahminleri hesapla ve geri dönüştür
    y_pred_line = model.predict(pred_points)
    y_pred_line = [y_pred * y_std + y_mean for y_pred in y_pred_line]
    
    ax3.plot(x_range, y_pred_line, 'r-', label='Regresyon', alpha=0.8)
    
    ax3.set_xlabel('Yatak Odası Sayısı')
    ax3.set_ylabel('Fiyat ($)')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.colorbar(scatter3, ax=ax3, label='Banyo Sayısı')
    ax3.set_title('Yatak Odası vs Fiyat')
    ax3.legend()
    
    # Gerçek vs Tahmin Karşılaştırması
    ax4 = fig.add_subplot(224)
    y_pred = model.predict(X_normalized)
    y_pred = [y_i * y_std + y_mean for y_i in y_pred]
    
    scatter4 = ax4.scatter(y, y_pred,
                         c=[x[2] for x in X],
                         cmap='viridis',
                         alpha=0.6)
    
    # Mükemmel tahmin çizgisi
    min_val = min(min(y), min(y_pred))
    max_val = max(max(y), max(y_pred))
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', label='Mükemmel Tahmin')
    
    ax4.set_xlabel('Gerçek Fiyat ($)')
    ax4.set_ylabel('Tahmin Edilen Fiyat ($)')
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.colorbar(scatter4, ax=ax4, label='Banyo Sayısı')
    ax4.set_title('Gerçek vs Tahmin')
    ax4.legend()
    
    plt.suptitle('Ev Fiyatları Çok Değişkenli Analiz', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Örnek tahmin
    test_sample = [[2000, 3, 2]]  # 2000 sqft, 3 yatak odası, 2 banyo
    test_normalized = [[(test_sample[0][i] - X_mean[i]) / X_std[i] for i in range(len(test_sample[0]))]]
    prediction_normalized = model.predict(test_normalized)
    prediction = prediction_normalized[0] * y_std + y_mean
    
    print(f"\nÖrnek Tahmin:")
    print(f"Yaşam alanı: {test_sample[0][0]} sqft")
    print(f"Yatak odası sayısı: {test_sample[0][1]}")
    print(f"Banyo sayısı: {test_sample[0][2]}")
    print(f"Tahmin edilen fiyat: ${prediction:,.2f}")
    
    # Model performans metrikleri
    y_pred_normalized = model.predict(X_normalized)
    y_pred = [y_i * y_std + y_mean for y_i in y_pred_normalized]
    
    # Model performans metrikleri ve örnek tahmin çıktısı
    # ... (önceki kodla aynı) ...
