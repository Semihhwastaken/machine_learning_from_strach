import numpy as np
from logistic_regression_with_multiclass import LogisticRegression
import matplotlib.pyplot as plt

def create_sample_data(n_samples=300):
    """Örnek veri seti oluştur - 3 farklı sınıf için"""
    np.random.seed(42)
    
    # 3 farklı merkez nokta oluştur - Merkezleri biraz daha ayıralım
    centers = [[3, 3], [-3, -3], [3, -3]]
    
    # Her sınıf için veri noktaları oluştur
    X = []
    y = []
    
    for class_idx, center in enumerate(centers):
        n_class_samples = n_samples // 3
        
        # Varyansı daha da azaltalım
        class_samples = np.random.randn(n_class_samples, 2) * 0.15  # Daha kompakt kümeler
        class_samples = class_samples + np.array(center)
        
        X.extend(class_samples.tolist())
        y.extend([class_idx] * n_class_samples)
    
    return X, y

def plot_decision_boundary(X, y, model):
    """Karar sınırlarını görselleştir"""
    x_min, x_max = min(x[0] for x in X) - 1, max(x[0] for x in X) + 1
    y_min, y_max = min(x[1] for x in X) - 1, max(x[1] for x in X) + 1
    
    # Izgara noktaları oluştur
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    # Her ızgara noktası için tahmin yap
    grid_points = [[x, y] for x, y in zip(xx.ravel(), yy.ravel())]
    Z = model.predict(grid_points)
    Z = np.array(Z).reshape(xx.shape)
    
    # Görselleştir
    plt.contourf(xx, yy, Z, alpha=0.4)
    
    # Orijinal veri noktalarını çiz
    colors = ['red', 'blue', 'green']
    for i in range(3):
        idx = [j for j, label in enumerate(y) if label == i]
        plt.scatter([X[j][0] for j in idx], 
                   [X[j][1] for j in idx], 
                   c=colors[i], 
                   label=f'Sınıf {i}')
    
    plt.legend()
    plt.title('Lojistik Regresyon - Karar Sınırları')
    plt.show()

def main():
    # Örnek veri oluştur
    X, y = create_sample_data()
    
    # Model oluştur ve eğit
    model = LogisticRegression(
        learning_rate=0.005,  # Daha düşük learning rate
        num_iterations=500    # Daha az iterasyon
    )
    model.fit(X, y)
    
    # Eğitim maliyetini görselleştir
    plt.figure(figsize=(10, 5))
    plt.plot(model.costs)
    plt.title('Eğitim Maliyeti')
    plt.xlabel('İterasyon')
    plt.ylabel('Maliyet')
    plt.show()
    
    # Karar sınırlarını görselleştir
    plot_decision_boundary(X, y, model)
    
    # Test noktalarını güncelle - merkezlere daha yakın noktalar
    test_points = [
        [3.0, 3.0],     # Sınıf 0 için merkeze yakın
        [-3.0, -3.0],   # Sınıf 1 için merkeze yakın
        [3.0, -3.0]     # Sınıf 2 için merkeze yakın
    ]
    
    predictions = model.predict(test_points)
    
    print("\nÖrnek Tahminler:")
    for point, pred in zip(test_points, predictions):
        print(f"Nokta {point}: Tahmin edilen sınıf = {pred}")

if __name__ == "__main__":
    main() 