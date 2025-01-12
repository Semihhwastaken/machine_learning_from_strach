import pandas as pd
import matplotlib.pyplot as plt
from polynomial_regression_ import PolynomialRegression
import numpy as np
import pickle
import os

def load_and_prepare_data(filename):
    # Veriyi pandas ile oku
    df = pd.read_csv(filename)
    
    # Sıcaklık ve satış verilerini al
    X = [[x] for x in df['Temperature (°C)'].values]
    y = df['Ice Cream Sales (units)'].values.tolist()
    
    return X, y

def prepare_house_data(filename):
    """Ev fiyatları verisini hazırlar ve polynomial regresyon için uygun formata getirir"""
    
    # Veriyi pandas ile oku
    df = pd.read_csv(filename)
    
    # Sıfır fiyatlı evleri ve eksik verileri temizle
    df = df[df['price'] > 0].dropna()
    
    # Özellik seçimi (örnek olarak ev büyüklüğü ve fiyat ilişkisi)
    X = [[x] for x in df['sqft_living'].values]  # Evin yaşam alanı
    y = df['price'].values.tolist()  # Ev fiyatı
    
    return X, y

def normalize_data(X, y):
    # X için normalizasyon
    X_values = [x[0] for x in X]
    X_mean = sum(X_values) / len(X_values)
    X_std = (sum((x - X_mean) ** 2 for x in X_values) / len(X_values)) ** 0.5
    
    X_norm = [[(x[0] - X_mean) / X_std] for x in X]
    
    # y için normalizasyon
    y_mean = sum(y) / len(y)
    y_std = (sum((yi - y_mean) ** 2 for yi in y) / len(y)) ** 0.5
    y_norm = [(yi - y_mean) / y_std for yi in y]
    
    return X_norm, y_norm, (X_mean, X_std), (y_mean, y_std)

def main():
    # Veri setini seç
    data_type = input("Veri seti seçin (1: Dondurma Satış, 2: Ev Fiyatları): ")
    
    if data_type == "1":
        X, y = load_and_prepare_data('Ice_cream selling data.csv')
        x_label = 'Sıcaklık (°C)'
        y_label = 'Dondurma Satışı (birim)'
        title = 'Sıcaklık - Dondurma Satışı İlişkisi'
    else:
        X, y = prepare_house_data('C:\\Users\\semih\\OneDrive\\Masaüstü\\coding_challange\\machine_learning\\data.csv')
        x_label = 'Yaşam Alanı (sqft)'
        y_label = 'Fiyat ($)'
        title = 'Ev Büyüklüğü - Fiyat İlişkisi'
    
    # Veriyi normalize et
    X_norm, y_norm, X_params, y_params = normalize_data(X, y)
    
    # Model dereceleri
    degrees = [2, 3,4,15]
    models = []
    mse_scores = []
    
    print("\nFarklı dereceler için test:")
    print("-" * 50)
    
    for degree in degrees:
        model = PolynomialRegression(degree=degree)
        model.fit(X_norm, y_norm)
        
        y_pred_norm = model.predict(X_norm)
        y_pred = [y_n * y_params[1] + y_params[0] for y_n in y_pred_norm]
        
        mse = sum((y_true - y_pred) ** 2 for y_true, y_pred in zip(y, y_pred)) / len(y)
        print(f"Derece {degree} - MSE: {mse:.4f}")
        
        models.append(model)
        mse_scores.append(mse)
    
    # En iyi modeli seç
    best_idx = mse_scores.index(min(mse_scores))
    best_model = models[best_idx]
    best_degree = degrees[best_idx]
    
    # Görselleştirme
    plt.figure(figsize=(12, 8))
    plt.scatter([x[0] for x in X], y, color='blue', alpha=0.5, label='Gerçek Veriler')
    
    X_range = np.linspace(min(x[0] for x in X), max(x[0] for x in X), 100)
    X_range_norm = [[(x - X_params[0]) / X_params[1]] for x in X_range]
    
    for i, (model, degree) in enumerate(zip(models, degrees)):
        y_pred_norm = model.predict(X_range_norm)
        y_pred = [y_n * y_params[1] + y_params[0] for y_n in y_pred_norm]
        plt.plot(X_range, y_pred, color=['red', 'green','blue','black'][i], 
                label=f'Polynomial (derece={degree})',
                linestyle='--' if i != best_idx else '-')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()