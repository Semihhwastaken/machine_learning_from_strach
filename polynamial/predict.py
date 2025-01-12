import pickle

def load_model(filename='best_model.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def predict_price(model_data, features):
    """
    Kaydedilmiş modeli kullanarak fiyat tahmini yap
    """
    model = model_data['model']
    X_params, y_params = model_data['params']
    
    # Özellikleri normalize et
    X_norm = []
    for i, value in enumerate(features):
        feature_mean, feature_std = X_params[i]
        if feature_std == 0:
            X_norm.append(value)
        else:
            X_norm.append((value - feature_mean) / feature_std)
    
    # Tahmin yap
    y_norm_pred = model.predict([X_norm])[0]
    y_mean, y_std = y_params
    
    # Tahmini gerçek değere dönüştür
    return y_norm_pred * y_std + y_mean

if __name__ == "__main__":
    # Modeli yükle
    model_data = load_model()
    
    # Örnek tahmin
    features = [2850, 3, 3, 4200, 2, 4, 2000]  # Örnek ev özellikleri
    predicted_price = predict_price(model_data, features)
    
    print(f"\nTahmin edilen fiyat: ${predicted_price:,.2f}") 