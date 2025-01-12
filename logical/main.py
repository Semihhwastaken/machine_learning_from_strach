from logistic_regression_ import LogisticRegression
import matplotlib.pyplot as plt

def calculate_accuracy(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

# Öğrenci verileri [çalışma_saati, önceki_sınav_notu]
X_train = [
    [2, 50],   # Başarısız örnekler
    [4, 60],
    [3, 55],
    [1, 45],
    [2, 48],
    [3, 52],
    [8, 85],   # Başarılı örnekler
    [7, 80],
    [6, 75],
    [9, 90],
    [5, 65],
    [10, 95],
    [7, 78],
    [8, 88],
    [6, 77]
]

y_train = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

X_test = [
    [3, 52],   # Test verileri
    [7, 82],
    [5, 70],
    [4, 57],
]

def plot_cost_history(costs):
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.title('Cost Function Değişimi')
    plt.xlabel('İterasyon')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()

def main():
    # Farklı learning rate'ler deneyelim
    learning_rates = [0.001, 0.01, 0.1]
    
    for lr in learning_rates:
        print(f"\nLearning Rate: {lr}")
        print("-" * 30)
        
        # Model oluştur
        model = LogisticRegression(learning_rate=lr, num_iterations=1000)
        
        # Modeli eğit
        model.fit(X_train, y_train)
        
        # Cost history grafiğini çiz
        plt.figure(figsize=(10, 6))
        plt.plot(model.costs)
        plt.title(f'Cost Function Değişimi (Learning Rate: {lr})')
        plt.xlabel('İterasyon')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()
        
        # Eğitim seti üzerinde doğruluk oranını hesapla
        train_predictions = model.predict(X_train)
        train_accuracy = calculate_accuracy(y_train, train_predictions)
        print(f"Eğitim doğruluk oranı: {train_accuracy:.2%}")
        
        # Test verileri üzerinde tahmin yap
        predictions = model.predict(X_test)
        
        # Sonuçları yazdır
        print("\nTest Sonuçları:")
        print("-" * 50)
        
        for i, (x, pred) in enumerate(zip(X_test, predictions)):
            print(f"\nÖğrenci {i+1}:")
            print(f"Çalışma saati: {x[0]} saat")
            print(f"Önceki sınav notu: {x[1]}")
            print(f"Tahmin: {'Başarılı' if pred == 1 else 'Başarısız'}")
        
        # Model parametrelerini yazdır
        print("\nModel Parametreleri:")
        print(f"Ağırlıklar: {model.weights}")
        print(f"Bias: {model.bias}")

if __name__ == "__main__":
    main() 