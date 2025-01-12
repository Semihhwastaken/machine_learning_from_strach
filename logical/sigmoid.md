# Lojistik Regresyon Detaylı Açıklama

## 1. Sınıf Yapısı ve Başlangıç

```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
```
- `learning_rate`: Bu parametre, modelin her adımda ağırlıklarını ne kadar değiştireceğini belirler. 
Öğrenme hızı, modelin daha iyi bir tahmin yapmayı öğrenebilmesi için temel bir parametredir.
Gradient Descent (Gradyan İnişi) gibi algoritmalarda, öğrenme hızı, ağırlıkların güncellenmesinde kullanılan adım büyüklüğüdür:

                    𝑊yeni = 𝑊eski − learning_rate × gradient

- `num_iterations`: Bu parametre, modelin eğitimi sırasında kaç adım atacağını belirler. Daha fazla iterasyon, modelin daha iyi bir tahmin yapmayı öğrenebilmesi için daha iyi sonuçlar verir. Çok fazla iterasyon overfittinge neden olabilir.


- `weights`: Modelin her bir özelliğe (feature) verdiği önemi temsil eden öğrenilebilir parametrelerdir. Ağırlıklar genellikle başlangıçta sıfır veya rastgele bir değerle başlatılır ve her iterasyonda öğrenme algoritması tarafından güncellenir. Bir regresyon veya sınıflandırma problemi çözülürken, ağırlıklar modelin girdilere (örneğin, bir veri kümesindeki sütunlar) nasıl tepki vereceğini belirler.


- `bias`: Modelin, girdiler sıfır olduğunda bile bir çıkış değeri (offset) üretmesini sağlayan bir öğrenilebilir parametredir. Bias, modelin veriye uyum sağlayabilmesi için dengeleme yapar.
Matematiksel olarak, bias bir sabit terimdir ve çıktıya eklenir:

                        𝑦tahmin = 𝑊 ⋅ 𝑋 + 𝑏

Burada b, girdilere bağlı olmayan bir parametredir. Bias, modeli daha esnek hale getirir ve yalnızca doğrusal olmayan ilişkileri öğrenmesini kolaylaştırmaz, aynı zamanda genel doğrusal ilişkilerde de dengeleyici bir rol oynar.

```python
    def sigmoid(self, z):
        # Taşmayı önlemek için z değerini sınırla
        if z < -20:
            return 0
        elif z > 20:
            return 1
        try:
            return 1 / (1 + math.exp(-z))
        except OverflowError:
            return 0 if z < 0 else 1
```

Matematiksel olarak sigmoid fonksiyonu şu şekilde tanımlanır:

                        𝜎(𝑧) = 1 / (1 + 𝑒^(-𝑧))
                        z = 𝑊 ⋅ 𝑋 + 𝑏
                        W = ağırlıklar
                        X = girdiler
                        b = bias

`z`: Modelin girdisi veya ağırlıklar ve bias ile hesaplanmış bir lineer kombinasyon.
`e`: Doğal logaritmanın tabanı (Euler sabiti, yaklaşık olarak 2.718).

1- `Çıkış Aralığı (Range):`

Sigmoid fonksiyonu, herhangi bir z girdisini 0 ile 1 arasında bir değere dönüştürür.
Bu, olasılık gibi yorumlanabilir, yani çıktı bir sınıfa ait olma ihtimali olarak düşünülebilir.

2- `Monotonluk:`

Fonksiyon her zaman artandır. Yani, z büyüdükçe çıktı da büyür.

3- `Asimptotlar:`

z→ − ∞ olduğunda 𝜎(𝑧)→0.
z→ + ∞ olduğunda 𝜎(𝑧)→1.

4- `Eşik Davranışı:`

z=0 olduğunda, sigmoid fonksiyonu tam olarak 0.5 değerini döndürür:

                        𝜎(0) = 1 / (1 + 𝑒^0) = 0.5


```python
    def fit(self, X, y):
        # Veriyi normalize et
        X_normalized = self._normalize_features(X)
        """
        Veri setindeki her özelliğin (örneğin yaş, gelir, vb.) farklı bir ölçeği olabilir. Bir özellik 0-1 arasında değerler alırken bir diğeri 0-1000 arasında olabilir. Bu ölçek farklılıkları gradyan inişinin düzgün çalışmasını engeller. Bu yüzden tüm özellikleri 0 ile 1 arasına çekiyoruz. Bu işleme min-max normalizasyonu denir.
        """
        
        num_samples = len(X)
        num_features = len(X[0])

        """
        num_samples: Kaç tane örnek (veri noktası) olduğunu belirler.
        num_features: Her örnekte kaç tane özellik (bağımsız değişken) olduğunu belirler.
        Örneğin:
        Eğer X şu şekilde bir matrisse:

        𝑋 = [25, 50,
            30, 60, 
            35, 70]

        Burada num_samples = 3 (3 satır, yani 3 veri noktası)
        num_features = 2 (2 sütun, yani 2 özellik)
        """
        
        self.weights = [0] * num_features
        self.bias = 0

        """
        weights: Her özelliğe ait bir ağırlık değeri (başlangıçta sıfır). Bu, modelin her özelliğin çıktı üzerindeki etkisini öğrenmesine yardımcı olur.
        bias: Modelin sabit bir kaydırma değeri (başlangıçta sıfır).
        Örneğin: Eğer 2 özellik varsa, weights = [0, 0] olur.
        """
        
        for _ in range(self.num_iterations):
            linear_pred = [sum(x_i * w_i for x_i, w_i in zip(x, self.weights)) + self.bias 
                         for x in X_normalized]
            predictions = [self.sigmoid(z) for z in linear_pred]

            """
            num_iterations: Weights and bias güncellenir, modelin tahminini iyileştirir.
            Bu, doğrusal bir fonksiyondur:

            𝑧 = 𝑤1𝑥1 + 𝑤2𝑥2 + ⋯ + 𝑤𝑛𝑥𝑛 + 𝑏
            x_i: Eğitim verisindeki bir örneğin özellik değerleri.
            w_i: O özelliğin ağırlığı.
            b: Bias (modelin sabit değeri).
            Her veri noktası için bu hesaplanır. Örneğin:

            Eğer weights = [0.5, 1], bias = 0, ve bir örnek x = [2, 3] ise:
            𝑧 = 0.5 ⋅ 2 + 1⋅3 + 0 = 4

            Örneğin, bir z değeri 4 olduğunda:

            σ(4) = 1 / (1 + e^(-4)) ≈ 0.982

            Bu değer, tahmin edilen olasılığı temsil eder.
            """
            
            dw = [0] * num_features
            db = 0
            
            for i in range(num_samples):
                difference = predictions[i] - y[i]
                for j in range(num_features):
                    dw[j] += X_normalized[i][j] * difference
                db += difference
            
            """
            Gradyanlar şu şekilde hesaplanır:

            ∂Loss/∂wj = 1/N ∑i=1N [(pi - yi) * xij]
            ∂Loss/∂b = 1/N ∑i=1N (pi - yi)

            """
            # Gradient descent güncelleme
            for j in range(num_features):
                self.weights[j] -= self.learning_rate * (dw[j] / num_samples)
            self.bias -= self.learning_rate * (db / num_samples)
            """
            wj = wj - α * ∂wj / ∂Loss -> gradient descent -> a = learning_rate
            b = b - α * ∂b / ∂Loss
            """
```

```python
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
```

`Neden Binary Cross-Entropy Kullanıyoruz?`
Bu maliyet fonksiyonu, lojistik regresyonun olasılık temelli doğasına uygundur:

Olasılık Hesapları: Sigmoid fonksiyonunun çıktılarını doğrudan olasılık olarak yorumlamamızı sağlar.
Kayıp Hesapları: Modelin 1 veya 0 sınıfı için doğru tahmin yapma becerisini ölçer.
Ayrıca:

Doğrusal regresyondaki gibi kare farklar (MSE) yerine bu yöntemi kullanırız çünkü:
Olasılıklar (0-1 aralığında) için kare farklar daha az uygun bir hata metriğidir.
Cross-Entropy, tahminlerin olasılık olmasını daha doğru ödüllendirir.







