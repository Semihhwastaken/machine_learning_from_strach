# 1. K-Neighbors Classifier Nedir?
K-Neighbors Classifier (KNN), temelinde örnek tabanlı öğrenme yöntemlerinden biridir. Yani, KNN modeli eğitim sırasında herhangi bir öğrenme işlemi yapmaz. Bunun yerine, tüm eğitim verilerini bellekte saklar ve bir test örneği geldiğinde bu örneği komşularıyla karşılaştırarak bir tahmin yapar.

Kısacası:
- Eğitim sırasında sadece verileri saklar.
- Tahmin yaparken, bir test örneğini eğitim veri kümesindeki en yakın k komşusuyla kıyaslar.
- Çoğunluğa dayalı bir karar verir

# 2. Nasıl Çalışır?
KNN algoritması şu adımlarla çalışır:

### 1. Mesafe Ölçümü
KNN, test örneği ile eğitim verisindeki her örnek arasındaki mesafeyi hesaplar. En sık kullanılan mesafe ölçümleri şunlardır:

- Öklid Mesafesi (𝐿2 normu):

$$d(x,y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$

- Manhattan Mesafesi (𝐿1 normu):

$$d(x,y) = \sum_{i=1}^n |x_i - y_i|$$

- Minkowski Mesafesi (𝐿p normu):

$$d(x,y) = (\sum_{i=1}^n |x_i - y_i|^p)^{\frac{1}{p}}$$

### 2. Komşuları Belirleme
Tüm mesafeler hesaplandıktan sonra, en küçük mesafeye sahip k adet komşu seçilir. Bu komşular test örneğine en yakın olan eğitim örnekleridir.

### 3. Tahmin Yapma
Sınıflandırma: Seçilen k komşunun sınıflarına bakılır ve çoğunluğun sınıfı test örneği için tahmin edilir.
Regresyon: Seçilen k komşunun etiket değerlerinin ortalaması alınır.

# 3. Hiperparametreler
KNN algoritmasında, birkaç önemli hiperparametre vardır:

### 1. k (Komşu Sayısı):
- k, test örneği için kaç komşunun dikkate alınacağını belirtir.
- Küçük bir k: Model daha esnek olur, ama gürültüye duyarlı hale gelir (overfitting).
- Büyük bir k: Model daha kararlı hale gelir, ama genel yapıyı kaçırabilir (underfitting).

### 2. Mesafe Ölçütü:
- Kullanılan mesafe ölçütü modelin başarısını doğrudan etkiler. Örneğin:
- Öklid mesafesi genelde daha yaygındır.
- Manhattan mesafesi, özelliklerin ölçeklendirilmesi durumunda daha kararlı olabilir.

### 3. Ağırlıklandırma:
- KNN, komşuların ağırlıklarını mesafeye göre ayarlayabilir.
- Uniform: Tüm komşular eşit ağırlık alır.
- Distance-weighted: Daha yakın komşulara daha fazla ağırlık verilir.

# 4. Avantajları ve Dezavantajları
### Avantajları:
- Basit ve Sezgisel: Çok az matematiksel karmaşıklık içerir.
- Eğitim Süresi Çok Hızlıdır: Model oluştururken sadece veriler saklanır.
- Çok Yönlü: Hem sınıflandırma hem de regresyonda kullanılabilir.

### Dezavantajları:
- Tahmin Süresi Yavaş: Her tahminde tüm veri kümesini dolaşır, bu da büyük veri kümelerinde yavaştır.
- Bellek Tüketimi Yüksek: Eğitim verisinin tamamını saklamak gerekir.
- Özellik Ölçeklendirmesi Gerektirir: Mesafeler özelliklerin ölçeğine duyarlı olduğundan, verilerin standartlaştırılması önemlidir.
- Gürültüye Duyarlı: Gürültülü veriler küçük k değerlerinde hatalı tahminlere yol açabilir.

# 5. Python ile KNN Uygulaması

```python
class KNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Eğitim verilerini saklar"""
        self.X_train = X
        self.y_train = y
```
### 1. __init__ metodu
    Bu metod, sınıfın başlatıcı metodudur ve sınıfın bir örneği oluşturulduğunda otomatik olarak çalışır. Bu metodda:
        - n_neighbors: Kullanıcıya komşu sayısını belirleme şansı tanır. Eğer kullanıcı bir değer girmezse, varsayılan olarak 5 komşu kullanılır.
        - X_train ve y_train: Eğitim verilerini saklayacak boş değişkenlerdir. Eğitim süreci başladığında bu verilere değer atanacaktır.

### 2. fit metodu
    Bu metod, modeli eğitmek için kullanılan metoddur. KNN algoritması model eğitimi sırasında hiçbir karmaşık hesaplama yapmaz, çünkü KNN bir instance-based (örnek tabanlı) algoritmadır.
        - Eğitim Verilerinin Saklanması: Bu adımda, gelen eğitim verisi (özellikler ve etiketler) modelin belleğine kaydedilir.
        - self.X_train: Eğitim özellikleri (özellikler matrisi).
        - self.y_train: Eğitim etiketleri (etiketler vektörü).

```python
def _euclidean_distance(self, x1, x2):
        """İki nokta arasındaki Öklid mesafesini hesaplar"""
        distance = 0
        for i in range(len(x1)):
            distance += (x1[i] - x2[i]) ** 2
        return distance ** 0.5
```

### 3. _euclidean_distance metodu
```python

    def _euclidean_distance(self, x1, x2):
        """İki nokta arasındaki Öklid mesafesini hesaplar"""
        distance = 0
        for i in range(len(x1)):
            distance += (x1[i] - x2[i]) ** 2
        return distance ** 0.5
```
    Bu metod, Öklid mesafesi hesaplamak için kullanılır. İki nokta arasındaki mesafeyi hesaplar, genellikle KNN algoritmasında komşuların yakınlık derecelerini belirlemek için kullanılır.

### 4. _get_neighbors metodu
```python
    def _get_neighbors(self, test_sample):
        """Test örneği için en yakın k komşuyu bulur"""
        distances = []
        for i, train_sample in enumerate(self.X_train):
            dist = self._euclidean_distance(test_sample, train_sample)
            distances.append((dist, self.y_train[i]))
        
        # Mesafelere göre sırala ve en yakın k komşuyu al
        distances.sort(key=lambda x: x[0])
        return [d[1] for d in distances[:self.n_neighbors]]
```
    Bu metod, verilen bir test örneği için en yakın k komşuyu bulur. KNN algoritmasında tahmin yaparken, en yakın k komşunun sınıfları kullanılır. Bu adım şu şekilde işler:

        - test_sample: Bu, tahmin yapılacak test örneğidir.
        - Mesafelerin Hesaplanması: Eğitim veri kümesindeki her bir örnek ile test örneği arasındaki Öklid mesafesi hesaplanır.
        - Mesafelerle Birleştirilmiş Etiketler: Her mesafe ile ilgili etiket, mesafelerin bir listesini oluşturur. Yani her bir mesafe ile birlikte hangi sınıfa ait olduğu bilgisi tutulur.
        - Komşu Seçimi: Mesafeler sıralanır ve en küçük k mesafeye sahip komşular seçilir.
        - Bu metodun amacı, test örneği için en yakın k komşuyu bulmaktır.

### 5. predict metodu
```python
    def predict(self, X):
        """Test örnekleri için tahminlerde bulunur"""
        predictions = []
        for test_sample in X:
            neighbors = self._get_neighbors(test_sample)
            # En sık görülen sınıfı bul
            prediction = max(set(neighbors), key=neighbors.count)
            predictions.append(prediction)
        return predictions
```
    Bu metod, modelin tahmin yapma işlevini yerine getirir.
    - Test Verisi Üzerinde Tahmin: Bu metot, birden fazla test örneği alabilir (yani bir veri kümesi) ve her bir örnek için tahmin yapar.
    - Komşu Seçimi: Her bir test örneği için _get_neighbors metodu çağrılır ve en yakın k komşu bulunur.
    - Sınıf Tahmini: En yakın k komşusunun sınıflarına bakılır ve bu sınıflar arasında çoğunluğu belirlenir (en sık görülen sınıf seçilir).
    - En sık görülen sınıf, test örneği için yapılan tahmindir. Python'da bunun için max(set(neighbors), key=neighbors.count) kullanılır. Bu, komşuların sınıf sayısını sayar ve en fazla görülen sınıfı seçer.

