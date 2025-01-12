# 1. Feature Selection Nedir?
Feature selection, veri setindeki özelliklerin bir alt kümesini seçerek daha iyi bir performans veya daha hızlı bir model oluşturmayı hedefler. İdeal bir özellik seçimi:
    Aşırı öğrenmeyi azaltır.
    Hesaplama maliyetini düşürür.
    Modelin genelleme yeteneğini artırır.
    Feature selection yöntemleri üç ana kategoriye ayrılır:
        Filter Methods: İstatistiksel ölçütlere dayanır. Örneğin, korelasyon katsayısı veya chi-square testi.
        Wrapper Methods: Özellik alt kümeleri deneyerek en iyi kombinasyonu arar.
        Embedded Methods: Özellik seçimini model eğitimine entegre eder. Örneğin, Lasso Regression.

# 2. Brute Force Algoritması
Brute Force, tüm olası özellik kombinasyonlarını test ederek en iyi kombinasyonu seçmeyi hedefler. Bu yöntem, basit ama pahalı bir yöntemdir, çünkü özellik sayısı arttıkça kombinasyon sayısı üstel olarak artar.

`Nasıl Çalışır?`
```	
1- Tüm olası özellik kombinasyonları oluşturulur.
2- Her kombinasyon için bir model eğitilir ve bir metrik (örneğin doğruluk, F1 skoru) hesaplanır.
3- En iyi metriğe sahip olan kombinasyon seçilir.


Avantajları:

- En iyi çözümü garanti eder (optimal).
- Özelliklerin model üzerindeki etkisini net bir şekilde görebiliriz.

Dezavantajları:

- Hesaplama maliyeti çok yüksektir. Eğer 𝑛 özellik varsa, 
- 2^n−1 kombinasyon test edilmelidir.
- Büyük veri setleri için pratik değildir.
```
# 3. Griddy Algoritması (Greedy Search)
Griddy Algoritması, brute force'un aksine, daha hızlı bir şekilde "yeterince iyi" çözümler bulmayı hedefler. Bu algoritma, her adımda en iyi yerel seçimi yapar, yani mevcut durumdaki en iyi özelliği seçer.

`Nasıl Çalışır?`
```	
1- Boş bir özellik kümesiyle başlanır.
2- Her adımda, eklenmesi model performansını en çok artıran özelliği ekler.
3- Belirli bir durdurma kriterine ulaşıldığında (örneğin, model performansında iyileşme olmaması) süreç sona erer.


`Avantajları:`

- Daha hızlıdır çünkü tüm kombinasyonları test etmez.
- Özellikle büyük veri setlerinde kullanılabilir.

`Dezavantajları:`

- Optimal çözümü garanti etmez (yerel maksimuma takılabilir).
- Performans metriklerinin seçim sırasına bağımlılığı vardır.
```
# 4. Regularization ile Özellik Seçimi
Düzenlileştirme, özellik seçimi için kullanılabilir ve bir modelin karmaşıklığını kontrol ederek daha iyi genelleştirme performansı sağlar. Brute force veya griddy algoritmalar gibi tüm olası kombinasyonları aramak yerine, düzenlileştirme yöntemleri bir ceza terimi ekleyerek gereksiz özelliklerin etkisini azaltır veya tamamen sıfıra indirir.

## `L1 Düzenlileştirme (Lasso Regression)`
    L1 düzenlileştirme, kayıp fonksiyonuna katsayıların mutlak değerlerinin toplamını bir ceza olarak ekler:

$$\text{Kayıp Fonksiyonu} = \text{Hata} + \lambda \sum_{j=1}^n |w_j|$$

    ### Nasıl Çalışır?
   
    - L1 düzenlileştirme, bazı katsayıları sıfıra indirir. Böylece model, sıfır olan özellikleri tamamen göz ardı eder.
    - Sıfır olmayan katsayıya sahip özellikler seçilmiş olur.
    - Lasso (L1 Regularization) için alpha parametresi, düzenlileştirme gücünü kontrol eden bir hiperparametredir. Bu parametre, modelin ne kadar sıkı düzenlileştirme uygulayacağını belirler ve bu durum, seçilen özelliklerin sayısını ve modelin davranışını doğrudan etkiler.
        1. Küçük Alpha Değerleri:
            - Düzenlileştirme Gücü Düşük: Küçük bir alpha değeri, düzenlileştirme cezasını azaltır ve model daha fazla esneklik kazanır.
            - Sonuç: Daha fazla özellik seçilir (daha az özellik elenir). Modelin genelleştirme gücü azalabilir ve aşırı öğrenme (overfitting) riski artar.
        2. Büyük Alpha Değerleri:
            - Düzenlileştirme Gücü Yüksek: Büyük bir alpha değeri, düzenlileştirme cezasını artırır ve model, gereksiz özelliklerin katsayılarını sıfıra çekmeye zorlanır.
            - Sonuç: Daha az özellik seçilir (daha fazla özellik elenir). Model daha sade hale gelir, ancak aşırı düzenlileştirme (underfitting) riski artabilir.
        3. Alpha = 0:
            - Hiç Düzenlileştirme Yok: Bu durumda model sadece klasik lineer regresyon gibi davranır ve hiçbir özellik elenmez. 
            - Sonuç: Tüm özellikler modelde kalır. Düzenlileştirme olmadan, modelin aşırı öğrenme riski artar.
    

        Avantajları:

            - Özellik seçimi doğrudan yapılır.
            - Yüksek boyutlu veri kümeleri için oldukça etkili.

        Dezavantajları:

            - Birbirine çok benzeyen (yüksek korelasyona sahip) özellikler varsa, bunlardan sadece birini seçer.

```python
class LassoRegression:
    def __init__(self, learning_rate=0.01, lambda_param=1.0, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
```
```
- learning_rate: Gradient descent'in adım büyüklüğüdür.
- lambda_param: Regularization gücünü kontrol eden hiperparametredir.
- n_iterations: Gradient descent'in kaç adımda tamamlanacağını belirler.
- weights: Ağırlıkların başlangıç değerlerini içerir.
- bias: Bias'ın başlangıç değerini içerir.
```

```python
def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])

        """
        X: Eğitim verisi, her bir satır bir örneği (veri noktasını), her bir sütun ise bir özelliği temsil eder.
        n_samples: Veri setindeki toplam örnek sayısını verir.
        n_features: Veri setindeki toplam özellik (feature) sayısını verir.
        """
        self.weights = [0] * n_features
        self.bias = 0

        """
        self.weights: Her özellik için öğrenilecek ağırlıkları temsil eder. Başlangıçta sıfır olarak ayarlanır.
        self.bias: Modelin sabit terimidir (yani tahminlerde tüm verilere eklenir). Başlangıçta sıfır olarak ayarlanır.
        """
        
        for _ in range(self.n_iterations):
            """iterasyon sayısı kadar döngü"""
            y_pred = self._predict(X)
            """
            _predict metodu, mevcut ağırlıklar ve bias kullanılarak tahmin edilen değerleri döndürür:
            𝑦_i = 𝑤 * 𝑋_i + 𝑏
            
            Bu tahminler, mevcut model parametrelerinin (weights ve bias) ne kadar doğru olduğunu anlamak için kullanılır.
            """
            
            dw = [0] * n_features
            db = 0
            """
            dw: Her bir ağırlık (𝑤_j) için gradyanı (türevini) tutar. Lasso düzenlileştirme için güncelleme burada yapılır.
            db: Bias (𝑏) için gradyanı tutar. 
            Gradyanlar, kayıp fonksiyonunun türevlerini temsil eder ve ağırlıkların hangi yönde güncellenmesi gerektiğini belirtir.
            """
            
            for i in range(n_samples):
                error = y_pred[i] - y[i]
                
                db += error
                
                
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                """
                error: Tahmin (𝑦_pred) ile gerçek değer (𝑦) arasındaki fark.
                Bias gradyanı (𝑑𝑏): Tüm örneklerin hata değerlerinin toplamıdır.
                Ağırlık gradyanı (𝑑𝑤[𝑗]): Her özellik için hata (𝑒𝑟𝑟𝑜𝑟) ve özellik değeri 
                (𝑋[𝑖][𝑗]) çarpılarak hesaplanır. Bu, her ağırlığın güncellenmesine katkıda bulunur.
                """
            
            
            db /= n_samples
            for j in range(n_features):
                dw[j] = dw[j] / n_samples + self.lambda_param * self._sign(self.weights[j])
                """
                Bias ve ağırlık gradyanları, veri sayısına (𝑛_samples) bölünerek ortalaması alınır.
                Lasso düzenlileştirme etkisi, self.lambda_param * self._sign(self.weights[j]) ile eklenir:
                self._sign(self.weights[j]): Ağırlığın işaretini verir (+1 veya -1). L1 düzenlileştirme, ağırlıkları sıfıra çekme eğilimindedir.
                """
            
            # Ağırlıkları güncelle
            self.bias -= self.learning_rate * db
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * dw[j]
                """
                Gradyan İniş Güncellemesi:
                𝑏 ← 𝑏 − 𝜂*𝑑𝑏
                𝑤_ 𝑗←𝑤_𝑗−𝜂*𝑑𝑤_𝑗
                Burada:
                η: Öğrenme oranı (learning rate). Güncelleme adımının büyüklüğünü belirler.
                Bias ve ağırlıklar, gradyanların ters yönünde güncellenir. Lasso etkisi sayesinde bazı ağırlıklar sıfıra yakınlaşır veya sıfır olur.
                """
```
```python
def _predict(self, X):
        predictions = []
        for xi in X:
            pred = self.bias
            for j in range(len(self.weights)):
                pred += self.weights[j] * xi[j]
            predictions.append(pred)
        return predictions
    
    def predict(self, X):
        return self._predict(X)
    
    def _sign(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        return 0
```

# _predict: Amacı: Modelin tahminler üretmesi.
    X: Girdi verisi (örneklerin ve özelliklerin bulunduğu matris).
    predictions: Tahmin edilen değerlerin (𝑦^) tutulduğu liste.
    Adımlar:
        - for xi in X: Girdideki her örneği (𝑥_𝑖) sırayla işler.
        - Başlangıç Tahmini: pred = self.bias
            - Model tahminine sabit bir başlangıç değeri (𝑏) eklenir.
        - Ağırlık ve Özellik Çarpımı:
            - Her özellik için, ağırlık (𝑤_𝑗) ile özellik değeri (𝑥_𝑖𝑗) çarpılır ve tahmine eklenir:
$$\text{pred} \leftarrow \text{pred} + w_j \cdot x_{ij}$$
        - Tahmin Listesine Ekleme:
            - Her bir örnek için hesaplanan tahmin, predictions listesine eklenir.
    Sonuç:
        - Tüm örnekler için tahminler döndürülür.


# sign:
    - Amacı: Sayının işaretini belirler. Bu, L1 düzenlileştirme sırasında kullanılır.
    - Girdi: x (bir sayı).
    - Çıkış:
        1 -> Eğer  𝑥>0
        -1 -> Eğer 𝑥<0
        0 -> Eğer 𝑥=0

    - `Neden İşaret Fonksiyonu?`
            - Lasso regresyonda 𝐿1 düzenlileştirme terimi, ağırlıkları sıfıra çekmek için kullanılır. Bu, ağırlıkların türevine işaret fonksiyonunun eklenmesini gerektirir:
$$\frac{\partial}{\partial w_j} \lambda |w_j| = \lambda \cdot \text{sign}(w_j)$$


# Ek Matematiksel Bilgiler

    Gradyanlar, kayıp fonksiyonunun türevleri kullanılarak hesaplanır. Kayıp fonksiyonu Lasso Regresyonda şu şekildedir:

$$L(w,b) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^n |w_j|$$
		
    Hataların ortalama karesel değeri (MSE - Mean Squared Error):
$$\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$ 

    L1 düzenlileştirme terimi, ağırlıkların cezalandırılmasını sağlar:
$$\lambda \sum_{j=1}^n |w_j|$$
    




	
