```python
class RidgeRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha  # regularizasyon parametresi
        self.max_iter = max_iter  # maksimum iterasyon sayısı
        self.tol = tol  # tolerans değeri
        self.weights = None
        self.bias = None
```
`alpha (Regularization Parameter)`
    Tanım: alpha Ridge Regression için çok önemli bir parametredir. Bu, modelde kullanılan regularization (düzenleme) gücünü kontrol eder.

    Amaç: Eğer yalnızca doğruluk odaklı bir model eğitirsek, model tüm veriyi "ezberleyebilir". Buna overfitting (aşırı öğrenme) denir.
    Regularization, modelin ağırlıklarını biraz "kısıtlayarak" bu aşırı öğrenmenin önüne geçer.
    alpha bu kısıtlamanın ne kadar güçlü olduğunu belirler.

`max_iter (Maximum Iteration)`
    Tanım: Gradyan inişi (gradient descent) gibi optimizasyon yöntemlerinde, model ağırlıklarını güncellerken belirli bir sayıda iterasyon yapılır. max_iter, bu sürecin maksimum kaç kez tekrar edileceğini belirler.

    Amaç: Eğer model istenilen çözümü bulamıyorsa (örneğin, eğri düzgün oturmuyorsa), bu sayede bir durdurma kriterimiz olur. Sonsuza kadar çalışmaz!

`tol (Tolerance)`
    Tanım: Bu, yakınsama kontrolü için bir eşik değeridir. Modelin ağırlıkları (weights) ve bias'ı güncellenirken, her bir iterasyondaki değişim çok küçük bir değerin altına düştüğünde eğitim durdurulur.

    Amaç: Bu tolerans değeri, modelin ne zaman yeterince iyi bir çözüm bulduğunu anlamak için kullanılır. Böylece gereksiz iterasyonlardan kaçınırız.

```python
def _normalize(self, X):
        # Özellikleri normalize et
        means = [sum(col) / len(col) for col in zip(*X)]
        stds = [sum((x - m) ** 2 for x in col) ** 0.5 / len(col) ** 0.5 
               for col, m in zip(zip(*X), means)]
        X_norm = [[(x - m) / s if s != 0 else 0 
                   for x, m, s in zip(row, means, stds)] 
                  for row in X]
        return X_norm, means, stds
```
`sum(col) / len(col)` -> Tüm sütunların ortalamasını alır.
`sum((x - m) ** 2 for x in col) ** 0.5 / len(col) ** 0.5` -> Tüm sütunların standart sapmasını alır.
`X_norm = [[(x - m) / s if s != 0 else 0 `
                   `for x, m, s in zip(row, means, stds)] `
                  `for row in X]` -> Tüm sütunları normalize eder.

`Normalizasyon Nedir?`

Normalizasyon, verileri belirli bir ölçeğe dönüştürme işlemidir. Örneğin, tüm özelliklerin ortalamasını 0, standart sapmasını 1 yapabilirsiniz.
Normalizasyonun amacı, her özelliğin model üzerinde eşit ağırlığa sahip olmasını sağlamaktır. Özellikle doğrusal modeller (örneğin Ridge veya Lasso) ve gradient tabanlı yöntemlerde (örneğin SGD) bu önemlidir.

```python
def fit(self, X, y):
        # Veriyi normalize et
        X_norm, self.means, self.stds = self._normalize(X)
        n_samples = len(X)
        n_features = len(X[0])
        
        # Ağırlıkları ve bias'ı başlat
        self.weights = [0.0] * n_features
        self.bias = 0.0
        
        # Gradyan inişi
        for _ in range(self.max_iter):
            weights_old = self.weights.copy()
            
            # Her örnek için gradyanları hesapla
            for i in range(n_samples):
                # Tahmin
                y_pred = sum(w * x for w, x in zip(self.weights, X_norm[i])) + self.bias
                
                # Hata
                error = y_pred - y[i]
                
                # Ağırlıkları güncelle
                for j in range(n_features):
                    self.weights[j] -= (2 * error * X_norm[i][j] + 
                                      2 * self.alpha * self.weights[j]) / n_samples
                
                # Bias'ı güncelle
                self.bias -= 2 * error / n_samples
            
            # Yakınsama kontrolü
            if all(abs(w - w_old) < self.tol 
                  for w, w_old in zip(self.weights, weights_old)):
                break
```
# Ridge Regresyon Nedir?
    Ridge regresyon, doğrusal regresyonun bir çeşididir ve overfitting problemini önlemek için modelin ağırlık parametrelerine bir ceza (𝐿2-norm) ekler. Ridge regresyonun optimizasyon problemi şu şekildedir:

    Kayıp fonksiyonu:
$$J(w,b) = \frac{1}{m} \sum_{i=1}^m (h_{w,b}(x^{(i)}) - y^{(i)})^2 + \alpha\|w\|^2$$

    Tahmin fonksiyonu:
$$h_{w,b}(x) = w^{\top}x + b$$

    Açık hali:
$$h_{w,b}(x^{(i)}) = w_1x_1^{(i)} + w_2x_2^{(i)} + ... + w_nx_n^{(i)} + b$$

    Yani kayıp fonksiyonu şu şekilde açılır:
$$J(w,b) = \frac{1}{m} \sum_{i=1}^m (w_1x_1^{(i)} + w_2x_2^{(i)} + ... + w_nx_n^{(i)} + b - y^{(i)})^2 + \alpha(w_1^2 + w_2^2 + ... + w_n^2)$$

    Burada:
- $J(w,b)$: Kayıp fonksiyonu
- $\|w\|^2 = \sum_{j=1}^d w_j^2$: L2-norm ceza terimi
- $\alpha$: Ceza teriminin ağırlığını belirleyen hiperparametre


# Kodun Mantığı
                        self.weights = [0.0] * n_features
                        self.bias = 0.0
    
    Ağırlıklar (𝑤): Her özelliğin katsayısını tutar. İlk başta hepsi 0.0 olarak başlatılır.
    Bias (𝑏): Tahmin fonksiyonundaki sabit terim. İlk başta 0.0 olarak başlatılır.

    Gradyan inişi, Ridge regresyonun kayıp fonksiyonunu minimize etmek için kullanılır. Kodun bu kısmı aşağıdaki adımlardan oluşur:

    Tahmin fonksiyonu:
$$y_{pred} = \sum_{j=1}^d w_jx_j + b$$
    Burada:
- $w_j$: Ağırlıklar
- $x_j$: Girdinin normalize edilmiş özellikleri

    Hata (error):
$$error = y_{pred} - y$$

    Gradyan inişi formülü:
$$w_j \leftarrow w_j - \eta \frac{\partial J}{\partial w_j}$$

    Ridge Regresyon için gradyan inişi formülü:
$$\frac{\partial J}{\partial w_j} = \frac{2}{m} \left[\sum_{i=1}^m (y_{pred}^{(i)} - y^{(i)})x_j^{(i)}\right] + 2\alpha w_j$$

    Burada:
- $\frac{\partial J}{\partial w_j}$: J fonksiyonunun $w_j$'ye göre kısmi türevi
- $m$: örnek sayısı
- $y_{pred}^{(i)}$: i. örnek için tahmin değeri
- $y^{(i)}$: i. örnek için gerçek değer
- $x_j^{(i)}$: i. örneğin j. özelliği
- $\alpha$: regularizasyon parametresi
- $w_j$: j. ağırlık

    Bias güncelleme formülü:
$$\frac{\partial J}{\partial b} = \frac{2}{m} \sum_{i=1}^m (y_{pred}^{(i)} - y^{(i)})$$

    Burada:
- $\frac{\partial J}{\partial b}$: J fonksiyonunun b'ye göre kısmi türevi
- $m$: örnek sayısı
- $y_{pred}^{(i)}$: i. örnek için tahmin değeri
- $y^{(i)}$: i. örnek için gerçek değer

# 	Regularizasyonun Rolü

    Overfitting ve Underfitting:
    Ridge regresyonun α hiperparametresi overfitting ve underfitting arasında denge kurar:
    𝛼 = 0: Ridge, klasik doğrusal regresyona döner.
    𝛼 → ∞: Tüm ağırlıklar sıfıra yaklaşır (underfitting).

    Ridge regresyonda ağırlıklar küçültülür, ancak tamamen sıfırlanmaz.

# Hiperparametre Seçimi (𝛼)
    Hiperparametre Seçiminin Önemi: Ridge regresyonda 𝛼, modelin genel performansını etkiler.
    Küçük 𝛼: Daha az ceza (overfitting riski artar).
    Büyük 𝛼: Daha fazla ceza (underfitting riski artar).

# α Nasıl Seçilir?
    Cross-validation (CV): Veri setini eğitim ve doğrulama kısımlarına bölerek en iyi 𝛼 değerini seçebilirsin.
    Grid Search: Çeşitli 𝛼 değerlerini deneyip doğrulama setindeki hatayı minimize eden 𝛼 değerini seçebilirsin.
    Bayesian Optimization veya Random Search: Hiperparametre optimizasyonu için daha gelişmiş yöntemler.

# Gradyan inişi nedir?

    Gradyan inişi (Gradient Descent), makine öğreniminde optimizasyon problemlerini çözmek için kullanılan en popüler yöntemlerden biridir. Farklı gradyan inişi türleri, veri miktarına, problem boyutuna ve performans gereksinimlerine göre tercih edilir. İşte Gradient Descent çeşitleri ve bunların ayrıntılı açıklamaları:

## 1. Batch Gradient Descent (Tam Yığın Gradyan İnişi)

    Nasıl Çalışır?

    Gradyan, tüm veri seti (𝑋 ve 𝑦) üzerinde hesaplanır.
    Ağırlıklar, tüm veri setindeki hata fonksiyonunun gradyanına göre güncellenir.
    Matematiksel Formül:
$$w = w - \eta \cdot \nabla_w J(w)$$
    𝑤: Ağırlık vektörü.
    𝜂: Öğrenme oranı.
    𝐽(𝑤): Tüm veri setindeki hata fonksiyonu.
    ∇𝑤𝐽(𝑤): Tüm veri seti üzerindeki gradyan.

    Avantajları:
        Her güncellemede en doğru gradyan bilgisi kullanılır.
        Kararlıdır ve daha iyi bir yakınsama sağlar.

    Dezavantajları:
        Büyük veri setlerinde hesaplama maliyeti yüksektir (her adımda tüm veri setini tarar).
        Bellek tüketimi fazladır.

## 2. Stochastic Gradient Descent (SGD) (Stokastik Gradyan İnişi)

    Nasıl Çalışır?

    Gradyan, her bir veri örneği için hesaplanır ve ağırlıklar buna göre güncellenir. Yani, her adımda tek bir veri örneği kullanılır.
    Matematiksel Formül:
$$w = w - \eta \cdot \nabla_w J(w; x_i, y_i)$$
    xi,yi: Tek bir veri örneği.

    Avantajları:
        Büyük veri setlerinde daha hızlıdır.
        Hesaplama maliyeti düşüktür.
        Daha kolay bellek yönetimi (tek bir veri örneği ile çalışır).

    Dezavantajları:
        Hata fonksiyonu (loss) her adımda oldukça dalgalanır.
        Daha düşük doğruluk veya yavaş yakınsama yaşanabilir.
        Yakınsama kontrolü zor olabilir.

## 3. Mini-Batch Gradient Descent

    Nasıl Çalışır?

    Tüm veri setini küçük gruplara (mini-batch) böler. Her mini-batch için gradyan hesaplanır ve ağırlıklar güncellenir.
    Matematiksel Formül:
$$w = w - \eta \cdot \nabla_w J(w; X_{batch}, y_{batch})$$
    𝑋𝑏𝑎𝑡𝑐ℎ, 𝑦𝑏𝑎𝑡𝑐ℎ: Mini-batch verisi.

    Avantajları:
        Batch ve SGD yöntemlerinin avantajlarını birleştirir.
        Daha dengeli ve daha hızlı yakınsama sağlar.
        GPU gibi paralel işleme birimleri için optimize edilebilir.

    Dezavantajları:
        Mini-batch boyutunun uygun seçilmesi gerekir (çok büyük veya çok küçük olursa sorun yaratır).

# Coordinate Descent

    Coordinate Descent (Koordinat İnişi), optimizasyon ve makine öğrenmesi algoritmalarında sıkça kullanılan bir yöntemdir. Temel olarak, çok değişkenli bir fonksiyonun minimum veya maksimum değerini bulmaya yönelik bir iteratif optimizasyon tekniğidir. Adından da anlaşılacağı gibi, bu yöntem, her seferinde bir parametreyi (koordinat) optimize ederek ilerler.

    Temel Kavramlar
    - Koordinat: Optimizasyon problemi genellikle çoklu değişkenlerden oluşur. Bu değişkenlerin her birini bir koordinat olarak düşünebiliriz. Örneğin, iki değişkenli bir fonksiyon 
$$f(x_1, x_2)$$ 
    olsun. Buradaki
$$(x_1, x_2)$$
    değişkenleri birer koordinattır.

    -İniş (Descent): İniş, bir fonksiyonun minimumunu bulmak için fonksiyonun değerini azaltma yönünde bir adım atmayı ifade eder. Her adımda, fonksiyonun değeri azalır.

    Koordinat İnişi Nasıl Çalışır?
        Coordinate Descent algoritması şu şekilde çalışır:

        1. Başlangıç Noktasını Seçme: İlk olarak bir başlangıç noktası seçilir. Bu nokta, optimizasyonun yapılacağı parametre uzayının bir noktasıdır.

        2. Koordinat Seçimi: Ardından, parametrelerin her birini (veya koordinatları) sırasıyla tek tek optimize ederiz. Yani, f(x_1, x_2, x_3, ..., x_n) fonksiyonu üzerinde, her bir parametre için sırayla optimize edilir.

        3. Her Koordinatın Optimize Edilmesi: Bir parametreyi optimize ederken, diğer parametreler sabit tutulur. Bu, bir tür tek değişkenli optimizasyon problemine indirgenir. Örneğin, x_1 parametresi için optimize ederken, x_2, x_3, ..., x_n sabit tutulur.

        4. Adım Adım İlerleme: Bu işlem, her koordinat için optimize edilene kadar tekrarlanır. Her adımda, sadece bir parametre değiştirilir ve sonuçlar bir sonraki adımda kullanılır.

        5. Tekrarlar: Yöntem, tüm parametreler optimize edilene kadar (ya da bir durma kriteri sağlanana kadar) devam eder.
