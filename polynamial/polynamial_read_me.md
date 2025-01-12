# Polynomial Regression Nedir?
```
Polynomial regression, basit doğrusal regresyonun bir genelleştirilmiş hâlidir. Doğrusal regresyon, bir bağımlı değişkeni (örneğin, y) bir bağımsız değişkenle (örneğin, x) doğrusal bir ilişkiyle modellemeye çalışırken, polynomial regression, bu ilişkiyi daha karmaşık polinom terimleriyle (örneğin, 𝑥^2,x^3) modellemeye çalışır.

Bu yöntem, veriler arasındaki ilişki doğrusal olmadığında ama yine de matematiksel olarak bir polinom eğrisiyle iyi temsil edilebileceğinde kullanılır.
```

## Matematiksel Modeli
```
Polynomial regression şu şekilde ifade edilebilir:

Polynomial regresyon, bağımlı değişken (y) ile bağımsız değişken (x) arasındaki ilişkiyi n'inci dereceden bir polinom ile modelleyen regresyon analizidir.
```

$$y = \beta_0 + \beta_1x + \beta_2x^2 + \beta_3x^3 + \cdots + \beta_nx^n + \epsilon$$

Burada:
- $y$ : Bağımlı değişken
- $x$ : Bağımsız değişken
- $\beta_0$ : Sabit terim (y-kesişimi)
- $\beta_1, \beta_2, ..., \beta_n$ : Polinom katsayıları
- $n$ : Polinomun derecesi
- $\epsilon$ : Hata terimi


## Nerelerde Kullanılır?
```
Karmaşık İlişkiler: Bağımlı ve bağımsız değişkenler arasında doğrusal olmayan bir ilişki olduğunda.

Verilerin Eğrisel Yapısı: Veriler bir eğri üzerinde yoğunlaşmışsa, polynomial regression bu eğriyi daha iyi modelleyebilir.

Tahmin Modelleri: Gerçek dünya uygulamalarında, örneğin ekonomi, biyoloji, ve mühendislikte yaygın olarak kullanılır.

Avantajlar
   - Basit doğrusal regresyona kıyasla doğrusal olmayan ilişkileri modelleyebilir.
   - Daha yüksek dereceli polinomlar, verilerin daha hassas bir şekilde modellenmesini sağlar.

Dezavantajlar
   - Aşırı Uyum (Overfitting): Polinom derecesi çok yüksek seçilirse, model veriye aşırı uyum sağlayarak genelleme gücünü kaybedebilir.
   - Dengesizlik: Düşük dereceler yetersiz kalabilirken, yüksek dereceler hesaplama maliyetini artırır.
```

# 1. __init__ fonsksiyonu
```python
class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.coefficients = None
```
```
- degree: Polinomial regresyonun derecesini belirtir. Derece 2 ise 𝑥^2, derece 3 ise 𝑥^3 gibi terimler oluşturulur.
- coefficients: Modelin öğrenilen katsayılarını saklar.
```

# 2. __create_polynomail_features__ fonksiyonu
```python
 def _create_polynomial_features(self, X):
        # Çoklu özellikler için polinomial özellikleri oluştur
        n_samples = len(X)
        n_features = len(X[0])
        
        # Başlangıçta bias terimi (1'ler)
        X_poly = [[1] * n_samples]
        
        # Her özellik için lineer terimler
        for j in range(n_features):
            X_poly.append([x[j] for x in X])
        
        # Etkileşim terimleri ve yüksek dereceli terimler
        if self.degree >= 2:
            for d in range(2, self.degree + 1):
                for j in range(n_features):
                    X_poly.append([x[j] ** d for x in X])
                    
                    # Özellikler arası etkileşimler
                    for k in range(j + 1, n_features):
                        X_poly.append([x[j] * x[k] for x in X])
        
        # Dikey matristen yatay matrise çevir
        return [[X_poly[j][i] for j in range(len(X_poly))] 
                for i in range(n_samples)]
```
```
- X: n×m boyutunda, n örnek ve 𝑚 özellikten oluşan veri kümesi (örneğin, bir tablodaki satırlar gözlemleri, sütunlar ise özellikler temsil eder).
- self.degree: Oluşturulacak polinomial terimlerin maksimum derecesi. Örneğin, degree = 2 kare terimleri (𝑥^2) ve çift özellik çarpımlarını (𝑥1 *𝑥2) ekleriz.

- X_poly = [[1] * n_samples] => Bu, her veri örneği için bir bias terimi ekler. Bias terimi, genellikle makine öğrenimi modellerinde 𝑤0 sabitini (bir çeşit başlangıç noktası) temsil eder. Matematiksel olarak, bu şunu ifade eder:
```

$$X_{poly} = \begin{bmatrix} 
1  \\ 
1 \\ 
\vdots \\ 
1 
\end{bmatrix}$$

```
Her özelliği olduğu gibi ekleriz. Eğer X şu şekilde bir veri kümesi ise:
```
$$X = \begin{bmatrix} 
x_{11} & x_{12} \\
x_{21} & x_{22} \\
x_{31} & x_{32}
\end{bmatrix}$$
```
Bu adımın sonunda X_Poly şunu içerir:
```
$$X_{poly} = \begin{bmatrix} 
1 & x_{11} & x_{12} \\
1 & x_{21} & x_{22} \\
1 & x_{31} & x_{32}
\end{bmatrix}$$

```python
for d in range(2, self.degree + 1):
    for j in range(n_features):
        X_poly.append([x[j] ** d for x in X])
         
         for k in range(j + 1, n_features):
                        X_poly.append([x[j] * x[k] for x in X])
```
Bu adım her özelliğin yüksek dereceli versiyonlarını ekler. Örneğin, $x_1^2$, $x_2^2$ gibi.
- Kare terimler: $x_1^2$, $x_2^2$
- Özellikler arası çarpımlar: $x_1 \cdot x_2$

Sonuç olarak, $X_{poly}$ matrisi şu şekli alır:

$$X_{poly} = \begin{bmatrix} 
1 & x_{11} & x_{12} & x_{11}^2 & x_{12}^2 & x_{11}\cdot x_{12} \\
1 & x_{21} & x_{22} & x_{21}^2 & x_{22}^2 & x_{21}\cdot x_{22} \\
1 & x_{31} & x_{32} & x_{31}^2 & x_{32}^2 & x_{31}\cdot x_{32}
\end{bmatrix}$$

```python
return [[X_poly[j][i] for j in range(len(X_poly))] 
        for i in range(n_samples)]
```

Bu işlem, matrisin sütun bazlı listesini satır bazlı bir listeye çevirir. Yani:

$$\begin{bmatrix} 
[1 & 1] \\
[x_1 & x_3] \\
[x_2 & x_4]
\end{bmatrix}$$

gibi bir sütun-bazlı yapı, her satırın özelliklerini içerecek şekilde düzenlenir:

$$\begin{bmatrix} 
1 & x_1 & x_2 \\
1 & x_3 & x_4
\end{bmatrix}$$

# 3. __matrix_multiply__ ve __matrix_transpose__ fonskiyonları

```python
def _matrix_multiply(self, A, B):
        # İki matrisin çarpımı
        result = [[sum(a * b for a, b in zip(row, col))
                  for col in zip(*B)]
                 for row in A]
        return result
    
    def _matrix_transpose(self, A):
        # Matrisin transpozunu al
        return list(map(list, zip(*A)))
```

```
Bir matris çarpımında,A matrisinin her bir satırını, B matrisinin her bir sütunu ile çarparız ve bu çarpımların toplamını yeni bir matrisin elemanı olarak yazarız. matrix_transpose, bir matrisin transpozunu almak için tasarlanmıştır. Transpoz işlemi, bir matrisin satırlarını sütunlarına, sütunlarını ise satırlarına dönüştürür.
```


# 4. __solve_equation__ fonksiyonu
```python
def _solve_equation(self, A, b):
        n = len(A)
        
        # Genişletilmiş matris [A|b]
        M = [row[:] + [b[i][0]] for i, row in enumerate(A)]
        
        # İleri eliminasyon
        for i in range(n):
            pivot = M[i][i]
            if abs(pivot) < 1e-10:
                # Pivot çok küçükse, diagonal elemanı artır
                M[i][i] += 1e-10
                pivot = M[i][i]
            
            for j in range(i + 1, n):
                factor = M[j][i] / pivot
                for k in range(i, n + 1):
                    M[j][k] -= factor * M[i][k]
        
        # Geri yerine koyma
        x = [0] * n
        for i in range(n - 1, -1, -1):
            x[i] = M[i][n]
            for j in range(i + 1, n):
                x[i] -= M[i][j] * x[j]
            x[i] /= M[i][i]
        
        return [[xi] for xi in x]
```
```
Bu kod, bir doğrusal denklem sistemi olan A⋅x=b problemini çözmek için Gauss Eliminasyonu yöntemini uygular. Bu yöntemde, önce ileri eliminasyon ile matris üçgen hale getirilir, ardından geri yerine koyma ile bilinmeyenler (x) çözülür.

Gaussian Eliminasyonu, bir doğrusal denklem sistemini çözmek için kullanılan bir matris çözümleme yöntemidir. Temel amacı, bir doğrusal denklem sistemini üçgen formuna (ya da bazen satır indirgenmiş biçime) getirmek ve bu şekilde bilinmeyenleri çözmektir.
```

# 5. __fit__ ve __predict__ fonksiyonları
```python
def fit(self, X, y):
        # Polinomial özellikleri oluştur
        X_poly = self._create_polynomial_features(X)
        
        # Normal denklem yöntemi
        X_t = self._matrix_transpose(X_poly)
        X_t_X = self._matrix_multiply(X_t, X_poly)
        X_t_y = self._matrix_multiply(X_t, [[yi] for yi in y])
        # Katsayıları hesapla
        self.coefficients = self._solve_equation(X_t_X, X_t_y)
        
        
        return self
    
    def predict(self, X):
        # Tahmin için X'i polinom özelliklerine dönüştür
        X_poly = self._create_polynomial_features(X)
        
        # Tahminleri hesapla
        predictions = self._matrix_multiply(X_poly, self.coefficients)
        return [pred[0] for pred in predictions]
```

```
fit, polinomal regresyon modelini eğitmek için kullanılan normal denklem yöntemini uygulamaktadır. Temel adımlar şunlardır:
- Polinomal özellikler oluşturulur (giriş verilerinden yüksek dereceli terimler ve etkileşimler elde edilir).
- Matris çarpımları ile normal denklem çözülür.
- Katsayılar hesaplanır ve modelin eğitimi tamamlanır.


predict, polinomal regresyon modelinde eğitilmiş katsayılarla, yeni giriş verisi üzerinde tahminler yapar. Temel adımlar şunlardır:

- Yeni giriş verisini polinomal özellikler haline dönüştürmek.
- Polinomal özellikleri ve eğitilmiş katsayıları çarparak tahminleri hesaplamak.
- Sonuçları döndürmek, burada her tahminin yalnızca ilk (ve tek) elemanı döndürülür.

Bu yöntemler, doğrusal regresyonun bir genellemesi olan polinomal regresyonu çözmek için güçlü bir yaklaşımdır.
```











