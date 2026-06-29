# xG Modeli: Junior Veri Bilimciler Icin Kapsamli Egitim Dokumani

> Bu dokuman, sifirdan kurulmus bir Expected Goals (xG) modelini ve oyuncu scouting
> framework'unu, bir junior veri bilimcinin her karari ve alternatiflerini anlayacagi
> sekilde anlatir. Sadece "ne yaptik" degil, "neden boyle yaptik ve baska ne yapilabilirdi"
> sorularina odaklanir. Cunku gercek ogrenme karar noktalarinda olur.

---

## Icindekiler

1. [Proje Nedir, Deger Onerisi Nedir](#1-proje-nedir-deger-onerisi-nedir)
2. [Problemin Dogru Tanimlanmasi](#2-problemin-dogru-tanimlanmasi)
3. [Veri: Kaynak, Kapsam ve Kritik Kararlar](#3-veri-kaynak-kapsam-ve-kritik-kararlar)
4. [Data Leakage: Projenin En Tehlikeli Dusmani](#4-data-leakage-projenin-en-tehlikeli-dusmani)
5. [Feature Engineering: Modelin Gercek Gucu](#5-feature-engineering-modelin-gercek-gucu)
6. [Train/Test Split: Neden Random Split Yanlis](#6-traintest-split-neden-random-split-yanlis)
7. [Modelleme Stratejisi: Basitten Karmasiga](#7-modelleme-stratejisi-basitten-karmasiga)
8. [Metrik Secimi: Neden Accuracy Degil](#8-metrik-secimi-neden-accuracy-degil)
9. [Kalibrasyon: xG'nin Kalbi](#9-kalibrasyon-xgnin-kalbi)
10. [Yorumlanabilirlik: SHAP](#10-yorumlanabilirlik-shap)
11. [Dogrulama: Modele Guvenmeyi Kanitlamak](#11-dogrulama-modele-guvenmeyi-kanitlamak)
12. [Scouting: Modelden Urune](#12-scouting-modelden-urune)
13. [Yapilmayan Seyler ve Nedenleri](#13-yapilmayan-seyler-ve-nedenleri)
14. [Gelistirme Fikirleri: Bundan Sonrasi](#14-gelistirme-fikirleri-bundan-sonrasi)
15. [Junior'a Tavsiyeler: Bu Projeden Cikarilacak Dersler](#15-juniora-tavsiyeler-bu-projeden-cikarilacak-dersler)

---

## 1. Proje Nedir, Deger Onerisi Nedir

### Tek cumle
Bir futbol sutunun gol olma olasiligini tahmin eden, iyi kalibre edilmis bir makine
ogrenmesi modeli kurduk ve bunu oyuncu/takim degerlendirme aracina donusturduk.

### xG (Expected Goals) nedir?
Her sutun kosullarina bakarak (mesafe, aci, savunmaci sayisi, kaleci konumu...) o sutun
gol olma olasiligini veren bir sayi. Ornek:

- Bos kaleye 3 metreden sut: xG = 0.95 (%95 gol olur)
- 30 metreden, 4 savunmaci arasindan sut: xG = 0.02 (%2 gol olur)

### Neden boyle bir sey gerekli? (Deger onerisi)

Futbolda gol nadir bir olaydir (~sutlarin %10'u). Nadir olaylar yuksek varyansa sahiptir:
bir oyuncu 5 macta 0 gol atabilir ama aslinda harika pozisyonlara girmistir. Skor,
kucuk orneklerde **sansin** etkisini tasiyamayacagimiz kadar gurultuludur.

xG bu gurultuyu filtreler:

1. **Takim analizi:** "3-0 kazandik" yerine "xG 0.8 - 2.1 kaybettik ama sans bizden yanaydi."
   Bir sonraki maci tahmin etmek icin xG, skordan daha iyi bir gostergedir. (Bu, finansta
   "gercek alfa vs sans" ayrimina birebir benzer.)
2. **Oyuncu scouting:** Gercek gol - xG = bitiricilik becerisi. Surekli xG'sinin ustunde
   atan oyuncu (Messi: +81) klinik bitiricidir. Bu sinyal transfer kararlarinda milyonlarca
   euro degerindedir.
3. **Benchmark kanitimiz:** Modelimiz, sektor standardi StatsBomb'un kendi xG'sini
   log-loss'ta %2.3 geciyor. Yani "oyuncak proje" degil, uretim kalitesinde bir model.

### Quant perspektifinden neden guzel bir problem?

xG problemi, kantitatif finanstaki bircok problemin kucuk bir kopyasi:

- **Olasilik tahmini > siniflandirma.** Bir trade'in "kazanir/kaybeder" etiketi degil,
  kazanma olasiligi onemlidir. Ayni sekilde xG'de "gol/degil" degil, P(gol) onemli.
- **Kalibrasyon kritik.** Risk modeliniz %1 temerrut diyorsa gercekten %1 olmali.
  Miscalibrated model, dogru siralasa bile (yuksek AUC) yanlis fiyatlama yapar.
- **Sinyal/gurultu ayrimi.** Skill vs luck ayrimi = alpha vs noise ayrimi.
- **Leakage hayati.** Gelecek bilgisini gecmise sizdiran her model backtest'te harika,
  gercekte felaket performans gosterir.

---

## 2. Problemin Dogru Tanimlanmasi

### Karar: Binary classification + olasilik ciktisi

```
Girdi:  Sutun cekildigi anin tum bilgileri (konum, oyuncular, baglamsal bayraklar)
Cikti:  P(is_goal = 1) -- 0 ile 1 arasi olasilik
```

**Neden boyle?** Cunku sorulan soru "bu sut gol mu olur?" degil, "bu sutun gol olma
olasiligi kac?". Bu ayrim her seyi degistirir:

| | Siniflandirma odakli | Olasilik odakli (bizim secim) |
|---|---|---|
| Hedef | Dogru etiket | Dogru olasilik |
| Ana metrik | Accuracy, F1 | **Log-loss, Brier** |
| Threshold | Gerekli (0.5?) | Gereksiz |
| Class imbalance | Sorun, oversample edilir | Sorun degil, dogal oran korunur |
| Kalibrasyon | Umursanmaz | **Zorunlu** |

### Alternatif yaklasimlar ve neden secilmedi

1. **Regresyon (gol sayisi tahmini):** Sut bazinda anlamsiz; bir sut 0 veya 1 gol olur.
   Mac bazinda Poisson regresyon yapilabilirdi ama o farkli bir problem (mac tahmini).
2. **Ranking (siralama):** Sutlari "tehlike sirasina" dizmek. AUC'yi optimize ederdik
   ama olasiliklar anlamsiz kalirdi. Scouting icin gol - xG farki hesaplanamazdi.
3. **Survival analizi / hazard model:** Sut bazli degil pozisyon bazli "gol gelene kadar
   gecen sure" modeli. Ilginc ama veri yapimiz sut bazli; asiri muhendislik olurdu.

> **Ders:** Problemi formulle etmeden once ciktinin nasil KULLANILACAGINI sor.
> Bizim cikti, toplanabilir olmali (oyuncu basina xG toplami) -- bu, kalibre olasilik
> gerektirir. Kullanim sekli, metrigi; metrik, modeli belirler. Tersi degil.

---

## 3. Veri: Kaynak, Kapsam ve Kritik Kararlar

### Kaynak: StatsBomb Open Data

- Ucretsiz, event-level futbol verisi (her pas, sut, mudahale ayri kayit).
- Sut eventi icin: konum (x, y), sonuc, vucut bolumu, teknik, oyun kalibi,
  baski altinda bayragi ve **freeze_frame** (sut anindaki tum oyuncularin konumu).
- StatsBomb kendi xG degerini de veriyor -> bizim icin **benchmark**.

### Cekilen veri: 65,822 sut, 2,651 mac, tum erkek profesyonel ligler

**Muhendislik notu:** Ilk implementasyon `statsbombpy.sb.events()` ile mac mac cekiyordu
ve 1+ saat surdu (her mac icin tum eventleri indirip parse ediyor). Cozum: GitHub raw
JSON endpoint'lerine dogrudan istek + `ThreadPoolExecutor(max_workers=16)` ile paralel
indirme. Sure dakikalara dustu.

> **Ders:** Kutuphane convenience fonksiyonlari prototip icindir. Veri hacmi buyuyunce
> alt katmana in (raw API, paralel I/O). I/O-bound isler thread'lerle paralellesir
> (GIL engel degil), CPU-bound isler process gerektirir.

### Kapsam kararlari (her biri bilincli ve gerekceli)

| Karar | Gerekce |
|---|---|
| **Penaltilar HARIC** | Penaltinin xG'si sabittir (~0.76) ve sut kosullarindan bagimsizdir. Modele dahil edilirse hem ogrenecek bir sey yoktur hem de dagitimi bozar. Sektor standardi da haric tutmaktir. |
| **Kendi kalesine goller HARIC** | Tanim geregi "sut" degildir; hedef degiskeni kirletir. |
| **Serbest vuruslar DAHIL** (shot_type kategorisi olarak) | Penaltinin aksine serbest vurusun kosullari degisken (mesafe, baraj). Model ogrenebilir. Ayri kategori olarak isaretlendi. |
| **Tum erkek ligleri** | Daha cok veri > lig homojenligi. Lig farklari ayrica test edildi (asagida). |

**Lig farki analizi:** Liglerin finishing_vs_xg farklari olculdu: -0.005 ile +0.014
arasi, yani ihmal edilebilir. Bu yuzden `competition` modele feature olarak **eklenmedi**.
Eklenebilirdi; ama olculen etki sifira yakinken feature eklemek varyansi artirir,
genelleme yetenegini dusurur.

> **Ders:** "Su feature'i da ekleyelim mi?" sorusunun cevabi sezgi degil olcumdur.
> Once etkiyi izole olarak olc, sonra karar ver.

---

## 4. Data Leakage: Projenin En Tehlikeli Dusmani

Leakage = tahmin aninda elimizde OLMAYACAK bilginin egitime sizmasi. Uc katmanda
savunma kurduk:

### 4.1 Hedef sizintisi: `statsbomb_xg` asla feature degil

StatsBomb'un kendi xG'si veride mevcut. Bunu feature yapmak modeli "StatsBomb'u kopyala"
makinesine cevirir: metrikler harika gorunur, ogrenilen sey sifir olur, ve benchmark
anlamsizlasir (kendinle yarisamazsin).

Kod seviyesinde zorlama: `src/features/pipeline.py` icinde `FORBIDDEN_COLUMNS` listesi
var; `statsbomb_xg`, `shot_outcome`, `is_goal`, `player`, `team` vb. fiziksel olarak
feature setine giremez.

> **Ders:** Leakage kurallarini dokumana degil KODA yaz. Dokuman unutulur,
> assertion unutulmaz.

### 4.2 Preprocessing sizintisi: fit yalnizca train'de

`OneHotEncoder`, `SimpleImputer`, `StandardScaler` -- hepsi sklearn `Pipeline` icinde
ve yalnizca train verisiyle fit ediliyor. Test verisi sadece transform gorur.

Neden onemli? Ornek: imputer'i tum veriyle fit edersen, test setinin medyani train'e
sizar. Etki kucuk gorunur ama prensip ihlali birikir; ve gercek dunyada (uretimde)
"tum veri" diye bir sey yoktur, sadece gecmis vardir.

### 4.3 Grup sizintisi: mac bazli split (Bolum 6'da detayli)

---

## 5. Feature Engineering: Modelin Gercek Gucu

Bu projenin asil farki burasi. Ham veri (x, y koordinati, birkac bayrak) dogrudan
modele verilebilirdi; biz bunun yerine **alan bilgisini geometriye cevirdik**.

### 5.1 Geometrik feature'lar (`src/features/geometry.py`)

| Feature | Hesap | Neden |
|---|---|---|
| `geom_distance` | Sut noktasindan kale merkezine (120, 40) Oklid mesafesi | En temel sinyal: uzak = zor |
| `geom_angle_rad` | Iki kale diregine giden vektorler arasindaki aci (dot/cross product) | Mesafe ayni olsa da kose pozisyonundan kale "dar" gorunur. Bu, mesafeden bagimsiz ikinci eksen |
| `geom_in_box` | Ceza sahasi icinde mi | Kesikli rejim degisikligi (savunma davranisi farkli) |

**Aci neden ham koordinattan ustun?** LightGBM (x, y)'den aciyi teorik olarak ogrenebilir
ama bunun icin cok sayida split harcamasi gerekir. Aciyi dogrudan vermek, modelin
"complexity budget"ini daha ince oruntulere ayirmasini saglar. Logistic Regression ise
aciyi (x, y)'den ASLA ogrenemez (dogrusal model, trigonometri uretemiyor) -- feature
muhendisligi LR icin hayat memat meselesi.

### 5.2 Freeze-frame feature'lari (`src/features/freeze_frame.py`)

Sut anindaki oyuncu konumlarindan:

- `ff_n_opponents_in_cone`: Sut noktasi ile iki direk arasindaki ucgende kac rakip var
  (point-in-triangle testi ile). Bloke olasiliginin dogrudan olcusu.
- `ff_dist_to_gk`, `ff_gk_off_line`, `ff_gk_y_offset`: Kaleci nerede? Cizgiden cikmis
  kaleci = bos kale riski. (SHAP'ta 1 numarali feature cikti.)
- `ff_dist_nearest_opponent`: Sutcu uzerindeki baski.
- `ff_n_teammates_in_box`: Hucum kalabaligi.

### 5.3 Baglamsal feature'lar (Faz 2.5'te eklendi)

- **Native StatsBomb bayraklari:** `shot_open_goal` (gol orani %74.7 -- 7.4x lift!),
  `shot_one_on_one`, `shot_deflected`, `shot_first_time` vb.
- **Key pass (asist) ozellikleri:** Sutten onceki pasin tipi. `kp_is_through_ball`
  (ara pasi) sonrasi gol orani %30.4 -- baseline'in 3 kati. Cunku ara pasi savunmayi
  zaten devre disi birakmistir; bu bilgi freeze-frame'de kismen gorunur ama pas tipi
  ek sinyal tasir.
- **Oyun durumu:** `score_diff` (sut anindaki skor farki). Bulgu: 3 fark geride olan
  takimlar ortalama 20.3 metreden, 3 fark onde olanlar 18.5 metreden sut atiyor;
  gol oranlari %7.3 vs %16.4. Geriden gelen umutsuzca uzaktan deniyor. **Kritik nokta:**
  StatsBomb'un xG'si oyun durumunu kullanmiyor (korelasyon olcumuyle dogrulandi) --
  modelimizin StatsBomb'u gectigi alanlardan biri bu.

### 5.4 Eklenmeyen feature'lar (bilincli)

| Reddedilen | Neden |
|---|---|
| `gk_angle_offset`, `defense_density_ratio` | Mevcut kolonlarin oranlari/kombinasyonlari -- agac modeli bunlari kendisi ogrenir, ek bilgi yok |
| `is_preferred_foot` | Acik veride yok; tahmin etmeye kalkmak leakage riski |
| `competition` | Olculen etki ~0 (yukarida) |

> **Ders (cok onemli):** Agac modellerine feature eklerken soru "bu sinyal mi?" degil,
> "bu YENI bilgi mi?" olmali. Mevcut kolonlarin dogrusal olmayan kombinasyonlarini
> agaclar zaten ogrenir. Yeni bilgi = veride baska hicbir kolondan turetilémeyecek sey
> (pas tipi, skor durumu gibi).

---

## 6. Train/Test Split: Neden Random Split Yanlis

### Problem

Ayni mactaki sutlar bagimsiz degildir: ayni iki takim, ayni savunma hatti, ayni kaleci,
ayni hava, ayni temposuyla oynanan mac. Sutlari rastgele bolersen ayni macin yarisi
train'e yarisi test'e gider. Model, macin "ortamini" train'den ezberler, test'te ayni
ortami gorur, metrikler sisirilir. Buna **grup sizintisi** denir.

(Finans paraleli: zaman serisinde rastgele split yapmak. Ayni gunun ticklerini hem
train hem test'e koyarsan backtest'in yalan soyler.)

### Cozum: mac bazli, stratified split (`src/preprocess/splitter.py`)

1. Mac basina gol orani hesapla.
2. Maclari gol orani ceyrekliklerine (quartile) ayir.
3. Her ceyreklikten %20 maci test'e ayir (GroupShuffleSplit mantigi).
4. Dogrula: train/test mac kesisimi = 0, gol oranlari %10.1 / %10.2 (dengeli).

Stratifikasyon neden? Maclar gol orani acisindan cok degisken (0 gollu maclar da var,
6 gollu de). Sadece rastgele mac secersen test setinin gol orani sapabilir; ceyreklik
bazli secim bunu dengeler.

### Alternatifler

- **Sezon bazli split** (eski sezonlar train, yeni sezonlar test): Zaman yonlu
  genellemeyi de test eder, daha da saglamdir. Kullanmadik cunku verimiz cok heterojen
  (farkli liglerin farkli sezonlari); sezon bazli boldugumuzde bazi liglerin tamami tek
  tarafa duserdi. Mac bazli split, grup sizintisini cozer ve dagilimi korur. Daha buyuk
  homojen veriyle (tek ligin 10 sezonu) sezon bazli tercih edilirdi.
- **Oyuncu bazli split:** Ilginc soru ("yeni oyunculara genelliyor mu?") ama bizim
  kullanim senaryomuz bu degil.

---

## 7. Modelleme Stratejisi: Basitten Karmasiga

### Asama 1: Logistic Regression baseline (Faz 3)

**Neden once LR?**

1. **Yorumlanabilir referans:** Katsayilarin isaretleri alan bilgisiyle karsilastirilabilir.
   Sanity check'ler: mesafe katsayisi negatif mi? (Evet: -1.94.) Aci pozitif mi? (Evet.)
   `shot_deflected` pozitif mi? (Evet: +2.25.) Isaretler ters cikarsa veri pipeline'inda
   bug var demektir -- model karmasiklasmadan once yakala.
2. **Metrik tabani:** Karmasik modelin "ne kadar deger kattigini" olcecek cubuk.
3. **Kalibrasyon dogal:** LR, log-loss'u dogrudan optimize ettigi icin ciktilari
   genelde iyi kalibredir.

**Onemli detay -- `class_weight` KULLANILMADI.** Imbalanced veri gorunce refleks olarak
`class_weight="balanced"` yazan junior cok gordum. Bu, P(gol) tahminlerini sistematik
olarak yukari iter (azinlik sinifin agirligi artar) ve **kalibrasyonu yok eder**.
Accuracy umursamiyorsak class weighting'e ihtiyac yok; dogal oran korunmali.

Sonuc: LR, StatsBomb xG'yi UC metrikte de gecti (log-loss 0.2489 vs 0.2546).
Feature muhendisligi o kadar iyiydi ki dogrusal model bile sektor standardini yendi.

### Asama 2: LightGBM (Faz 4)

**Neden gradient boosting?** Tabular veri + orta boyut (65k satir) + karisik tip
feature'larda gradient boosting hala state-of-the-art. Neural network bu olcekte ve bu
yapida (heterojen tabular feature'lar) tipik olarak GBM'i gecemez ve cok daha fazla
tuning ister.

**Neden XGBoost degil LightGBM?**
- Native kategorik destek (OHE'siz; OHE agac modellerde split verimsizligi yaratir)
- Histogram-based: daha hizli egitim
- NaN'lari native isler (imputation'a gerek yok -- imputer sadece LR pipeline'inda)

**Hyperparameter tuning:** 5 konfigurasyonluk kucuk grid, **GroupKFold (group=match_id)**
ile 5-fold CV. CV'de de mac bazli bolme sart -- yoksa CV skoru sisirilir ve yanlis
konfigurasyon secersin. Kazanan: `n_estimators=1000, lr=0.02, depth=5, leaves=31,
subsample=0.7, colsample=0.7` -- yani **yavas ogrenen, agresif regularize edilmis**
bir model. Gurultulu hedefte (gol ~%10 ve sansa bagli) bu beklenen sonuctur.

**Neden devasa grid degil 5 konfig?** Iyi feature'larla model secimi ikincildir.
Grid'i 10x buyutmek log-loss'ta belki 4. ondaliga dokunur; ayni emek feature'a
harcansa 2. ondaliga dokunabilir. Emegin marjinal getirisi feature tarafindadir.

### Sonuc tablosu (test seti, n=13,143)

| Metrik | LightGBM | Baseline LR | StatsBomb xG |
|---|---|---|---|
| Log-loss | **0.2487** | 0.2489 | 0.2546 |
| Brier | **0.0711** | 0.0712 | 0.0724 |
| ROC-AUC | 0.8346 | **0.8349** | 0.8245 |

### LR ile LightGBM'in neredeyse esit cikmasi ne anlama geliyor?

Bu bir basarisizlik DEGIL; en onemli bulgulardan biri:

1. Feature'lar sinyali **dogrusal olarak ayristirilabilir** hale getirmis. Aciyi,
   koni icindeki rakibi, kaleci mesafesini biz hesapladik; modele ogrenecek dogrusal
   olmayan az sey kaldi.
2. Problemin **aleatorik (indirgenemez) belirsizligi** yuksek: ayni kosullarda ayni
   sut bazen girer bazen girmez. Hicbir model bunu asamaz; tavana yakinsadik.
3. Pratik cikti: uretimde LR bile kullanilabilir (daha ucuz, tam yorumlanabilir).
   LightGBM'i sectik cunku marjinal olarak daha iyi kalibre (asagida).

---

## 8. Metrik Secimi: Neden Accuracy Degil

### Accuracy tuzagi

Gol orani %10. "Hicbir sut gol olmaz" diyen sabit model %90 accuracy alir ve tamamen
ise yaramazdir. Imbalanced veride accuracy, bilgi tasimaz.

### Kullandigimiz metrik hiyerarsisi (oncelik sirasiyla)

1. **Log-loss (birincil):** `-(y*log(p) + (1-y)*log(1-p))` ortalamasi. Olasilik
   tahmininin "proper scoring rule"u: minimize etmek, gercek olasiligi soylemeyi
   optimal kilar. Yanlis ve EMIN tahminleri cok sert cezalandirir (p->0 iken gol
   olursa ceza sonsuza gider). Quant dunyasinda da olasilik modellerinin standardi.
2. **Brier skoru (ikincil):** `(p - y)^2` ortalamasi. Yine proper scoring rule ama
   uc tahminlere log-loss kadar sert degil. Iki proper metrik ayni yonu gosteriyorsa
   sonuc sagламdir.
3. **Kalibrasyon egrisi + ECE (gorsel/yapisal):** Bolum 9.
4. **ROC-AUC (yalnizca tani amacli):** Siralama gucunu olcer, olasiligin dogrulugunu
   OLCMEZ. Tum tahminleri 2'ye bolsen AUC degismez ama xG toplamlari anlamsizlasir.
   Bu yuzden asla birincil metrik degil.

> **Ders:** "Proper scoring rule" kavramini ogren. Bir metrik, dogru olasiligi
> raporlamayi optimal strateji yapiyorsa proper'dir (log-loss, Brier). Accuracy ve
> AUC proper degildir. Olasilik onemliyse metrik proper olmali.

---

## 9. Kalibrasyon: xG'nin Kalbi

### Tanim

Model "%20" dediginde gercekte ~%20 olmasi. Olcum: tahminleri binle, her binin ortalama
tahminini gercek gol oraniyla karsilastir (reliability diagram). Sayisallastirma:
ECE (Expected Calibration Error) = bin farklarinin ortalamasi.

### Neden xG icin hayati?

Scouting cikti: `oyuncunun_gol_sayisi - SUM(xG)`. Bu toplam ancak her bir xG dogru
olasiliksa anlamlidir. Miscalibrated model -> yanlis toplam -> yanlis oyuncu degerlemesi.
AUC'si yuksek ama kalibrasyonu bozuk model, scouting icin **ise yaramaz**.

### Surec

1. LightGBM ciktilarinda ECE olcustuk: **0.0176** (esik 0.02'nin altinda).
2. Plan haziri vardi: ECE > 0.02 olsaydi isotonic regression uygulanacakti
   (`CalibratedClassifierCV`). Gerek kalmadi.

**Neden boosting'de kalibrasyon sorunu beklenir?** Boosting, log-loss'u optimize etse de
asiri agac sayisi/derinlikle uc degerlere itebilir; ayrica early stopping ve
regularizasyon dagilimi sikistirabilir. Bizim agresif regularize konfigurasyonumuz
(depth=5, subsample=0.7) buna engel oldu.

**Isotonic vs Platt:** Platt (sigmoid fit) parametriktir, az veriyle calisir ama tek
S-egrisi varsayar. Isotonic non-parametriktir, monotonik herhangi bir bozulmayi duzeltir
ama cok veri ister. 50k+ ornekle isotonic dogru secimdir; 5k altinda Platt'a don.

> **Ders:** Kalibrasyon kontrolu, olasilik ciktili her modelde ZORUNLU son adimdir.
> "Model bitti" demeden once reliability diagram ciz. Cizmiyorsan model bitmemistir.

---

## 10. Yorumlanabilirlik: SHAP

### Ne yaptik

TreeExplainer ile her tahmini feature katkilarina ayristirdik. Ilk 5 (ortalama |SHAP|):

1. `ff_dist_to_gk` (0.416) -- kaleciye mesafe
2. `geom_angle_rad` (0.400) -- sut acisi
3. `ff_n_opponents_in_cone` (0.246) -- konideki savunmacilar
4. `ff_dist_nearest_opponent` (0.190) -- baski
5. `shot_body_part` (0.139) -- vucut bolumu

### Bu listenin asil islevi: dogrulama

Plan dokumana onceden su yazildi: "Distance ve angle baskin olmali. Degilse bug."
Bu, **onceden kayitli bir hipotez** (quant'larin pre-registration disiplini). SHAP
sonucu hipotezi dogruladi: geometri ve baski baskin -- modelin ogrendigi sey futbol
fiziği, veri artefakti degil.

Eger 1 numara `match_id` benzeri bir sey cikssaydi: leakage. Eger `kp_pass_angle`
ciksaydi: muhtemelen veri hatasi. SHAP burada hata dedektoru olarak calisiyor.

**Ilginc bulgu:** `ff_dist_to_gk` mesafenin onune gecti. Mantikli: `geom_distance`
ile kaleci mesafesi koreledir ama kaleci mesafesi ek olarak "kaleci avlanmaya cikmis mi"
bilgisini tasir. SHAP, korelasyonlu feature'larda katkiyi paylasitirir -- bu yuzden
`geom_distance`'in dusuk gorunmesi "mesafe onemsiz" demek DEGILDIR.

> **Ders:** SHAP degerleri nedensellik degildir ve korelasyonlu feature'larda katki
> bolusumu yanilticidir. SHAP'i "model neye bakiyor" tanisi olarak kullan,
> "futbolda neyin onemli oldugu" iddiasi olarak degil.

---

## 11. Dogrulama: Modele Guvenmeyi Kanitlamak

Test metrikleri yeterli degil; modeli kullanim senaryosunda dogruladik (Faz 5):

### 11.1 Sut seviyesi
- Model vs StatsBomb korelasyonu: **r = 0.881.** Iki bagimsiz model ayni sinyali
  goruyor -- karsiliklı dogrulama. Kalan %12 fark, bizim ek feature'larimizdan
  (score_diff, key pass) geliyor.
- Model vs gercek sonuc: r = 0.474. Dusuk gorunur ama DOGRUDUR: tek sut binary ve
  sansa bagli. (StatsBomb da 0.462.) Bu sayinin dusuklugu problem degil; problemin
  dogasi.

### 11.2 Agregasyon seviyesi (asil kullanim)
- **Takim bazinda** (130 takim): toplam xG vs toplam gol korelasyonu **r = 0.9956.**
- **Oyuncu bazinda** (303 oyuncu, min 10 sut): **r = 0.9809.**

Neden agregasyonda korelasyon ucuyor? Merkezi limit teoremi: tekil sutlarin gurultusu
toplamda birbirini goturur, sinyal kalir. Kullanim senaryomuz (sezonluk degerlendirme)
tam da agregasyon seviyesinde -- yani model tam ihtiyac noktasinda guclu.

### 11.3 Hata analizi
- Yuksek xG (>=0.5) sutlarin %67.5'i gol oldu -- istatistiksel olarak tutarli.
- Dusuk xG (<=0.1) gollerin profili: uzak (19m), dar aci, kaleciden uzak. Yani model
  bunlara dusuk olasilik vermekte HAKLI; gol olmalari beklenen kuyruk olayi
  (9,404 dusuk-xG suttan 338 gol = %3.6, zaten <%10 tahmin ediliyordu).

> **Ders:** "Modelin yanildigi" ornekleri tek tek inceleme aliskanligi edin. Cogu zaman
> model yanilmamistir; dusuk olasilikli olay gerceklesmistir. Ikisini ayirt edebilmek,
> olasilik dusuncesinin ozudur.

---

## 12. Scouting: Modelden Urune

Model bir SAYI uretir; urun bir KARAR destekler. Faz 6'da koprusu kuruldu:

### Oyuncu analizi
- Oyuncu basina: `goals_above_xg = gercek_gol - SUM(xG)`.
- **Minimum 20 sut filtresi:** 5 sutla +2 fark anlamsizdir (sans). Esik, kucuk ornek
  gurultusunu keser.
- **Guven araligi:** `SE = sqrt(n * p * (1-p))` (binom varyans). Messi'nin +80.6 farki
  SE=17.6 ile **z ~ 4.6** -- istatistiksel olarak ezici. 23 sutlu bir oyuncunun +3.5'i
  ise SE'sine gore zayif sinyal.

Bulgular kendini dogruladi: listenin tepesinde Messi (+80.6), Suarez (+13.3),
Mbappe (+9.3) -- futbol dunyasinin bilinen klinik bitiricileri. Altinda xG'sini
yakamayan profiller. **Bilinen gercekleri yeniden kesfetmek, modelin bilinmeyenler
hakkindaki iddialarina guven kazandirir.**

### Takim analizi
- Hucum farki: `goals_for - xG_for` (klinik hucum)
- Savunma farki: `xG_against - goals_against` (kurtaran kalecilik/savunma)
- Kadran grafigi: iki ekseni birlestirip takim profilini tek bakista verir.

> **Ders:** Ciktiyi her zaman belirsizligiyle birlikte raporla. "+3.5 gol" degil,
> "+3.5 +/- 2.7 gol". Belirsizliksiz sayi, karar vericiyi yanlis yonlendirir.

---

## 13. Yapilmayan Seyler ve Nedenleri

Junior icin en ogretici bolum: neyi YAPMAMAYI sectik?

| Yapilmayan | Neden |
|---|---|
| **Neural network** | 65k tabular satirda GBM'den iyi performans beklenmez; tuning maliyeti yuksek, yorumlanabilirlik dusuk. Yanlis arac. |
| **Oversampling/SMOTE** | Olasilik problemi! Sinif oranini degistirmek kalibrasyonu temelli bozar. SMOTE, siniflandirma odakli problemler icindir ve orada bile tartismali. |
| **Accuracy raporlama** | %10 pozitif oranla yaniltici (Bolum 8). |
| **Devasa hyperparameter grid** | Marjinal getiri feature muhendisliginin cok altinda. |
| **Stacking/ensemble** | LR ile LightGBM zaten ayni noktaya yakinsiyor; ensemble'in katacagi varyans azaltimi minimal, karmasiklik maliyeti gercek. |
| **AutoML** | Ogrenme projesinde sureci disaridan almak amaca aykiri; ayrica AutoML kalibrasyona ve grup sizintisina bizim kadar dikkat etmez. |
| **Post-hoc kalibrasyon** | Olculdu, gerek yoktu (ECE 0.0176 < 0.02). "Her zaman isotonic uygula" reflekssi yanlis: gereksiz kalibrasyon, ek varyans getirir. |
| **`competition` feature'i** | Olculen etki ~0. |

> **Ders:** Bir seyin yapilmamasi karar gerektirir ve gerekce dokumante edilmelidir.
> "Neden X yok?" sorusuna "dusunmedik" cevabi ile "olctuk, getirisi yoktu" cevabi
> arasinda profesyonellik ucurumu vardir.

---

## 14. Gelistirme Fikirleri: Bundan Sonrasi

Siralama: beklenen getiri / efor oranina gore.

1. **Sut sonrasi model (xGOT / post-shot xG):** Sutun kaleye gidis yonunu de kullanarak
   "kalecinin kurtarma sansini" modelle. Bitiricilik sinyalini sut yerlestirme ve sut
   secimi olarak ikiye ayirir -- scouting degerini ciddi artirir.
2. **Zamansal dogrulama:** Eski sezonlarda egit, en yeni sezonda test et. Modelin
   zamana dayanikliligini kanitlar (futbol taktigi evrilir; concept drift olcumu).
3. **Hiyerarsik / karma model (mixed effects):** Oyuncu bitiriciligini modelin ICINE
   random effect olarak koy. Su anki "gol - xG" yaklasimi two-stage; tek asamali
   hiyerarsik model, kucuk ornekli oyuncular icin dogal shrinkage saglar
   (quant'larin James-Stein / partial pooling sezgisi).
4. **Belirsizlik araliklari:** Conformal prediction veya bootstrap ile her xG'ye
   guven araligi. "0.35" yerine "0.35 [0.28, 0.43]".
5. **Kadin futbolu verisi:** StatsBomb'ta mevcut; ayri model mi tek model mi sorusu
   basli basina guzel bir transfer learning calismasi.
6. **Canli (in-game) xG:** Mac ici kumulatif xG akisi -- yayin/bahis urunlerinin temeli.

---

## 15. Junior'a Tavsiyeler: Bu Projeden Cikarilacak Dersler

1. **Once problemi, sonra modeli sec.** "Olasilik mi etiket mi" sorusu, tum metrik ve
   model zincirini belirledi. Cogu proje bu soruyu hic sormadan accuracy optimize eder.

2. **Feature muhendisligi > model secimi.** LR'nin StatsBomb'u gectigi an, bu projenin
   en onemli aniydi. Sektor standardini dogrusal modelle yenmek, sinyalin feature'larda
   oldugunu kanitlar. Once alan bilgisini matematige cevir, sonra model dene.

3. **Leakage'i koda gom.** FORBIDDEN_COLUMNS listesi, pipeline-ici fit, mac bazli split,
   GroupKFold -- dort ayri katman. Leakage tek katmanla onlenmez.

4. **Baseline'siz "iyi model" iddiasi bostur.** LightGBM'in 0.2487'si tek basina anlamsiz;
   LR'in 0.2489'u ve StatsBomb'un 0.2546'si yaninda anlam kazaniyor.

5. **Kalibrasyonu olc, varsayma.** Reliability diagram cizilmeden olasilik modeli bitmez.

6. **Sanity check'leri ONCEDEN yaz.** "Distance katsayisi negatif olmali", "SHAP'ta
   geometri baskin olmali" -- bunlar kosmadan once yazildi. Sonuca bakip hikaye uydurmak
   (HARKing) ile hipotezi onceden kaydetmek arasindaki fark, bilim ile numerolojinin farkidir.

7. **Buyuk sayilarin gucunu kullan.** Sut seviyesinde r=0.47, sezon seviyesinde r=0.99.
   Gurultulu tahminler toplamda guclu sinyale donusur -- urununu dogru agregasyon
   seviyesinde konumla.

8. **Belirsizligi raporla.** Esik filtreleri (min 20 sut) + standart hatalar olmadan
   siralama tablolari yalan soyler.

9. **Sureci dokumante et.** Bu projede her fazin plan dokumani, kabul kriterleri ve
   sonuc kayitlari var (docs/ klasoru). Alti ay sonra "neden boyleydi" sorusunun cevabi
   kafanda degil, repoda olmali.

10. **"Yapmadiklarini" da savun.** SMOTE kullanmamak, NN denememek, grid'i buyutmemek --
    bunlarin hepsi gerekceli karar. Mulakatlarda "neden X kullanmadin?" sorusuna verilen
    cevap, "X kullandim" demekten cok daha ayirt edicidir.

---

## Ek: Proje Yapisi ve Calistirma

```
XG_PROJE/
|-- docs/        -> faz planlari + bu dokuman (sureç kaydi)
|-- src/
|   |-- features/    -> geometry, freeze_frame, transformers, pipeline
|   |-- preprocess/  -> splitter (mac bazli split)
|   |-- models/      -> baseline (LR), gradient_boosting (LightGBM)
|   |-- evaluation/  -> metrics (log-loss, Brier, reliability diagram)
|   |-- scouting/    -> player_analysis, team_analysis
|   |-- dashboard/   -> theme, logos (Streamlit arayuzu)
|-- scripts/     -> extract_shots, build_features, train, train_gbm,
|                   validate, scouting, fetch_logos
|-- app.py       -> interaktif dashboard (streamlit run app.py)
```

Calistirma sirasi: `extract_shots.py` -> `build_features.py` -> `train.py` ->
`train_gbm.py` -> `validate.py` -> `scouting.py` -> `streamlit run app.py`

### Anahtar sonuclar (ozet karti)

| Gosterge | Deger |
|---|---|
| Veri | 65,822 sut / 2,651 mac / %10.13 gol orani |
| Final model | LightGBM (depth=5, lr=0.02, n=1000, subsample=0.7) |
| Log-loss (test) | 0.2487 (StatsBomb: 0.2546 -> %2.3 iyilesme) |
| ECE | 0.0176 (post-hoc kalibrasyon gereksiz) |
| Takim dogrulamasi | r = 0.9956 (xG toplami vs gercek gol) |
| En buyuk bitiricilik sinyali | Messi +80.6 gol (z ~ 4.6) |

---

*Hazirlanma tarihi: 2026-06-12. Sorularin olursa once docs/ altindaki faz planlarina bak;
her kararin gerekcesi ve sonucu orada kayitli.*
