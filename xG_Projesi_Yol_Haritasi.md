# xG (Expected Goals) Modeli — Proje Yol Haritası

## Proje Özeti
Sıfırdan bir Expected Goals modeli kurup, üstüne bir oyuncu değerlendirme (scouting) katmanı inşa etmek. Amaç sadece "iyi accuracy" değil; **iyi kalibre edilmiş**, gerekçelendirilmiş ve gerçek bir benchmark'a (StatsBomb xG) karşı doğrulanmış bir model.

## Teknoloji & Veri
- **Veri:** StatsBomb Open Data (`statsbombpy` kütüphanesi veya GitHub'dan ham JSON). Event-level, freeze-frame içeren en zengin ücretsiz kaynak.
- **Stack:** Python, pandas, scikit-learn, XGBoost/LightGBM, matplotlib/seaborn, SHAP.
- **(Opsiyonel sunum katmanı):** FastAPI + React (zaten bildiğin stack).

---

## Faz 0 — Kurulum & Veri Tanıma
**Amaç:** Tekrar üretilebilir bir iskelet ve veriye hakimiyet.

- Repo yapısı: `data/`, `notebooks/`, `src/`, `models/`, `reports/`.
- `venv` + `requirements.txt`, sabit `random_state`.
- `statsbombpy` ile competitions → matches → events akışını öğren.
- Bir maçın event verisini incele: bir şut event'inde neler var? (`location`, `shot.outcome`, `shot.body_part`, `shot.type`, `under_pressure`, `shot.freeze_frame`, ve StatsBomb'un kendi `shot.statsbomb_xg` değeri).

**Çıktı:** Çalışan ortam + "veri sözlüğü" notu (hangi alan ne işe yarıyor).

---

## Faz 1 — Veri Çıkarımı & EDA
**Amaç:** Modellenecek temiz bir şut tablosu.

- Tüm maçlardan şut event'lerini çek, tek bir DataFrame'e indir (`is_goal` hedef değişkeni).
- **Kritik kapsam kararı:** Penaltıları ayır (xG'leri sabit ~0.76, modele dahil etme). Açık oyun şutlarıyla başla; serbest vuruşları ayrı kategori yap.
- EDA: şut konum dağılımı (heatmap), bölgeye göre gol dönüşüm oranı, taban gol oranı (~%10 — sınıf dengesizliğini buradan gör).

**DS karar noktası:** "Hangi şutları dahil ediyorum, neden?" — bunu yaz.
**Çıktı:** `shots.parquet` + EDA notebook'u.

---

## Faz 2 — Feature Engineering
**Amaç:** Modelin "iyi gol şansı" sezgisini öğreneceği değişkenler.

- **Mesafe:** şut konumundan kale merkezine uzaklık.
- **Açı:** iki direk arasında kalan açı (asıl güçlü geometrik özellik — formülü doğru kur).
- **Vücut bölgesi** (ayak/kafa/diğer), **şut tipi**, **play pattern**, **under_pressure**.
- **İleri seviye (ayrıştırıcı): freeze_frame'den** → top ile kale arasındaki savunmacı sayısı, en yakın savunmacıya mesafe, kalecinin konumu. Bu kısım projeni sıradan bir xG'den ayırır.
- Kategorikleri one-hot encode et.

**Tuzak:** StatsBomb'un `statsbomb_xg` değerini **asla** feature olarak kullanma (data leakage). O sadece karşılaştırma için.
**Çıktı:** Feature pipeline (`src/features.py`).

---

## Faz 3 — Baseline Model
**Amaç:** Yorumlanabilir, dürüst bir referans nokta.

- **Logistic Regression** (xG'nin klasik baseline'ı, katsayıları yorumlanabilir).
- **Split kararı:** rastgele değil, maç/sezon bazında ayır; hedefe göre stratify et (nadir olay).
- Sınıf dengesizliğini ele al ve seçimini gerekçelendir (`class_weight` vs. olduğu gibi bırakıp olasılığa güvenmek — xG'de genelde ikincisi tercih edilir).
- **Metrikler:** log-loss + Brier score (asıl olanlar), ROC-AUC (ikincil).

**Çıktı:** Baseline skorları tablosu.

---

## Faz 4 — Gelişmiş Model & Kalibrasyon
**Amaç:** Performansı artırmak ama doğru sebeple.

- **Gradient Boosting** (XGBoost/LightGBM), hiperparametre tuning (cross-validation ile).
- Baseline'a karşı log-loss/Brier üzerinden karşılaştır.
- **Kalibrasyon** işin kalbi: reliability diagram (model %30 dediğinde gerçekten ~%30 mu?). Gerekirse Platt scaling / isotonic regression.
- **SHAP** ile feature etkilerini yorumla (açı ve mesafe baskın çıkmalı — sağlamasını yaparsın).

**DS karar noktası:** "Hangi yaklaşımları deneyebilirdik, hangisini neden seçtik?" — mentorunun istediği çerçeve.
**Çıktı:** Final model + kalibrasyon grafikleri.

---

## Faz 5 — Doğrulama & Benchmark
**Amaç:** Modelin gerçekten anlamlı olduğunu kanıtlamak.

- Kendi xG'ni StatsBomb xG'siyle karşılaştır (şut bazında korelasyon).
- **Toplulaştırma testi:** oyuncu/takım bazında sezonluk toplam xG ile gerçek gol sayısı korele mi? (xG'nin "tahmin gücü" testi).
- Hata analizi: model nerede yanılıyor?

**Çıktı:** Doğrulama raporu.

---

## Faz 6 — Oyuncu Değerlendirme / Scouting
**Amaç:** xG modelini oyuncu bazında analiz aracına dönüştürmek.

- Oyuncu bazında: gerçek gol − beklenen gol (xG) → "bitiricilik" sinyali (over/under-performer).
- Takım bazında: toplam xG vs gerçek gol → hangi takımlar şansları üstünde/altında oynuyor?
- En çok over/under-perform eden oyuncuları listele, minimum şut eşiği koyarak küçük örneklem yanılsamasını önle.

**Çıktı:** Oyuncu/takım değerlendirme modülü + scouting raporu.

---

## Faz 7 — Sunum & Portföy
- README: problem, veri, yaklaşım, sonuçlar, kararların gerekçesi.
- "Neyi neden seçtik" anlatısı (STAR çerçevesine de uyar).
- Görseller: kalibrasyon, SHAP, scouting çıktıları.

---

## Önemli Tuzaklar (özet)
1. StatsBomb xG'yi feature olarak kullanmak → leakage. Sadece benchmark.
2. Penaltıları modele katmak → modeli bozar, ayır.
3. Rastgele split → maç bazında split daha dürüst.
4. Sadece accuracy'ye bakmak → nadir olayda yanıltıcı; log-loss + kalibrasyon esas.
5. Freeze-frame'i atlamak → atlamazsan model sıradanlıktan çıkar.

---

## Claude Code ile Başlangıç
1. Boş repo aç, içine bu dosyayı ve `CLAUDE.md`'yi koy.
2. Claude Code'a faz faz ilerlemesini söyle — hepsini tek seferde isteme.

**Önerilen ilk prompt:**
> "CLAUDE.md ve yol haritasını oku. Faz 0 ve Faz 1'i uygulayalım: repo iskeletini kur, `statsbombpy` ile bir yarışmanın tüm şut event'lerini çekip `is_goal` hedefiyle temiz bir parquet üret, ve temel EDA notebook'u hazırla. Her veri kapsamı kararını (penaltı/serbest vuruş hariç tutma vb.) bana gerekçesiyle sor."
