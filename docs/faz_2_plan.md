# Faz 2 -- Feature Engineering Plani

## Amac
`shots_raw.parquet` -> `shots_features.parquet` (modele girecek temiz feature matrisi).

Yol haritasinda belirtilen "freeze_frame'den savunmaci/kaleci ozellikleri" projeyi siradan bir xG'den ayiran kisim. Bu fazda OOP tabanli, scikit-learn uyumlu transformer'lar yazip pipeline kuracagiz.

---

## Mimari Kararlar

### Modul yapisi (src/features/)
```
src/features/
  __init__.py
  geometry.py          # Distance, angle hesaplamalari (saf fonksiyonlar)
  freeze_frame.py      # Freeze frame parse ve savunmaci/kaleci ozellikleri
  transformers.py      # sklearn-uyumlu Transformer siniflari
  pipeline.py          # Tam feature pipeline tanimi
```

### Neden OOP / Transformer?
- scikit-learn Pipeline ile entegrasyon (CLAUDE.md kurali: "tum preprocessing pipeline icinde, train data uzerinde fit edilmeli")
- Test edilebilirlik, yeniden kullanim
- `fit_transform` / `transform` ayrimi -> data leakage'i fiziken imkansiz hale getirir

---

## Adim 2.1: Geometrik Ozellikler (Saf Fonksiyonlar)

### Dosya: `src/features/geometry.py`

StatsBomb pitch koordinatlari: x=[0,120], y=[0,80]. Kale x=120'de, y=[36,44] arasinda (genislik 8 yard ~7.32m).

#### Hesaplanacaklar:
1. **`distance_to_goal(x, y)`**: Sut konumundan kale merkezine (120, 40) Oklid mesafesi.
2. **`angle_to_goal(x, y)`**: Iki direk arasinda kalan aci (radyan + derece).
   - Formul: `atan2((g * x_dist), (x_dist^2 + y_dist^2 - (g/2)^2))` (g = kale genisligi)
   - Daha guvenli yontem: Iki direkten gelen vektorler arasindaki aci (cross/dot product).
3. **`x_dist_to_goal_line(x)`**: Kale cizgisine x-mesafesi.
4. **`y_dist_to_goal_center(y)`**: Kale merkezinden y-sapma (mutlak deger).
5. **`is_left_side(y)`** veya **`y_normalized`**: Sahanin hangi tarafi.

### Test:
- Pytest yerine basit assertion'larla: Penalti noktasinda (108, 40) mesafe ~12m, aci genis olmali.
- Sut x=120, y=40 -> mesafe 0 olmali (degenerate case, log icin).

---

## Adim 2.2: Freeze Frame Ozellikleri

### Dosya: `src/features/freeze_frame.py`

Freeze frame, sut anindaki tum oyuncularin (top atan + 21 kisi) konumlarini iceren JSON. Veri zaten string olarak parquet'te.

Ornek freeze_frame elemani:
```python
{
  "location": [105.3, 42.1],
  "player": {"id": 1234, "name": "..."},
  "position": {"id": 23, "name": "Goalkeeper"},
  "teammate": False  # False = rakip, True = ayni takim
}
```

#### Hesaplanacak Ozellikler:
1. **`num_defenders_in_cone`**: Sut yapan ile her iki direk arasinda olusan ucgenin icindeki rakip oyuncu sayisi.
   - Geometri: Ucgen ucleri (shot_x, shot_y), (120, 36), (120, 44). Bir noktanin ucgen icinde olup olmadigi.
2. **`num_defenders_within_3m`**: Top etrafinda 3m yaricapta kac rakip var.
3. **`distance_to_nearest_defender`**: En yakin rakibe Oklid mesafesi.
4. **`distance_to_gk`**: Kaleciye mesafe.
5. **`gk_distance_from_goal`**: Kalecinin kale cizgisinden uzakligi (kaleci cikmis mi?).
6. **`gk_y_offset_from_center`**: Kalecinin kale merkezinden y-sapmasi.
7. **`num_teammates_in_box`**: Ceza sahasi icindeki takim arkadasi sayisi (asist yardimi sezgisi).

#### Edge case'ler:
- Freeze frame bos olabilir (eski maclar veya bazi sutlar) -> default degerler (NaN veya marker degeri).
- Kaleci yoksa (cok nadir) -> distance_to_gk = NaN, ayri bir flag.

---

## Adim 2.3: Kategorik Ozellikler

### Mevcut kategorikler:
- `shot_body_part` (Right Foot, Left Foot, Head, Other) -- ~4 deger
- `shot_type` (Open Play, Free Kick, Corner) -- 3 deger
- `shot_technique` (Normal, Half Volley, Volley, Lob, Overhead Kick, Backheel, Diving Header) -- ~7 deger
- `play_pattern` (Regular Play, From Corner, From Free Kick, From Throw In, From Counter, ...) -- ~9 deger
- `under_pressure` -- zaten boolean, encoding gereksiz

### Encoding stratejisi:
- **One-Hot Encoding** (sklearn `OneHotEncoder` -- `handle_unknown='ignore'` ile robust)
- Yuksek kardinaliteli alan yok, OHE guvenli.

---

## Adim 2.4: Transformer Siniflari

### Dosya: `src/features/transformers.py`

OOP, sklearn `BaseEstimator + TransformerMixin` mirasi:

```python
class GeometryFeatures(BaseEstimator, TransformerMixin):
    """Adds distance, angle, x_dist_to_goal, y_dist_to_center."""

class FreezeFrameFeatures(BaseEstimator, TransformerMixin):
    """Parses freeze_frame JSON and adds defender/GK features."""

class ShotFeaturePipeline(BaseEstimator, TransformerMixin):
    """Orchestrates: geometry -> freeze_frame -> categorical encoding."""
```

Her transformer'in:
- `fit(X, y=None)` -> egitilebilir parametre yoksa pass.
- `transform(X)` -> yeni kolonlari ekler veya secer.
- `get_feature_names_out()` -> SHAP/feature importance icin sart.

---

## Adim 2.5: Pipeline ve Cikti

### Dosya: `src/features/pipeline.py`

```python
def build_feature_pipeline() -> Pipeline:
    return Pipeline([
        ("geometry", GeometryFeatures()),
        ("freeze_frame", FreezeFrameFeatures()),
        ("categorical", ColumnTransformer([
            ("ohe", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ], remainder="passthrough")),
    ])
```

### Script: `scripts/build_features.py`
- `shots_raw.parquet` yukle.
- Pipeline'i fit_transform et (henuz split yok -- bu Faz 3'te olacak).
- **KRITIK:** Bu adimda pipeline'i fit etmemiz veri sizdirir mi? Hayir -- cunku:
  - Geometric features stateless (parametresiz).
  - Freeze frame parse stateless.
  - One-hot encoder fit ediyor ama train/test split Faz 3'te yapilacak ve pipeline o zaman yeniden fit edilecek. Bu adimda sadece feature matrisini *gozlemlemek* icin uretiyoruz; modele yine pipeline icinden gidecegiz.
  - Alternatif: Bu adimda sadece geometri + freeze_frame'i kosup, OHE'yi Faz 3'e birakabiliriz. Daha temiz.

**Karar:** Bu fazda sadece **stateless feature'lari** (geometry + freeze_frame) hesaplayip `shots_features.parquet` kaydedecegiz. OHE pipeline tanimi `src/features/pipeline.py`'de hazir bekleyecek, Faz 3'te train data uzerinde fit edilecek.

---

## Adim 2.6: Dogrulama Notebook'u

### Dosya: `notebooks/02_feature_validation.ipynb`

- Yeni feature'lari gorsellestir:
  - Distance vs goal rate (Faz 1'deki ile ayni cikmali -- sanity check).
  - Angle vs goal rate.
  - num_defenders_in_cone vs goal rate (bekleme: arttikca azalir).
  - distance_to_gk vs goal rate.
- Feature'lar arasi korelasyon matrisi (multicollinearity sezgisi).
- StatsBomb xG ile geometric feature'lar arasi korelasyon.

---

## Tuzaklar

1. **statsbomb_xg'yi ASLA feature olarak alma** -- pipeline'a girmemeli. Parquet'te ayri kolon olarak kalsin.
2. **Freeze frame koordinat sistemi**: Sut yapan takim hep sagdan saldiriyor (StatsBomb otomatik flipler). Yine de dogrula.
3. **Aci hesabi**: Sut x>120 (kale arkasi) olabilir? Olmamali ama edge case kontrol.
4. **NaN politikasi**: Freeze frame eksik olan sutlar var mi? Hesapla, raporla.

---

## Basari Kriterleri

- [x] `src/features/geometry.py` -- saf, test edilebilir fonksiyonlar.
- [x] `src/features/freeze_frame.py` -- JSON parse + 10 defender/GK feature.
- [x] `src/features/transformers.py` -- sklearn-uyumlu siniflar (GeometryFeatures, FreezeFrameFeatures).
- [x] `src/features/pipeline.py` -- tam pipeline tanimi (stateless + categorical).
- [x] `scripts/build_features.py` -> `data/shots_features.parquet` (geometry + freeze_frame).
- [x] `notebooks/02_feature_validation.ipynb` -- gorsel + sayisal validasyon.
- [x] Distance ve angle'in gol oraniyla beklenen iliskiyi gosterdigi dogrulandi.
- [x] num_defenders_in_cone arttikca gol orani dustugu gosterildi.

---

## Sonuclar

- **Cikti:** `data/shots_features.parquet` (65,822 sut x 34 kolon, 16 yeni feature).
- **Sure:** Tum transform 3.2 saniye.
- **Freeze frame kapsami:** %100 sut, %99.9 kaleci gorunur (58 sutta GK NaN).

### Sanity Check Sonuclari
| Feature | Gol orani araligi | Dogrulama |
|---|---|---|
| geom_distance | 31.7% (yakin) -> 1.9% (uzak) | Monotonik azalir |
| geom_angle | 3.1% (dar) -> 31.7% (genis) | Monotonik artar |
| ff_n_opponents_in_cone | 44.2% (0 def) -> 4.7% (3 def) | Monotonik azalir |

### StatsBomb xG ile Korelasyon (sadece dogrulama icin, feature degil)
| Feature | Pearson r |
|---|---|
| geom_angle_rad | +0.69 |
| ff_dist_to_gk | -0.57 |
| geom_distance | -0.56 |
| geom_x_dist | -0.47 |
| geom_in_box | +0.41 |

Acik ve mesafe baskin sinyal -- xG literaturuyle bire bir uyumlu. Faz 3 modellemeye hazir.

---

Durum: TAMAMLANDI (2026-06-06)
