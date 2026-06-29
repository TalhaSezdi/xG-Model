# Faz 8 -- Interaktif Dashboard (Streamlit)

## Amac
Projeyi birine anlatirken kullanilacak interaktif bir web arayuzu.
Streamlit ile tek komutla calisir, grafikleri canli gosterir, filtreleme yapar.

---

## Sayfa Yapisi

### Sayfa 1: Proje Ozeti (Overview)
- Baslik + tek cumle aciklama.
- 4 buyuk metrik karti: Log-loss, Brier, AUC, ECE.
- 3-model karsilastirma tablosu (LightGBM vs LR vs StatsBomb).
- Veri seti istatistikleri: toplam sut, gol orani, mac sayisi, lig sayisi.

### Sayfa 2: Model Performansi
- Kalibrasyon egrisi (interaktif): LightGBM vs StatsBomb vs LR.
- Olasilik dagilimi: model xG histogrami (gol vs gol degil).
- xG decile karsilastirmasi: bar chart (model vs StatsBomb vs gercek oran).

### Sayfa 3: Feature Importance (SHAP)
- SHAP bar chart (top 15).
- Feature aciklamalari tablo halinde.
- Tek feature secince: o feature'in xG ile iliskisi (scatter/box).

### Sayfa 4: Oyuncu Scouting
- Aranabilir/filtrelenebilir oyuncu tablosu.
- Filtreler: min sut sayisi (slider), lig secimi.
- Overperformer/underperformer bar chart (top N, slider ile N secimi).
- Tek oyuncu secince: o oyuncunun sut dagilimi detayi.

### Sayfa 5: Takim Analizi
- Takim tablosu: hucum + savunma over/under.
- Scatter: xG vs actual goals (hover ile takim adi).
- Hucum vs savunma over/under grouped bar chart.

---

## Teknik
- Streamlit (pip install streamlit).
- Plotly Express grafikleri (interaktif, hover, zoom).
- Tek dosya: app.py (proje kokunde).
- Calistirma: `streamlit run app.py`
- Veri: shots_features.parquet + model predictions (on-the-fly).

---

## Basari Kriterleri
- [x] 5 sayfa calisiyor (Overview, Model Performance, Feature Importance, Player Scouting, Team Analysis).
- [x] Plotly grafikleri: hover, zoom, pan interaktif.
- [x] Oyuncu tablosu aranabilir + filtrelenebilir (min sut slider, lig secimi).
- [x] Tek komutla basliyor: `streamlit run app.py`

---

Durum: TAMAMLANDI (2026-06-08)
