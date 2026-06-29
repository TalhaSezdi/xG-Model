# Faz 3 -- Baseline Model (Logistic Regression)

## Amac
Yorumlanabilir, kalibrasyon-duyarli bir baseline model kurup metrik referans
noktasi olusturmak. Faz 4'te gradient boosting ile karsilastiracagimiz temel.

---

## Mimari Kararlar

### Train/Test Split
- **Mac bazli split** (random shot split DEGIL). Ayni macin sutlari ayni sette kalir.
- Stratify: is_goal oraniyla (~%10) stratified group split.
- Oran: %80 train / %20 test.
- Neden: Ayni mac icerisindeki sutlar bagimli (ayni takim, ayni savunma hatti).
  Random split data leakage yaratir.

### Pipeline (leakage-proof)
1. Split ONCE yap.
2. OneHotEncoder'i SADECE train uzerinde fit et.
3. Numeric NaN'lari median impute (train medianlariyla).
4. LR'yi fit et.
5. Test'te predict_proba al.

### Model
- sklearn LogisticRegression, C=1.0 (default regularization).
- class_weight: Kullanmiyoruz -- xG'de "olasilik kalitesi" esas, class balance
  accuracy icin gerekli ama biz accuracy'ye bakmiyoruz. Calibration bozulur.
- solver: lbfgs, max_iter=1000.

### Metrikler (oncelik sirasi)
1. Log-loss (primary)
2. Brier score
3. Calibration curve (reliability diagram) -- gorsel
4. ROC-AUC (secondary, discrimination)
5. StatsBomb xG'ye karsi: ayni metrikler (log-loss, Brier) benchmark olarak.

### Kalibrasyon Karsilastirma
- Reliability diagram: model %20 dediginde gercekten ~%20 gol oluyor mu?
- StatsBomb xG icin ayni grafik (ikisi yan yana).

---

## Adimlar

### 3.1: src/models/baseline.py
- `BaselineXGModel` sinifi: fit, predict_proba, evaluate.
- Pipeline icinde: imputer + OHE + LR.

### 3.2: src/preprocess/splitter.py
- `MatchBasedSplitter`: GroupShuffleSplit wrapper (group=match_id, stratify=is_goal).

### 3.3: scripts/train.py
- Veri yukle, split yap, pipeline fit, evaluate, sonuclari raporla.
- Modeli joblib ile kaydet (models/ dizini).

### 3.4: src/evaluation/metrics.py
- log_loss, brier_score, roc_auc hesapla.
- calibration_curve wrapper (sklearn).
- reliability_diagram ciz.
- StatsBomb xG'yi ayni metriklerle karsilastir.

### 3.5: LR katsayilari yorumlama
- Coefficients tablosu: hangi feature'lar pozitif/negatif?
- Sanity: distance negatif, angle pozitif, open_goal pozitif olmali.

---

## Basari Kriterleri
- [x] Mac bazli train/test split calisiyor (0 overlap, gol orani train/test ~%10.1/%10.2).
- [x] Pipeline train-only fit, test'te sadece transform.
- [x] Log-loss ve Brier score rapor edildi.
- [x] Calibration curve (reliability diagram) cizildi (reports/baseline_calibration.png).
- [x] StatsBomb xG benchmark olarak karsilastirildi -- 3 metrikte de LR kazandi.
- [x] LR katsayilari yorumlandi (distance negatif, angle pozitif, open_goal pozitif).

---

## Sonuclar (2026-06-07)

### Baseline LR vs StatsBomb xG (test seti, n=13,143)
| Metrik | Baseline LR | StatsBomb xG | Kazanan |
|---|---|---|---|
| Log-loss | 0.2489 | 0.2546 | Baseline LR |
| Brier score | 0.0712 | 0.0724 | Baseline LR |
| ROC-AUC | 0.8349 | 0.8245 | Baseline LR |

### Split
- Train: 52,679 sut / 2,123 mac
- Test: 13,143 sut / 528 mac
- Match overlap: 0

### Top LR Katsayilari (en etkili feature'lar)
1. shot_deflected: +2.25 (sekme = gol sansi)
2. geom_distance: -1.94 (uzak = dusuk xG)
3. shot_open_goal: +0.59
4. kp_is_through_ball: +0.57
5. ff_n_opponents_in_cone: -0.37

Tum sanity check'ler gecti.

---

Durum: TAMAMLANDI (2026-06-07)
