# Faz 4 -- Gradient Boosting + Kalibrasyon

## Amac
Baseline LR'yi gradient boosting ile gecmek, kalibrasyon dogrulamak, SHAP ile
yorumlamak. Final model.

---

## Adimlar

### 4.1: src/models/gradient_boosting.py
- LightGBM (daha hizli train, categorik native destek).
- Hyperparameter tuning: 5-fold GroupKFold CV (group=match_id).
- Tuning alani: n_estimators, learning_rate, max_depth, num_leaves,
  min_child_samples, subsample, colsample_bytree.
- Objective: binary logloss.

### 4.2: Kalibrasyon kontrolu
- Reliability diagram: LightGBM vs Baseline LR vs StatsBomb xG (uc yana yana).
- Eger LightGBM miscalibrated ise (boosting'de sik gorulur):
  Platt scaling veya isotonic regression uygula (CalibratedClassifierCV).

### 4.3: SHAP Analizi
- TreeExplainer ile global feature importance.
- Beeswarm plot: hangi feature ne yonde etki ediyor.
- Beklenti: distance ve angle baskin olmali. Degilse bug.

### 4.4: Model karsilastirma tablosu
- Baseline LR vs LightGBM vs StatsBomb xG (log-loss, Brier, AUC).
- Hepsi ayni test seti uzerinde.

### 4.5: Final model secimi ve kayit
- En iyi modeli (kalibrasyon + metrik) sec.
- models/final_xg.joblib olarak kaydet.

---

## Basari Kriterleri
- [x] LightGBM log-loss (0.2487) < Baseline LR log-loss (0.2489).
- [x] Calibration: ECE=0.0176, post-hoc gerekmedi.
- [x] SHAP: ff_dist_to_gk (#1), geom_angle_rad (#2), geom_distance (#7). PASS.
- [x] 3-model karsilastirma tablosu rapor edildi.

---

## Sonuclar (2026-06-07)

### Best Config (CV)
n_estimators=1000, lr=0.02, depth=5, leaves=31, subsample=0.7, colsample=0.7
CV log-loss: 0.2557

### Test Karsilastirma
| Metrik | LightGBM | Baseline LR | StatsBomb |
|---|---|---|---|
| Log-loss | 0.2487 | 0.2489 | 0.2546 |
| Brier | 0.0711 | 0.0712 | 0.0724 |
| AUC | 0.8346 | 0.8349 | 0.8245 |

### SHAP Top 5
1. ff_dist_to_gk (0.416)
2. geom_angle_rad (0.400)
3. ff_n_opponents_in_cone (0.246)
4. ff_dist_nearest_opponent (0.190)
5. shot_body_part (0.139)

### Yorum
LR ve LightGBM cok yakin -- feature engineering zaten sinyali lineer olarak
yakalanabilir hale getirdi. Bu iyi haber: modelin "kara kutu"ya ihtiyaci yok,
feature'lar iyi tasarlanmis.

StatsBomb'u her iki model de rahatca geciyor (%2.3 log-loss iyilesmesi).

---

Durum: TAMAMLANDI (2026-06-07)
