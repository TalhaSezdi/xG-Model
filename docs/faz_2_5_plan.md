# Faz 2.5 -- Feature Genisletme Plani

## Amac
Agac modeline gercek yeni bilgi katacak feature'lari ekle. Sadece mevcut
kolonlardan tureyemeyen, veri setinde HIC olmayan sinyaller. Re-extraction
gerektiren uc kategori.

## Kapsam Karari (gerekce)
Eklenen her sey "agac modeli bunu zaten ogrenir mi?" suzgecinden gecti:
- **Eklenenler:** yeni bilgi (top nasil geldi, native flag'ler, skor durumu).
- **Atlananlar:** mevcut kolonlarin yeniden kodlamasi/etkilesimi (gk_angle_offset,
  density ratio, koridor), leakage riskli (preferred_foot), native flag varken
  heuristic (is_open_goal kurali).

---

## Adim 2.5.1: extract_shots.py Genisletme

Mevcut script ham event'leri atip sadece sut alanlarini sakliyordu. Yeniden
yapilandirma: her mac icin TUM event'lere erisim gerek (key pass lookup + kosan skor).

### A. Native StatsBomb Shot Flag'leri (bedava, ground-truth)
StatsBomb shot objesinde hazir gelen, sadece True oldugunda yazilan flag'ler:
- `shot_open_goal` <- shot.open_goal
- `shot_one_on_one` <- shot.one_on_one
- `shot_first_time` <- shot.first_time (gelisine vurus)
- `shot_deflected` <- shot.deflected
- `shot_follows_dribble` <- shot.follows_dribble
- `shot_aerial_won` <- shot.aerial_won
- `shot_redirect` <- shot.redirect

### B. Key Pass (Asist) Ozellikleri
shot.key_pass_id ile ayni macin pas event'ine baglan, sut oncesi pasin
karakterini cek. Asist yoksa (rebound, kendi calimi) flag 0 + None.
- `kp_has_key_pass` (1/0)
- `kp_pass_height` (Ground Pass / Low Pass / High Pass) -- kategorik
- `kp_is_cross` (1/0)
- `kp_is_cutback` (1/0) <- pass.cut_back
- `kp_is_through_ball` (1/0) <- pass.technique == "Through Ball"
- `kp_pass_length` (float)
- `kp_pass_angle` (float)

### C. Game State (Skor Durumu)
Mac event'lerini kronolojik isleyip sut anindaki kosan skoru tut.
- `score_diff` = (sut atan takim golleri) - (rakip golleri), SUTTAN ONCE.
- Goller: outcome Goal olan sutlar + "Own Goal For" event'leri (penalti golleri
  dahil cunku gercek skoru degistirir). Snapshot golden ONCE alinir (sutun kendi
  sonucu kendi feature'ina sizmaz).

---

## Adim 2.5.2: Re-extraction
- `extract_shots.py` yeniden calistir -> `data/shots_raw.parquet` (uzerine yaz).
- Dogrula: yeni kolonlar dolu, null oranlari makul, gol orani hala ~%10.

## Adim 2.5.3: build_features.py Yeniden Calistir
- `data/shots_features.parquet` yeniden uret (stateless pipeline yeni kolonlari tasir).

## Adim 2.5.4: pipeline.py Guncelle
- Yeni numeric/boolean feature'lari NUMERIC/BOOLEAN listelerine ekle.
- `kp_pass_height`'i CATEGORICAL_COLUMNS'a ekle (Faz 3'te OHE).

## Adim 2.5.5: Validation
- Notebook 02'ye yeni feature'larin gol oraniyla iliskisi (sanity):
  - through_ball / cutback / cross -> gol orani (through_ball ve cutback yuksek olmali).
  - one_on_one, open_goal, first_time -> gol orani.
  - score_diff -> gol orani (geride olan takim daha kotu pozisyondan vurur).

---

## Atlananlar (Backlog / agac modeli icin gereksiz)
- gk_angle_offset -- GK konumu zaten var, agac ogrenir.
- defense_density_ratio -- iki kolonun orani, agac ogrenir.
- opponents_in_direct_corridor -- koni metriklerinin ince varyanti, marjinal.
- is_open_goal heuristic -- native shot.open_goal var.
- is_preferred_foot -- open data'da yok, leakage riski, dusuk getiri.

---

## Basari Kriterleri
- [x] extract_shots.py native flag + key pass + score_diff ekliyor.
- [x] shots_raw.parquet yeniden uretildi (65,822 sut x 33 kolon), yeni kolonlar dogru.
- [x] shots_features.parquet guncel (49 kolon).
- [x] pipeline.py yeni feature'lari iceriyor (kategorik + numeric + boolean listeleri).
- [x] Validation: through_ball/cutback gol oranini yukseltiyor (sanity gecti).

---

## Sonuclar (2026-06-06)

Re-extraction: 65,822 sut korundu, gol orani %10.13 (degismedi), 0 hata.
14 yeni feature eklendi. Tutarlilik: kp null sayisi = asistsiz sut sayisi (18,438).

### Sanity Check (gol orani lift = baseline'a gore kat)
| Feature | Gol orani | Lift |
|---|---|---|
| shot_open_goal | %74.7 | 7.4x |
| shot_deflected | %36.4 | 3.6x |
| shot_one_on_one | %25.1 | 2.5x |
| kp_is_through_ball | %30.4 | 3.0x |
| kp_is_cutback | %16.2 | 1.6x |
| kp_is_cross | %14.8 | 1.5x |
| shot_aerial_won | %7.5 | 0.74x (kafa, dusuk -- dogru) |

### Game State (kosan skor mantiginin kaniti)
score_diff -3 -> %7.3 gol / 20.3m mesafe; +3 -> %16.4 gol / 18.5m mesafe.
Tam monotonik: geride olan uzaktan/umutsuz vuruyor, onde olan yakindan.

### Deger Katma Noktasi
score_diff, is_goal ile korele (0.066) ama StatsBomb xG ile zayif (0.089).
StatsBomb game state kullanmiyor -> modelimizin fark yaratabilecegi alan.

### Atlananlar (karara sadik kalindi)
gk_angle_offset, defense_density_ratio, opponents_in_direct_corridor,
is_open_goal heuristic, is_preferred_foot -- agac modeli icin gereksiz/riskli.

---

Durum: TAMAMLANDI (2026-06-06)
