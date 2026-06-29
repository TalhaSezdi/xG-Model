# Faz 0 & Faz 1 -- Uygulama Plani

## Onaylanan Veri Kapsami Kararlari
1. **Yarisma kapsami:** Tum erkek profesyonel lig/turnuva verileri (La Liga, PL, CL, WC, Euro vb.). Kadin futbolu haric.
2. **Penaltilar:** Haric tutulacak (sabit xG ~0.76, modele bilgi katmiyor).
3. **Serbest vuruslar:** Dahil edilecek, `play_pattern` feature'i ile ayirt edilecek.
4. **Own goal'ler:** Haric tutulacak (gurultu).

---

## Faz 0 -- Repo Iskeleti & Ortam Kurulumu

### Adim 0.1: Dizin yapisi
```
XG_PROJE/
  CLAUDE.md
  xG_Projesi_Yol_Haritasi.md
  requirements.txt
  docs/
  data/           # .gitignore'da, veri dosyalari burada
  notebooks/      # EDA notebook'lari
  src/
    __init__.py
    features/
      __init__.py
    models/
      __init__.py
    preprocess/
      __init__.py
    calibration/
      __init__.py
    evaluation/
      __init__.py
    scouting/
      __init__.py
  scripts/
    extract_shots.py
    train.py
    evaluate.py
```

### Adim 0.2: requirements.txt
Temel bagimliliklar: statsbombpy, pandas, pyarrow, scikit-learn, xgboost, lightgbm, shap, matplotlib, seaborn, jupyter, notebook.

### Adim 0.3: .gitignore
data/ dizini, env/, __pycache__, .ipynb_checkpoints vb.

---

## Faz 1 -- Veri Cikarimi & EDA

### Adim 1.1: extract_shots.py
- statsbombpy ile tum yarismalari listele.
- Erkek profesyonel yarismalari filtrele.
- Her yarismanin her sezonundaki tum maclari cek.
- Her mactan sut event'lerini cek.
- Filtreleme:
  - Penaltilari haric tut (`shot_type == "Penalty"` olan satirlar).
  - Own goal'leri haric tut (`shot_outcome == "Own Goal"` iceren satirlar veya ilgili flag).
  - Serbest vuruslari dahil et.
- Hedef degisken: `is_goal` (1 = Goal, 0 = diger tum sonuclar).
- Her sut icin saklanacak ham alanlar:
  - match_id, competition, season
  - location (x, y)
  - shot_outcome, shot_type, shot_body_part, shot_technique
  - play_pattern, under_pressure
  - shot_freeze_frame (JSON olarak)
  - statsbomb_xg (SADECE benchmark icin, feature olarak KULLANILMAYACAK)
  - player, team
- Cikti: `data/shots_raw.parquet`

### Adim 1.2: Veri dogrulama
- Satir sayisi, null orani, is_goal dagilimi (beklenen ~%10) kontrol.
- Duplicate kontrol (ayni mac, ayni dakika, ayni oyuncu).

### Adim 1.3: EDA Notebook (notebooks/01_eda.ipynb)
- Temel istatistikler: toplam sut, gol orani, yarisma bazinda dagilim.
- Sut konum dagilimlari (scatter/heatmap).
- Bolgeye gore gol donusum orani.
- Body part, play pattern, shot type bazinda dagilimlar.
- Sinif dengesizligi gorsellestirmesi.
- statsbomb_xg dagilimi (benchmark referans).

---

## Basari Kriterleri
- [x] Repo yapisi CLAUDE.md'deki arsitekture uygun.
- [x] `scripts/extract_shots.py` calisip `data/shots_raw.parquet` uretiyor.
- [x] Penaltilar ve own goal'ler haric tutulmus (1032 penalti cikarildi).
- [x] Serbest vuruslar dahil, play_pattern ile isaretli (3524 serbest vurus).
- [x] is_goal orani ~%8-12 arasinda (gerceklesen: %10.13).
- [x] EDA notebook temel gorselleri icerir.

---

## Sonuclar

- **Toplam sut:** 65,822 (2,651 mac, 17 yarisma, 67 sezon)
- **Gol orani:** %10.13 (6,667 gol)
- **Null deger:** 0
- **Filtreler:** 1,032 penalti haric tutuldu, 0 own goal bulundu
- **Serbest vuruslar:** 3,524 adet, dahil
- **Potansiyel duplicate:** 20 (ihmal edilebilir)
- **En buyuk yarisma:** La Liga (20,936 sut)

---

Durum: TAMAMLANDI (2026-06-06)
