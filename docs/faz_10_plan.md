# Faz 10 -- Tablo Iyilestirmeleri + Takim Logolari

## Amac
- Tum tablolari okunabilir, anlamli Turkce kolonlarla yeniden duzenle.
- st.column_config ile sayi formatlama, progress bar, renk kodlama.
- En cok sutla gecen ~80 takim icin logo ekle (TheSportsDB ucretsiz API).

---

## Adimlar

### 10.1: src/dashboard/logos.py
- TheSportsDB ucretsiz API ile takim logo URL'lerini once-off cek.
- Sonuclari `data/team_logos.json` olarak kaydet (API her acilista cagirilmaz).
- Fallback: logo yoksa takim baslangic harfi ile renk avatar.

### 10.2: Tablo iyilestirmeleri (app.py)

**Oyuncu Tablosu kolonlari:**
| Eski | Yeni (Turkce) | Format |
|---|---|---|
| rank | Sira | sayi |
| player | Oyuncu | metin |
| shots | Sut Sayisi | sayi |
| actual_goals | Gol | sayi |
| xg_sum | Toplam xG | 2 ondalik |
| xg_per_shot | Sut Basi xG | progress bar (0-1) |
| actual_conversion | Gercek Donusum | progress bar (0-1) |
| goals_above_xg | Gol - xG Farki | +/- renk |
| goals_above_xg_se | Guven Araligi (+/-) | 2 ondalik |

**Takim Tablosu kolonlari:**
| Eski | Yeni (Turkce) | Format |
|---|---|---|
| team | Takim | logo + isim |
| shots_for | Atilan Sut | sayi |
| goals_for | Atilan Gol | sayi |
| xg_for | Hucum xG | 2 ondalik |
| attacking_over_under | Hucum Farki (Gol-xG) | +/- renk |
| xg_against | Savunma xG | 2 ondalik |
| goals_against | Yenilen Gol | sayi |
| defensive_over_under | Savunma Farki | +/- renk |
| net_goals | Net Gol | +/- renk |

**Lig Tablosu (Overview):**
- Kolon adlari Turkce.
- Gol orani sutunu progress bar.

### 10.3: Logolari onceden indir
- `scripts/fetch_logos.py` -- TheSportsDB'den JSON cek, `data/team_logos.json` yaz.
- app.py bu JSON'u okur, tabloda `st.column_config.ImageColumn` ile gosterir.

---

## Basari Kriterleri
- [ ] Tum tablo kolonlari Turkce ve aciklayici.
- [ ] Sayi sutunlarinda uygun format (%, ondalik, +/-).
- [ ] Takim tablosunda logo gosteriliyor (en az 40 takim).
- [ ] Hiz: logo JSON onceden yatirildigi icin app yavaslamamali.

---

Durum: BEKLEMEDE
