# Faz 6 -- Player Scouting Framework

## Amac
xG modelini oyuncu degerlendirme aracina donusturmek. Overperformer/underperformer
tespiti, finishing skill sinyali cikarma, takim bazli analiz.

---

## Adimlar

### 6.1: src/scouting/player_analysis.py
- Oyuncu bazli aggregasyon sinifi.
- Input: shots_features.parquet + model predictions.
- Per player: actual_goals, xG_sum, goals_minus_xG, shots, xG_per_shot, conversion_rate.
- Minimum sut threshold (default=20) ile filtrele.
- Confidence band: goals-xG'nin standard error'u (binom varyans).

### 6.2: src/scouting/team_analysis.py
- Takim bazli: xG_for, xG_against, goals_for, goals_against.
- Net xG = xG_for - xG_against.
- Over/under = (goals_for - xG_for) -- hucum performansi.
- Savunma: goals_against - xG_against -- savunma performansi.

### 6.3: scripts/scouting.py
- Tum veri setinde (train+test) model tahminlerini uret.
- Oyuncu ve takim tablolarini hesapla.
- Top 20 overperformer + underperformer listesi.
- Gorseller: bar chart (top/bottom 15 oyuncu), scatter (xG vs goals).

### 6.4: Raporlar
- reports/scouting_players.csv: Tam oyuncu tablosu.
- reports/scouting_teams.csv: Tam takim tablosu.
- reports/scouting_top_players.png: Over/under bar chart.
- reports/scouting_team_scatter.png: Takim xG vs goals scatter.

---

## Basari Kriterleri
- [x] Oyuncu tablosu: 878 oyuncu, min 20 sut, confidence band dahil.
- [x] Takim tablosu: 242 takim, hucum + savunma over/under.
- [x] Top overperformer: Messi (#1, +80.6), Suarez (#2), Mbappe (#3). PASS.
- [x] 4 rapor dosyasi uretildi (2x CSV, 2x PNG).

---

## Sonuclar (2026-06-08)

### Oyuncu Tablosu (min 20 sut, 878 oyuncu)
| Rank | Oyuncu | Shots | Goals | xG | G-xG |
|---|---|---|---|---|---|
| 1 | Messi | 2583 | 442 | 361.4 | +80.6 |
| 2 | Suarez | 628 | 132 | 118.7 | +13.3 |
| 3 | Mbappe | 294 | 55 | 45.7 | +9.3 |
| ... | ... | ... | ... | ... | ... |
| 864 | Braithwaite | 164 | 14 | 23.7 | -9.7 |
| 865 | Iniesta | 370 | 24 | 32.2 | -8.2 |
| 866 | Dzeko | 86 | 6 | 13.2 | -7.2 |

Yorumlar:
- Messi'nin +80.6 finisher sinyali istatistiksel olarak cok anlamli (SE=17.6, z=4.6).
- Iniesta ve Dzeko -- cok sut atiyor ama gol donusturmede zayif.
- Ronaldo net negatif (-4.0): veri setinde kucuk ornek (390 sut) ve StatsBomb
  acik veri kapsami Ronaldo'nun en verimli donemlerini tam yakalamamis olabilir.

### Takim Tablosu
- Barcelona basicilik: +135 gol over xG -- Messi etkisi net.
- PSG: hem hucum (+13.7) hem savunma (+14.9) over-performer.
- 242 takim qualify etti (min 30 sut).

---

Durum: TAMAMLANDI (2026-06-08)
