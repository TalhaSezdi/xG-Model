"""
Interactive xG dashboard -- Streamlit application.

Run with: streamlit run app.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
import streamlit as st
from streamlit_option_menu import option_menu

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gradient_boosting import prepare_lgbm_data
from src.preprocess.splitter import match_based_split
from src.scouting.player_analysis import build_player_table
from src.scouting.team_analysis import build_team_table
from src.dashboard.logos import add_logo_column
from src.dashboard.theme import (
    PALETTE,
    SEQ_PRIMARY,
    DIVERGING,
    inject_css,
    register_plotly_template,
    style_fig,
    kpi_card,
    render_kpis,
    section_header,
    hero,
    sidebar_brand,
)

DATA_PATH = PROJECT_ROOT / "data" / "shots_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

GOAL_MAP = {0: PALETTE["no_goal"], 1: PALETTE["goal"]}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "final_xg_lgbm.joblib")


@st.cache_data
def load_results():
    with open(REPORTS_DIR / "lgbm_results.json") as f:
        return json.load(f)


@st.cache_data
def load_predictions_and_split():
    raw_df = pd.read_parquet(DATA_PATH)
    mdl = joblib.load(MODELS_DIR / "final_xg_lgbm.joblib")
    X_all, _ = prepare_lgbm_data(raw_df)
    raw_df["model_xg"] = mdl.predict_proba(X_all)[:, 1]
    train_df, test_df = match_based_split(raw_df, test_size=0.20, random_state=42)
    return raw_df, train_df, test_df


# ---------------------------------------------------------------------------
# Sayfa ayarlari + tema
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="xG Analitik",
    page_icon="O",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
register_plotly_template()

df, train_df, test_df = load_predictions_and_split()
model = load_model()
results = load_results()

# ---------------------------------------------------------------------------
# Sidebar navigasyon
# ---------------------------------------------------------------------------

with st.sidebar:
    sidebar_brand('xG <span class="accent">Analitik</span>', "Beklenen Gol Modeli")

    page = option_menu(
        menu_title=None,
        options=["Genel Bakis", "Model Performansi", "Ozellik Analizi", "Oyuncu Kesfet", "Takim Analizi"],
        icons=["speedometer2", "graph-up", "bar-chart-steps", "person-badge", "shield-shaded"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": PALETTE["text_muted"], "font-size": "16px"},
            "nav-link": {
                "font-size": "14px",
                "font-weight": "600",
                "color": PALETTE["text_muted"],
                "padding": "10px 14px",
                "margin": "3px 0",
                "border-radius": "8px",
                "--hover-color": PALETTE["surface_alt"],
            },
            "nav-link-selected": {
                "background-color": PALETTE["surface_alt"],
                "color": PALETTE["primary"],
                "font-weight": "700",
            },
        },
    )

    st.markdown(
        f"""
        <div style="position: relative; margin-top: 2rem; padding: 0.8rem;
             background: {PALETTE['surface_alt']}; border-radius: 10px;
             border: 1px solid {PALETTE['border']};">
            <div style="font-size: 0.72rem; color: {PALETTE['text_muted']};
                 text-transform: uppercase; letter-spacing: 0.06em;">Model</div>
            <div style="font-size: 0.95rem; font-weight: 700; color: {PALETTE['text']};
                 margin-top: 2px;">LightGBM</div>
            <div style="font-size: 0.78rem; color: {PALETTE['primary']}; margin-top: 4px;">
                StatsBomb'u %2.3 geciyor</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# SAYFA 1: Genel Bakis
# ===========================================================================

if page == "Genel Bakis":
    hero(
        'Beklenen Gol <span class="accent">(xG)</span> Modeli',
        "StatsBomb acik verisinden 65.000+ sut uzerinde, her sutun gol olma olasiligini tahmin eden model.",
    )

    section_header("|", "Model Kalitesi")
    render_kpis([
        kpi_card("Log-loss", "0.2487", "StatsBomb'dan %2.3 iyi", PALETTE["primary"], "up"),
        kpi_card("Brier Skoru", "0.0711", "StatsBomb'dan %1.7 iyi", PALETTE["primary"], "up"),
        kpi_card("ROC-AUC", "0.8346", "Ayirt etme gucu", PALETTE["accent_blue"], "neutral"),
        kpi_card("Kalibrasyon (ECE)", "0.018", "Iyi kalibre", PALETTE["primary"], "up"),
    ])

    section_header("#", "Veri Seti")
    render_kpis([
        kpi_card("Toplam Sut", f"{len(df):,}", "Penaltilar haric", PALETTE["secondary"]),
        kpi_card("Gol Orani", f"{df['is_goal'].mean():.1%}", "Sinif dengesi", PALETTE["secondary"]),
        kpi_card("Mac Sayisi", f"{df['match_id'].nunique():,}", "Mac bazli bolme", PALETTE["secondary"]),
        kpi_card("Lig Sayisi", f"{df['competition'].nunique()}", "Erkek profesyonel futbol", PALETTE["secondary"]),
    ])

    section_header("=", "Model Karsilastirmasi")
    st.caption("Test seti, n=13.143 sut. Log-loss ve Brier icin dusuk olan daha iyi.")

    metrics = ["Log-loss", "Brier Skoru", "ROC-AUC"]
    lgbm_vals = [0.2487, 0.0711, 0.8346]
    lr_vals = [0.2489, 0.0712, 0.8349]
    sb_vals = [0.2546, 0.0724, 0.8245]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="LightGBM (Bizim)", x=metrics, y=lgbm_vals,
                         marker_color=PALETTE["primary"]))
    fig.add_trace(go.Bar(name="Temel LR", x=metrics, y=lr_vals,
                         marker_color=PALETTE["accent_blue"]))
    fig.add_trace(go.Bar(name="StatsBomb xG", x=metrics, y=sb_vals,
                         marker_color=PALETTE["secondary"]))
    fig.update_layout(barmode="group", height=380,
                      title="Uc Model Karsilastirmasi")
    st.plotly_chart(style_fig(fig), width="stretch")

    col1, col2 = st.columns([3, 2])
    with col1:
        section_header(">", "Liglere Gore Sutlar")
        comp_stats = df.groupby("competition").agg(
            Sut=("is_goal", "count"),
            Gol=("is_goal", "sum"),
            gol_orani=("is_goal", "mean"),
        ).sort_values("Sut", ascending=False).reset_index()
        comp_stats["Gol Orani"] = comp_stats["gol_orani"].round(4)
        comp_stats = comp_stats.drop(columns="gol_orani").rename(columns={"competition": "Lig"})
        st.dataframe(
            comp_stats,
            width="stretch",
            hide_index=True,
            height=380,
            column_config={
                "Lig": st.column_config.TextColumn("Lig", width="large"),
                "Sut": st.column_config.NumberColumn("Sut Sayisi", format="%d"),
                "Gol": st.column_config.NumberColumn("Gol", format="%d"),
                "Gol Orani": st.column_config.ProgressColumn(
                    "Gol Orani", min_value=0, max_value=0.25, format="%.1f%%",
                    help="Sutlarin yuzde kaci gol oldu"
                ),
            },
        )

    with col2:
        section_header("*", "Mesafeye Gore Gol Orani")
        df["_dist_bin"] = pd.cut(df["geom_distance"], bins=[0, 6, 11, 16, 22, 40],
                                 labels=["0-6m", "6-11m", "11-16m", "16-22m", "22m+"])
        dist_rate = df.groupby("_dist_bin", observed=True)["is_goal"].mean().reset_index()
        fig = px.bar(dist_rate, x="_dist_bin", y="is_goal",
                     labels={"_dist_bin": "Kaleye mesafe", "is_goal": "Gol orani"},
                     color="is_goal", color_continuous_scale=SEQ_PRIMARY)
        fig.update_layout(height=380, coloraxis_showscale=False,
                          title="Yakin sutlar daha cok gol olur")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_fig(fig), width="stretch")


# ===========================================================================
# SAYFA 2: Model Performansi
# ===========================================================================

elif page == "Model Performansi":
    hero('Model <span class="accent">Performansi</span>',
         "Kalibrasyon, olasilik dagilimi ve StatsBomb ile sut bazli uyum.")

    y_test = test_df["is_goal"].values
    model_xg_test = test_df["model_xg"].values
    sb_xg_test = test_df["statsbomb_xg"].values

    section_header("~", "Kalibrasyon Egrisi")
    st.caption("Model %20 dediginde, gercekten ~%20'si gol oluyor mu? Kosegene yakin = iyi.")
    n_bins = st.slider("Bin sayisi", 5, 20, 10)

    prob_true_m, prob_pred_m = calibration_curve(y_test, model_xg_test, n_bins=n_bins)
    prob_true_s, prob_pred_s = calibration_curve(y_test, sb_xg_test, n_bins=n_bins)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Mukemmel",
                             line=dict(color=PALETTE["text_muted"], width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=prob_pred_m, y=prob_true_m, mode="lines+markers",
                             name="LightGBM (Bizim)",
                             line=dict(color=PALETTE["primary"], width=3),
                             marker=dict(size=9)))
    fig.add_trace(go.Scatter(x=prob_pred_s, y=prob_true_s, mode="lines+markers",
                             name="StatsBomb xG",
                             line=dict(color=PALETTE["secondary"], width=3, dash="dash"),
                             marker=dict(size=9, symbol="square")))
    fig.update_layout(height=520, xaxis_title="Tahmin Edilen Olasilik",
                      yaxis_title="Gozlenen Siklik",
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    st.plotly_chart(style_fig(fig), width="stretch")

    col1, col2 = st.columns(2)

    with col1:
        section_header("|", "xG Dagilimi")
        goals = test_df[test_df["is_goal"] == 1]["model_xg"]
        no_goals = test_df[test_df["is_goal"] == 0]["model_xg"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=no_goals, nbinsx=50, name="Gol Degil",
                                   marker_color=PALETTE["no_goal"], opacity=0.75))
        fig.add_trace(go.Histogram(x=goals, nbinsx=50, name="Gol",
                                   marker_color=PALETTE["goal"], opacity=0.8))
        fig.update_layout(barmode="overlay", height=400,
                          xaxis_title="Model xG", yaxis_title="Sayi",
                          title="Tahmin Edilen xG: Goller vs Gol Olmayanlar")
        st.plotly_chart(style_fig(fig), width="stretch")

    with col2:
        section_header("=", "Desil Guvenilirligi")
        deciles = pd.qcut(sb_xg_test, q=10, labels=False, duplicates="drop")
        decile_df = pd.DataFrame({"decile": deciles, "model_xg": model_xg_test,
                                  "sb_xg": sb_xg_test, "actual": y_test.astype(float)})
        agg = decile_df.groupby("decile").agg(
            model_xg=("model_xg", "mean"), sb_xg=("sb_xg", "mean"),
            actual=("actual", "mean")).reset_index()
        labels = [f"D{i+1}" for i in range(len(agg))]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=agg["sb_xg"], name="StatsBomb xG",
                             marker_color=PALETTE["secondary"]))
        fig.add_trace(go.Bar(x=labels, y=agg["model_xg"], name="Model xG",
                             marker_color=PALETTE["primary"]))
        fig.add_trace(go.Scatter(x=labels, y=agg["actual"], name="Gercek Oran",
                                 mode="lines+markers",
                                 line=dict(color=PALETTE["accent_blue"], width=3)))
        fig.update_layout(barmode="group", height=400,
                          xaxis_title="xG Desili", yaxis_title="Olasilik",
                          title="Desil Bazinda Tahmin vs Gercek")
        st.plotly_chart(style_fig(fig), width="stretch")

    section_header("o", "Sut Bazli Uyum")
    st.caption("Her nokta bir sut. Kosegene yakin = modelimiz ve StatsBomb hemfikir (r=0.881).")
    sample_size = st.slider("Ornek boyutu", 1000, len(test_df), min(5000, len(test_df)), step=500)
    sample = test_df.sample(n=sample_size, random_state=42)
    sample = sample.assign(Sonuc=sample["is_goal"].map({0: "Gol Degil", 1: "Gol"}))

    fig = px.scatter(sample, x="statsbomb_xg", y="model_xg", color="Sonuc",
                     color_discrete_map={"Gol Degil": PALETTE["no_goal"], "Gol": PALETTE["goal"]},
                     opacity=0.45,
                     labels={"statsbomb_xg": "StatsBomb xG", "model_xg": "Model xG"})
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Tam uyum",
                             line=dict(color=PALETTE["secondary"], width=1.5, dash="dash")))
    fig.update_layout(height=520, title=f"Model xG vs StatsBomb xG (n={sample_size:,})")
    st.plotly_chart(style_fig(fig), width="stretch")


# ===========================================================================
# SAYFA 3: Ozellik Analizi
# ===========================================================================

elif page == "Ozellik Analizi":
    hero('Ozellik <span class="accent">Analizi</span>',
         "Modelin tahminlerini en cok ne etkiliyor? SHAP degerleri ile olcuyoruz.")

    shap_data = results["top_15_shap"]

    feature_desc = {
        "ff_dist_to_gk": "Kaleciye olan mesafe",
        "geom_angle_rad": "Kale direklerinin olusturdugu aci (radyan)",
        "ff_n_opponents_in_cone": "Sut konisindeki rakip sayisi",
        "ff_dist_nearest_opponent": "En yakin rakibe mesafe",
        "shot_body_part": "Vurusun yapildigi vucut bolumu",
        "shot_technique": "Sut teknigi (vole, yari vole vb.)",
        "geom_distance": "Kale merkezine mesafe",
        "ff_gk_off_line": "Kalecinin cizgiden ne kadar ciktigi",
        "score_diff": "Sut anindaki skor farki",
        "geom_x_dist": "Kale cizgisine yatay mesafe",
        "geom_angle_deg": "Kaleye aci (derece)",
        "geom_y_dist": "Kale merkezine yanal mesafe",
        "shot_deflected": "Sutun sekme sonrasi olup olmadigi",
        "kp_pass_length": "Asist pasinin uzunlugu",
        "kp_pass_angle": "Asist pasinin acisi",
    }

    col1, col2 = st.columns([3, 2])
    with col1:
        section_header("=", "SHAP Onem Sirasi (Ilk 15)")
        shap_df = pd.DataFrame(shap_data).sort_values("mean_abs_shap", ascending=True)
        shap_df["etiket"] = shap_df["feature"].map(lambda f: feature_desc.get(f, f))
        fig = px.bar(shap_df, x="mean_abs_shap", y="etiket", orientation="h",
                     color="mean_abs_shap", color_continuous_scale=SEQ_PRIMARY,
                     labels={"mean_abs_shap": "Ortalama |SHAP|", "etiket": ""})
        fig.update_layout(height=560, coloraxis_showscale=False,
                          title="Tahmine En Cok Katki Yapan Ozellikler")
        st.plotly_chart(style_fig(fig), width="stretch")

    with col2:
        section_header("i", "Ilk 4 Ozellik Ne Anlatiyor")
        st.markdown(f"""
        <div style="background:{PALETTE['surface']}; border:1px solid {PALETTE['border']};
             border-radius:12px; padding:1.2rem; line-height:1.7;">
        <b style="color:{PALETTE['primary']}">1. Kaleciye mesafe</b> -- kaleci ile sutcu
        arasindaki uzaklik. Yakin = zor kurtaris.<br><br>
        <b style="color:{PALETTE['primary']}">2. Sut acisi</b> -- genis aci = daha buyuk
        kale hedefi gorunuyor.<br><br>
        <b style="color:{PALETTE['primary']}">3. Konideki rakipler</b> -- sut yolunu
        kapatan savunmaci sayisi.<br><br>
        <b style="color:{PALETTE['primary']}">4. En yakin rakip</b> -- sutcunun uzerindeki
        baski.<br><br>
        <span style="color:{PALETTE['text_muted']}">Geometri ve baski baskin -- alan bilgisi
        tam olarak bunu ongordugu icin model dogru seyleri ogrenmis.</span>
        </div>
        """, unsafe_allow_html=True)

    section_header("o", "Tek Ozellik Kesfet")
    st.caption("Bir ozellik secin, xG ile nasil iliskilendigini gorun.")
    numeric_features = ["geom_distance", "geom_angle_deg", "ff_dist_to_gk",
                        "ff_n_opponents_in_cone", "ff_dist_nearest_opponent",
                        "ff_gk_off_line", "score_diff", "kp_pass_length"]
    selected_feature = st.selectbox("Ozellik", numeric_features,
                                    format_func=lambda f: f"{f}  --  {feature_desc.get(f, '')}")

    col1, col2 = st.columns(2)
    sample = df.sample(n=min(8000, len(df)), random_state=42)
    sample = sample.assign(Sonuc=sample["is_goal"].map({0: "Gol Degil", 1: "Gol"}))

    with col1:
        fig = px.scatter(sample, x=selected_feature, y="model_xg", color="Sonuc",
                         color_discrete_map={"Gol Degil": PALETTE["no_goal"], "Gol": PALETTE["goal"]},
                         opacity=0.35, labels={"model_xg": "Model xG"})
        fig.update_layout(height=420, title=f"{feature_desc.get(selected_feature, selected_feature)} vs Model xG")
        st.plotly_chart(style_fig(fig), width="stretch")

    with col2:
        col_data = df[selected_feature].dropna()
        if len(col_data) > 100:
            bins = pd.qcut(col_data, q=10, duplicates="drop")
            binned = df.groupby(bins, observed=True).agg(
                mean_xg=("model_xg", "mean"), goal_rate=("is_goal", "mean")).reset_index()
            binned["bin_label"] = [f"Q{i+1}" for i in range(len(binned))]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=binned["bin_label"], y=binned["mean_xg"],
                                 name="Ortalama Model xG", marker_color=PALETTE["primary"]))
            fig.add_trace(go.Scatter(x=binned["bin_label"], y=binned["goal_rate"],
                                     name="Gercek Gol Orani", mode="lines+markers",
                                     line=dict(color=PALETTE["secondary"], width=3)))
            fig.update_layout(height=420, xaxis_title=f"{selected_feature} (dusukten yuksege)",
                              yaxis_title="Olasilik",
                              title="Model xG gercegi takip ediyor")
            st.plotly_chart(style_fig(fig), width="stretch")


# ===========================================================================
# SAYFA 4: Oyuncu Kesfet
# ===========================================================================

elif page == "Oyuncu Kesfet":
    hero('Oyuncu <span class="accent">Kesfet</span>',
         "Beklenen golun uzerinde (veya altinda) atan oyuncular -- bitiricilik sinyali.")

    col1, col2, col3 = st.columns(3)
    with col1:
        min_shots = st.slider("Minimum sut sayisi", 10, 100, 20, step=5)
    with col2:
        top_n = st.slider("Gosterilecek oyuncu (N)", 5, 30, 12)
    with col3:
        competitions = ["Tumu"] + sorted(df["competition"].unique().tolist())
        selected_comp = st.selectbox("Lig", competitions)

    filtered = df.copy()
    if selected_comp != "Tumu":
        filtered = filtered[filtered["competition"] == selected_comp]

    player_df = build_player_table(filtered, min_shots=min_shots)

    if len(player_df) > 0:
        best = player_df.iloc[0]
        worst = player_df.iloc[-1]
        render_kpis([
            kpi_card("Gecen Oyuncu", f"{len(player_df)}", f"Min {min_shots} sut", PALETTE["accent_blue"]),
            kpi_card("En Iyi Bitirici",
                     best["player"].encode("ascii", "replace").decode()[:18],
                     f"+{best['goals_above_xg']:.1f} gol (xG'nin uzerinde)", PALETTE["primary"], "up"),
            kpi_card("En Verimsiz",
                     worst["player"].encode("ascii", "replace").decode()[:18],
                     f"{worst['goals_above_xg']:.1f} gol (xG'nin altinda)", PALETTE["negative"], "down"),
        ])

    col1, col2 = st.columns(2)
    with col1:
        section_header("+", f"En Iyi {top_n} (Beklenenin Uzerinde)")
        top = player_df.head(top_n).sort_values("goals_above_xg")
        fig = px.bar(top, x="goals_above_xg", y="player", orientation="h",
                     error_x="goals_above_xg_se",
                     color="goals_above_xg", color_continuous_scale=SEQ_PRIMARY,
                     labels={"goals_above_xg": "xG Ustu Gol", "player": ""})
        fig.update_layout(height=max(400, top_n * 30), coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig), width="stretch")

    with col2:
        section_header("-", f"En Kotu {top_n} (Beklenenin Altinda)")
        bottom = player_df.tail(top_n).sort_values("goals_above_xg")
        fig = px.bar(bottom, x="goals_above_xg", y="player", orientation="h",
                     error_x="goals_above_xg_se",
                     color="goals_above_xg",
                     color_continuous_scale=[[0, PALETTE["negative"]], [1, "#7A2A3A"]],
                     labels={"goals_above_xg": "xG Ustu Gol", "player": ""})
        fig.update_layout(height=max(400, top_n * 30), coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig), width="stretch")

    section_header("#", "Tam Oyuncu Tablosu")
    search = st.text_input("Oyuncu ara", placeholder="orn. Messi")
    display_df = player_df.copy()
    if search:
        display_df = display_df[display_df["player"].str.contains(search, case=False, na=False)]
    for c in ["xg_sum", "sb_xg_sum", "xg_per_shot", "actual_conversion",
              "goals_above_xg", "goals_above_xg_se"]:
        if c in display_df.columns:
            display_df[c] = display_df[c].round(2)
    display_df = display_df.rename(columns={
        "rank": "Sira",
        "player": "Oyuncu",
        "shots": "Sut",
        "actual_goals": "Gol",
        "xg_sum": "Toplam xG",
        "sb_xg_sum": "StatsBomb xG",
        "xg_per_shot": "Sut Basi xG",
        "actual_conversion": "Gercek Donusum",
        "goals_above_xg": "Gol - xG Farki",
        "goals_above_xg_se": "Guven Araligi",
    })
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        height=440,
        column_config={
            "Sira": st.column_config.NumberColumn("Sira", width="small", format="%d"),
            "Oyuncu": st.column_config.TextColumn("Oyuncu", width="large"),
            "Sut": st.column_config.NumberColumn("Sut", format="%d", help="Toplam sut sayisi"),
            "Gol": st.column_config.NumberColumn("Gol", format="%d", help="Atilan gol sayisi"),
            "Toplam xG": st.column_config.NumberColumn("Toplam xG", format="%.2f",
                help="Modelin bu oyuncuya verdigi toplam beklenen gol"),
            "StatsBomb xG": st.column_config.NumberColumn("StatsBomb xG", format="%.2f"),
            "Sut Basi xG": st.column_config.ProgressColumn(
                "Sut Basi xG", min_value=0, max_value=0.5, format="%.3f",
                help="Her sutun ortalama gol olasiligi"
            ),
            "Gercek Donusum": st.column_config.ProgressColumn(
                "Donusum Orani", min_value=0, max_value=0.5, format="%.3f",
                help="Gercekte gol donusum orani"
            ),
            "Gol - xG Farki": st.column_config.NumberColumn(
                "Gol - xG", format="%.2f",
                help="Pozitif = beklenenin ustunde gol atan (klinik bitirici)"
            ),
            "Guven Araligi": st.column_config.NumberColumn("Guven (+/-)", format="%.2f",
                help="Istatistiksel guven araligi -- kucuk ornek = yuksek belirsizlik"),
        },
    )

    section_header("o", "Oyuncu Detay")
    selected_player = st.selectbox("Oyuncu sec", player_df["player"].tolist())
    if selected_player:
        p_shots = filtered[filtered["player"] == selected_player].copy()
        p_row = player_df[player_df["player"] == selected_player].iloc[0]
        render_kpis([
            kpi_card("Sut", f"{int(p_row['shots'])}", accent=PALETTE["accent_blue"]),
            kpi_card("Gol", f"{int(p_row['actual_goals'])}", accent=PALETTE["primary"]),
            kpi_card("xG Toplam", f"{p_row['xg_sum']:.1f}", accent=PALETTE["secondary"]),
            kpi_card("Gol - xG", f"{p_row['goals_above_xg']:+.1f}",
                     "Bitiricilik sinyali",
                     PALETTE["primary"] if p_row["goals_above_xg"] >= 0 else PALETTE["negative"],
                     "up" if p_row["goals_above_xg"] >= 0 else "down"),
        ])

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(p_shots, x="model_xg", nbins=20,
                               color=p_shots["is_goal"].map({0: "Gol Degil", 1: "Gol"}),
                               color_discrete_map={"Gol Degil": PALETTE["no_goal"], "Gol": PALETTE["goal"]},
                               labels={"model_xg": "Sut xG", "color": "Sonuc"})
            fig.update_layout(barmode="overlay", height=380,
                              title=f"{selected_player[:30]} -- Sut xG Dagilimi")
            st.plotly_chart(style_fig(fig), width="stretch")
        with col2:
            shot_map = p_shots.assign(Sonuc=p_shots["is_goal"].map({0: "Gol Degil", 1: "Gol"}))
            fig = px.scatter(shot_map, x="x", y="y", color="model_xg",
                             symbol="Sonuc",
                             symbol_map={"Gol Degil": "circle", "Gol": "star"},
                             color_continuous_scale=SEQ_PRIMARY,
                             labels={"x": "Saha X", "y": "Saha Y", "model_xg": "xG"})
            fig.update_layout(height=380, title=f"{selected_player[:30]} -- Sut Haritasi")
            fig.update_yaxes(range=[0, 80])
            fig.update_xaxes(range=[60, 120])
            st.plotly_chart(style_fig(fig), width="stretch")


# ===========================================================================
# SAYFA 5: Takim Analizi
# ===========================================================================

elif page == "Takim Analizi":
    hero('Takim <span class="accent">Analizi</span>',
         "Hucum ve savunmada beklenen gol performansinin uzerinde ve altinda kalan takimlar.")

    min_team_shots = st.slider("Minimum sut (hucum + savunma)", 30, 200, 50, step=10)
    team_df = build_team_table(df, min_shots=min_team_shots)

    if len(team_df) > 0:
        best_atk = team_df.iloc[0]
        render_kpis([
            kpi_card("Gecen Takim", f"{len(team_df)}", f"Min {min_team_shots} sut", PALETTE["accent_blue"]),
            kpi_card("En Iyi Hucum",
                     best_atk["team"].encode("ascii", "replace").decode()[:18],
                     f"+{best_atk['attacking_over_under']:.1f} gol (xG ustu)", PALETTE["primary"], "up"),
            kpi_card("Dogrulama r", "0.9956", "Takim xG vs gercek goller", PALETTE["primary"], "up"),
        ])

    section_header("o", "Hucum: xG vs Gercek Goller")
    st.caption("Kosegen ustu = beklenenin ustunde atiyor. Daire boyutu = sut sayisi.")
    fig = px.scatter(team_df, x="xg_for", y="goals_for", hover_name="team",
                     size="shots_for", color="attacking_over_under",
                     color_continuous_scale=DIVERGING, color_continuous_midpoint=0,
                     labels={"xg_for": "Beklenen Gol (xG)", "goals_for": "Gercek Gol",
                             "attacking_over_under": "Fark", "shots_for": "Sut"})
    mx = max(team_df["xg_for"].max(), team_df["goals_for"].max()) + 10
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="Beklenen (y=x)",
                             line=dict(color=PALETTE["text_muted"], width=1, dash="dash")))
    fig.update_layout(height=560)
    st.plotly_chart(style_fig(fig), width="stretch")

    section_header("=", "Performans Kadranlari")
    st.caption("Sag ust: klinik hucum VE saglam savunma. Sol alt: her iki ucta da zorlanma.")
    fig = px.scatter(team_df, x="attacking_over_under", y="defensive_over_under",
                     hover_name="team", size="shots_for", color="net_goals",
                     color_continuous_scale=DIVERGING, color_continuous_midpoint=0,
                     labels={"attacking_over_under": "Hucum Farki (gol - xG)",
                             "defensive_over_under": "Savunma Farki (xG - yenilen gol)",
                             "net_goals": "Net Gol"})
    fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["border"])
    fig.add_vline(x=0, line_dash="dash", line_color=PALETTE["border"])
    fig.update_layout(height=560)
    st.plotly_chart(style_fig(fig), width="stretch")

    section_header("#", "Tam Takim Tablosu")
    search_team = st.text_input("Takim ara", placeholder="orn. Barcelona")
    display_team = team_df.copy()
    if search_team:
        display_team = display_team[display_team["team"].str.contains(search_team, case=False, na=False)]
    for c in ["xg_for", "attacking_over_under", "xg_against", "defensive_over_under", "net_xg"]:
        if c in display_team.columns:
            display_team[c] = display_team[c].round(1)
    add_logo_column(display_team, team_col="team")
    display_team = display_team.rename(columns={
        "rank": "Sira",
        "team": "Takim",
        "shots_for": "Atilan Sut",
        "goals_for": "Atilan Gol",
        "xg_for": "Hucum xG",
        "attacking_over_under": "Hucum Farki",
        "shots_against": "Yenilen Sut",
        "goals_against": "Yenilen Gol",
        "xg_against": "Savunma xG",
        "defensive_over_under": "Savunma Farki",
        "net_xg": "Net xG",
        "net_goals": "Net Gol",
    })
    st.dataframe(
        display_team,
        width="stretch",
        hide_index=True,
        height=500,
        column_config={
            "Logo": st.column_config.ImageColumn("", width="small",
                help="Takim amblemi"),
            "Sira": st.column_config.NumberColumn("Sira", width="small", format="%d"),
            "Takim": st.column_config.TextColumn("Takim", width="large"),
            "Atilan Sut": st.column_config.NumberColumn("Atilan Sut", format="%d"),
            "Atilan Gol": st.column_config.NumberColumn("Atilan Gol", format="%d"),
            "Hucum xG": st.column_config.NumberColumn("Hucum xG", format="%.1f",
                help="Modelin bu takima verdigi toplam beklenen gol"),
            "Hucum Farki": st.column_config.NumberColumn("Hucum Farki", format="%.1f",
                help="Gercek Gol - Hucum xG. Pozitif = klinik hucum"),
            "Yenilen Sut": st.column_config.NumberColumn("Yenilen Sut", format="%d"),
            "Yenilen Gol": st.column_config.NumberColumn("Yenilen Gol", format="%d"),
            "Savunma xG": st.column_config.NumberColumn("Savunma xG", format="%.1f",
                help="Rakibin beklenen gol toplami (ne kadar tehdit altinda)"),
            "Savunma Farki": st.column_config.NumberColumn("Savunma Farki", format="%.1f",
                help="Savunma xG - Yenilen Gol. Pozitif = saglam savunma"),
            "Net xG": st.column_config.NumberColumn("Net xG", format="%.1f",
                help="Hucum xG - Savunma xG"),
            "Net Gol": st.column_config.NumberColumn("Net Gol", format="%d",
                help="Atilan Gol - Yenilen Gol"),
        },
    )
