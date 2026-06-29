"""
Dashboard theme system: color palette, CSS injection, Plotly template,
and reusable UI component helpers (KPI cards, section headers).

Centralizes all visual identity so every page shares one design language.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


# ---------------------------------------------------------------------------
# Color palette (football-analytics, dark theme)
# ---------------------------------------------------------------------------

PALETTE = {
    "bg": "#0E1117",
    "surface": "#1A1F2B",
    "surface_alt": "#222836",
    "border": "#2A3140",
    "primary": "#00D4A0",       # pitch turquoise -- model / positive
    "secondary": "#FF6B35",     # orange -- benchmark / highlight
    "accent_blue": "#4DA3FF",   # statsbomb / neutral series
    "goal": "#00D4A0",          # goal events
    "no_goal": "#5B6478",       # non-goal events
    "positive": "#00D4A0",      # overperform
    "negative": "#FF5470",      # underperform
    "text": "#E6E8EB",
    "text_muted": "#9CA3AF",
    "grid": "#252B38",
}

# Sequential scales tuned for the dark theme
SEQ_PRIMARY = [
    [0.0, "#0E2A24"],
    [0.5, "#0A8F70"],
    [1.0, "#00D4A0"],
]

DIVERGING = [
    [0.0, "#FF5470"],
    [0.5, "#2A3140"],
    [1.0, "#00D4A0"],
]


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

def inject_css() -> None:
    """Inject custom CSS for a professional look."""
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* App background */
    .stApp {{
        background: {PALETTE['bg']};
    }}

    /* Hide default Streamlit chrome */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{
        background: transparent;
        height: 0;
    }}

    /* Tighten main container top padding */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {PALETTE['surface']};
        border-right: 1px solid {PALETTE['border']};
    }}

    /* Headings */
    h1, h2, h3, h4 {{
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: {PALETTE['text']};
    }}

    /* KPI card */
    .kpi-card {{
        background: {PALETTE['surface']};
        border: 1px solid {PALETTE['border']};
        border-left: 4px solid var(--accent, {PALETTE['primary']});
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        height: 100%;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-2px);
        border-color: var(--accent, {PALETTE['primary']});
    }}
    .kpi-label {{
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {PALETTE['text_muted']};
        margin-bottom: 0.35rem;
    }}
    .kpi-value {{
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.1;
        color: {PALETTE['text']};
    }}
    .kpi-delta {{
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }}
    .kpi-delta.up {{ color: {PALETTE['primary']}; }}
    .kpi-delta.down {{ color: {PALETTE['negative']}; }}
    .kpi-delta.neutral {{ color: {PALETTE['text_muted']}; }}

    /* Section header */
    .section-header {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 1.6rem 0 0.4rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid {PALETTE['border']};
    }}
    .section-header .sh-icon {{
        font-size: 1.15rem;
    }}
    .section-header .sh-title {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {PALETTE['text']};
        letter-spacing: -0.01em;
    }}

    /* Hero header */
    .hero {{
        background: linear-gradient(135deg, {PALETTE['surface']} 0%, {PALETTE['surface_alt']} 100%);
        border: 1px solid {PALETTE['border']};
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.4rem;
    }}
    .hero-title {{
        font-size: 2.0rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: {PALETTE['text']};
        margin: 0;
    }}
    .hero-title .accent {{ color: {PALETTE['primary']}; }}
    .hero-sub {{
        font-size: 1.0rem;
        color: {PALETTE['text_muted']};
        margin-top: 0.4rem;
        font-weight: 400;
    }}

    /* Sidebar brand */
    .brand {{
        padding: 0.5rem 0.2rem 1.2rem 0.2rem;
        border-bottom: 1px solid {PALETTE['border']};
        margin-bottom: 0.8rem;
    }}
    .brand-title {{
        font-size: 1.3rem;
        font-weight: 800;
        color: {PALETTE['text']};
        letter-spacing: -0.02em;
    }}
    .brand-title .accent {{ color: {PALETTE['primary']}; }}
    .brand-sub {{
        font-size: 0.72rem;
        color: {PALETTE['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.15rem;
    }}

    /* Dataframe rounding */
    [data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid {PALETTE['border']};
    }}

    /* Tabs */
    button[data-baseweb="tab"] {{
        font-weight: 600;
    }}

    /* Slider / widget labels */
    .stSlider label, .stSelectbox label, .stTextInput label, .stRadio label {{
        font-weight: 600;
        color: {PALETTE['text']};
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly template
# ---------------------------------------------------------------------------

def register_plotly_template() -> str:
    """Register and return the name of the shared Plotly template."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family="Inter, sans-serif", color=PALETTE["text"], size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[
            PALETTE["primary"], PALETTE["secondary"], PALETTE["accent_blue"],
            "#B388FF", "#FFD166", "#06D6A0",
        ],
        xaxis=dict(
            gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["border"],
            linecolor=PALETTE["border"], tickfont=dict(color=PALETTE["text_muted"]),
            title_font=dict(color=PALETTE["text_muted"], size=12),
        ),
        yaxis=dict(
            gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["border"],
            linecolor=PALETTE["border"], tickfont=dict(color=PALETTE["text_muted"]),
            title_font=dict(color=PALETTE["text_muted"], size=12),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=PALETTE["text"], size=12),
            bordercolor=PALETTE["border"],
        ),
        margin=dict(l=50, r=30, t=50, b=50),
        hoverlabel=dict(
            bgcolor=PALETTE["surface_alt"],
            font=dict(family="Inter, sans-serif", color=PALETTE["text"]),
            bordercolor=PALETTE["border"],
        ),
        title=dict(font=dict(size=15, color=PALETTE["text"])),
    )
    pio.templates["xg_dark"] = template
    return "xg_dark"


def style_fig(fig: go.Figure, height: int | None = None) -> go.Figure:
    """Apply the shared template and common layout to a figure."""
    fig.update_layout(template="xg_dark")
    if height is not None:
        fig.update_layout(height=height)
    return fig


# ---------------------------------------------------------------------------
# UI component helpers
# ---------------------------------------------------------------------------

def kpi_card(label: str, value: str, delta: str | None = None,
             accent: str | None = None, direction: str = "neutral") -> str:
    """Return HTML for a KPI card.

    Args:
        label: Small uppercase label.
        value: Main metric value.
        delta: Optional sub-line (e.g. comparison).
        accent: Left-border color (hex). Defaults to primary.
        direction: 'up' | 'down' | 'neutral' -- colors the delta.
    """
    accent = accent or PALETTE["primary"]
    delta_html = ""
    if delta:
        delta_html = f'<div class="kpi-delta {direction}">{delta}</div>'
    return f"""
    <div class="kpi-card" style="--accent: {accent};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


def render_kpis(cards: list[str]) -> None:
    """Render a row of KPI cards (list of kpi_card HTML strings)."""
    cols = st.columns(len(cards))
    for col, html in zip(cols, cards):
        with col:
            st.markdown(html, unsafe_allow_html=True)


def section_header(icon: str, title: str) -> None:
    """Render a consistent section header with icon + title + divider."""
    st.markdown(
        f"""
        <div class="section-header">
            <span class="sh-icon">{icon}</span>
            <span class="sh-title">{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(title_html: str, subtitle: str) -> None:
    """Render a hero header block."""
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">{title_html}</div>
            <div class="hero-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_brand(title_html: str, subtitle: str) -> None:
    """Render the sidebar brand block."""
    st.markdown(
        f"""
        <div class="brand">
            <div class="brand-title">{title_html}</div>
            <div class="brand-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
