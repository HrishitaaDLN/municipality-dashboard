"""
Municipal Fiscal & Sustainability Dashboard
Midwest Municipalities · Fiscal Typology · Climate Actions

Setup and data files: see README.md in this folder.

Run:
    pip install -r requirements.txt
    streamlit run dashboard.py
"""

from __future__ import annotations

import html
import io
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the very first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Municipal Sustainability Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Financial indicator columns (order matters for charts) ────────────────────
FIN_COLS: list[str] = [
    "Cash & Investment Coverage",
    "Financial Slack Ratio",
    "Long-Term Leverage",
    "Long-Term Debt per Capita(per resident)",
    "Debt Service Burden",
    "Pension Exposure Ratio",
    "OPEB Liability-Scaled",
    "OPEB Expense-Scaled",
    "Capital Intensity",
    "Infrastructure Burden per Capita(per resident)",
    "Net Investment Capacity",
    "Restricted Rigidity",
    "Liability Pressure",
    "Asset Coverage",
]
# Indicators where higher raw value = worse fiscal health → flip sign before PCA
FLIP_COLS: frozenset[str] = frozenset([
    "Long-Term Leverage",
    "Long-Term Debt per Capita(per resident)",
    "Debt Service Burden",
    "OPEB Liability-Scaled",
    "OPEB Expense-Scaled",
    "Capital Intensity",
    "Infrastructure Burden per Capita(per resident)",
    "Restricted Rigidity",
    "Liability Pressure",
])

# ── Sustainability score columns ──────────────────────────────────────────────
SUS_COL   = "Total\n(/48)"
SUS_SUBS  = [
    ("Governance\n(/9)",        "Governance",        9),
    ("Data & Analytics\n(/21)", "Data & Analytics",  21),
    ("Action Planning\n(/18)",  "Action Planning",   18),
]

# ── Climate network memberships ───────────────────────────────────────────────
NET_COLS: dict[str, str] = {
    "grc":            "GRC",
    "c4":             "C4",
    "iclei":          "ICLEI",
    "ev_ready":       "EV Ready",
    "uscm_mcpa":      "USCM/MCPA",
    "climate_mayors": "Climate Mayors",
}

# ── Commission authority levels ───────────────────────────────────────────────
COMM_LBL: dict[int, str] = {
    1: "Advisory only",
    2: "Advisory + recommendations",
    3: "Decision authority",
    4: "Full authority",
}

# ── 2×2 typology quadrants ────────────────────────────────────────────────────
QUAD_META: dict[str, tuple[str, str, str]] = {
    # key → (display name, css-class suffix, hex colour)
    "Low pension burden / High liquidity":  ("Q1 · Resilient",  "q1", "#34d399"),
    "Low pension burden / Low liquidity":   ("Q2 · Stable",     "q2", "#60a5fa"),
    "High pension burden / High liquidity": ("Q3 · Pressured",  "q3", "#fbbf24"),
    "High pension burden / Low liquidity":  ("Q4 · Vulnerable", "q4", "#f87171"),
}

# ── Action sectors & colours ──────────────────────────────────────────────────
SEC_COL: dict[str, str] = {
    "Energy":      "#fbbf24",
    "Transport":   "#60a5fa",
    "Waste":       "#34d399",
    "Water":       "#22d3ee",
    "Buildings":   "#f87171",
    "Land Use":    "#a78bfa",
    "Air Quality": "#fb923c",
    "Community":   "#f472b6",
    "Governance":  "#94a3b8",
    "Other":       "#475569",
}
FOCUS_SECTORS: list[str] = ["Energy", "Transport", "Waste"]
ACTION_PAGE_SIZE: int = 12

# ── Fallback data path (when no file is uploaded) ─────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DISK_FISCAL  = BASE_DIR / "midwest data (1).xlsx"
DISK_ACTIONS = BASE_DIR / "municipality_actions.xlsx"

# ── Friendly axis/metric labels for municipality audiences ────────────────────
FRIENDLY: dict[str, str] = {
    "PC1_pension_axis": "Pension Health Score",
    "liquidity_axis":   "Cash Liquidity Score",
    "fiscal_score":     "Overall Fiscal Health",
    "pca_2x2_type":     "Fiscal Cluster",
}

# ── Plain-language: how `fiscal_health` is built (see load_fiscal) ────────────
FISCAL_HEALTH_EXPLAINER_MD = """
**What this number is**  
The **Overall Fiscal Health** value is a *peer index*: it summarizes how this city compares to **other cities in the same spreadsheet** across the financial columns we have (not a dollar total and not a bond rating).

**What inputs we use (everyday wording)**  
Typical fields include **cash and investment coverage**, **financial slack**, **debt levels and debt service**, **pension and OPEB exposure**, **capital intensity**, **infrastructure burden per resident**, **restricted-fund rigidity**, **liability pressure**, **asset coverage**, and similar balance-sheet and budget-stress measures—whatever columns are present in your uploaded Midwest file.

**How it is calculated**  
1. Missing values are filled with the **column median** so one blank field does not drop a city.  
2. For indicators where a *higher raw number means worse stress*, we **flip the sign** so that, everywhere, **larger = financially stronger** on that line item.  
3. Each column is **standardized** (compared to the dataset mean and spread) so large and small cities are judged on the same scale.  
4. **Overall Fiscal Health** is the **average** of those standardized values for that city.

**How to read the score**  
- **Near 0** — close to the **typical** city in this sample.  
- **Positive** — **stronger than average** on balance across the combined indicators.  
- **Negative** — **weaker than average** on balance.  

Because it is an average of standardized columns, most cities land in a **narrow band** around zero; small gaps (for example 0.29 vs 0.32) mean “very similar overall financial position in this dataset,” which is why we use it to pick fair sustainability peers.
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
# CSS / THEME  — FIXED HIGH-CONTRAST VERSION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Base ──────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: #ffffff !important;
}

.stApp  {
    background: #080c18;
    color: #f8fafc;
}

/* ── Sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0c1020;
    border-right: 1px solid #1e293b;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* ── Typography ───────────────────────────────────────── */
h1 {
    font-family: 'Playfair Display', serif !important;
    color: #ffffff !important;
    font-size: 1.8rem !important;
    letter-spacing: -.3px;
    line-height: 1.2;
}

h2 {
    font-family: 'Playfair Display', serif !important;
    color: #f8fafc !important;
    font-size: 1.15rem !important;
}

h3 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
    font-size: .72rem !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 16px 0 5px !important;
}

p, li, label, span, div {
    color: #f1f5f9;
    line-height: 1.6;
}

/* ── KPI card ─────────────────────────────────────────── */
.kpi {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}

.kpi-val {
    font-size: 1.6rem;
    font-weight: 700;
    color: #60a5fa;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1;
}

.kpi-lbl {
    font-size: .7rem;
    color: #f1f5f9;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 5px;
}

/* ── City comparison card ─────────────────────────────── */
.ccard {
    border-radius: 12px;
    padding: 18px 20px;
    border: 1px solid #1e293b;
}

.cc-q1 {
    background: linear-gradient(135deg,rgba(52,211,153,.12),transparent);
    border-top: 3px solid #34d399;
}

.cc-q2 {
    background: linear-gradient(135deg,rgba(96,165,250,.12),transparent);
    border-top: 3px solid #60a5fa;
}

.cc-q3 {
    background: linear-gradient(135deg,rgba(251,191,36,.12),transparent);
    border-top: 3px solid #fbbf24;
}

.cc-q4 {
    background: linear-gradient(135deg,rgba(248,113,113,.12),transparent);
    border-top: 3px solid #f87171;
}

.cname  {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    margin-bottom: 2px;
    color: #ffffff;
}

.cstate {
    font-size: .7rem;
    color: #cbd5e1;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Cluster badge ────────────────────────────────────── */
.badge {
    display: inline-block;
    font-size: .65rem;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    letter-spacing: .5px;
    margin-top: 6px;
}

.bq1 {
    background: rgba(52,211,153,.2);
    color: #34d399;
}

.bq2 {
    background: rgba(96,165,250,.2);
    color: #60a5fa;
}

.bq3 {
    background: rgba(251,191,36,.2);
    color: #fbbf24;
}

.bq4 {
    background: rgba(248,113,113,.2);
    color: #f87171;
}

/* ── Metric row ───────────────────────────────────────── */
.mrow {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #1e293b;
    font-size: .82rem;
}

.mrow:last-child {
    border-bottom: none;
}

.ml {
    color: #f8fafc;
    font-weight: 500;
}

.mv {
    color: #ffffff;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .76rem;
}

/* ── Network membership pill ──────────────────────────── */
.npill {
    display: inline-block;
    font-size: .62rem;
    padding: 3px 10px;
    border-radius: 12px;
    margin: 2px;
    font-weight: 700;
}

.non {
    background: rgba(96,165,250,.18);
    color: #bfdbfe;
    border: 1px solid rgba(96,165,250,.3);
}

.noff {
    background: rgba(255,255,255,.05);
    color: #94a3b8;
    border: 1px solid #334155;
}

/* ── Section label ────────────────────────────────────── */
.sec-lbl {
    font-size: .6rem;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: #ffffff;
    margin: 14px 0 5px;
    font-weight: 700;
}

/* ── Progress bar ─────────────────────────────────────── */
.pbar {
    height: 4px;
    background: #1e293b;
    border-radius: 2px;
    margin-top: 3px;
}

.pbar-fill {
    height: 100%;
    border-radius: 2px;
    opacity: .95;
}

/* ── Action card ──────────────────────────────────────── */
.acard {
    background: #0f172a;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    border-left: 3px solid #60a5fa;
}

.acard-name {
    font-size: .8rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 4px;
}

.acard-text {
    font-size: .74rem;
    color: #e2e8f0;
    line-height: 1.55;
}

.acard-tag  {
    display: inline-block;
    font-size: .58rem;
    padding: 2px 8px;
    border-radius: 10px;
    margin-top: 6px;
    font-weight: 700;
}

/* ── Info banner ──────────────────────────────────────── */
.info-banner {
    background: #0f172a;
    border: 1px solid #334155;
    border-left: 3px solid #60a5fa;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: .82rem;
    color: #f1f5f9;
    margin: 8px 0 14px;
}

/* ── Tabs ─────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1e293b;
    gap: 0;
}

[data-baseweb="tab"] {
    color: #cbd5e1 !important;
    font-size: .85rem !important;
    padding: 10px 18px !important;
    border-bottom: 2px solid transparent !important;
}

[aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom-color: #60a5fa !important;
}

button[data-baseweb="tab"]:hover {
    color: #93c5fd !important;
}

/* ── Sidebar section label ────────────────────────────── */
.sb-lbl {
    font-size: .62rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #ffffff;
    margin-bottom: 4px;
    font-weight: 700;
    display: block;
}

/* ── File uploader ────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #111827;
    border: 1px dashed #334155;
    border-radius: 9px;
    padding: 4px;
}

/* ── Divider ──────────────────────────────────────────── */
hr {
    border-color: #334155 !important;
    margin: 1rem 0 !important;
}

/* ── Streamlit widgets text ───────────────────────────── */
.stSelectbox label,
.stMultiSelect label,
.stCheckbox label,
.stRadio label {
    color: #ffffff !important;
}

/* ── Markdown text ────────────────────────────────────── */
[data-testid="stMarkdownContainer"] * {
    color: #f8fafc;
}

/* ── Dataframe / table text ───────────────────────────── */
table {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PURE HELPER FUNCTIONS  (no Streamlit calls — easily testable)
# ══════════════════════════════════════════════════════════════════════════════

def qname(label: str)  -> str: return QUAD_META.get(label, (label, "", ""))[0]
def qcss(label: str)   -> str: return QUAD_META.get(label, ("", "q2", ""))[1]
def qcolor(label: str) -> str: return QUAD_META.get(label, ("", "", "#60a5fa"))[2]


def kpi_card(value: str, label: str,
             color: str = "#60a5fa", size: str = "1.6rem") -> str:
    return (
        f'<div class="kpi">'
        f'<div class="kpi-val" style="font-size:{size};color:{color}">{value}</div>'
        f'<div class="kpi-lbl">{label}</div>'
        f'</div>'
    )


def net_pills_html(row: pd.Series) -> str:
    pills = "".join(
        f'<span class="npill {"non" if int(pd.to_numeric(row.get(c, 0), errors="coerce") or 0) else "noff"}">'
        f"{lbl}</span>"
        for c, lbl in NET_COLS.items()
        if c in row.index
    )
    return f'<div style="margin-top:6px">{pills}</div>'


def sector_tag_html(sector: str) -> str:
    color = SEC_COL.get(sector, "#475569")
    return (
        f'<span class="acard-tag" '
        f'style="background:{color}22;color:{color};border:1px solid {color}44">'
        f"{sector}</span>"
    )


def progress_bar(pct: float, color: str) -> str:
    return (
        f'<div class="pbar">'
        f'<div class="pbar-fill" style="width:{pct:.0f}%;background:{color}"></div>'
        f"</div>"
    )


def mrow(label: str, value: str) -> str:
    return f'<div class="mrow"><span class="ml">{label}</span><span class="mv">{value}</span></div>'


def markdown_bold_to_html(text: str) -> str:
    """Turn **segments** into <strong> for HTML callouts (Markdown is not applied inside raw HTML)."""

    def _repl(match: re.Match[str]) -> str:
        return "<strong>" + html.escape(match.group(1)) + "</strong>"

    return re.sub(r"\*\*(.+?)\*\*", _repl, text)


def base_chart_layout(**overrides) -> dict:
    layout = dict(
        paper_bgcolor="#080c18",
        plot_bgcolor="#0f172a",

        font=dict(
            family="IBM Plex Sans",
            color="#ffffff"
        ),

        legend=dict(
            bgcolor="#111827",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(size=10, color="#ffffff"),
        ),

        hoverlabel=dict(
            bgcolor="#111827",
            font_size=11,
            font_family="IBM Plex Sans",
            font_color="#ffffff",
        ),

        margin=dict(l=50, r=20, t=28, b=52),
    )

    layout.update(overrides)
    return layout


def dark_axis(**kw) -> dict:
    base = dict(
        gridcolor="#1e293b",
        zeroline=False,
        tickfont=dict(color="#ffffff"),
        titlefont=dict(color="#ffffff"),
    )

    base.update(kw)
    return base

def dark_axis(**kw) -> dict:
    base = dict(gridcolor="#0f1e30", zeroline=False)
    base.update(kw)
    return base


def rgba(hex6: str, alpha: float) -> str:
    """Convert a 6-char hex colour to an rgba() string Plotly accepts."""
    h = hex6.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (both functions are @st.cache_data — keyed by file bytes hash)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Analysing fiscal data…")
def load_fiscal(raw: bytes) -> pd.DataFrame:
    xl  = pd.ExcelFile(io.BytesIO(raw))
    sht = "Sheet2" if "Sheet2" in xl.sheet_names else xl.sheet_names[0]
    df  = xl.parse(sht)
    df.columns = [str(c).strip() for c in df.columns]

    # Coerce network columns to int
    for col in NET_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Keep only rows with at least some financial data; impute medians
    fin = [c for c in FIN_COLS if c in df.columns]
    df  = df.dropna(subset=fin, how="all").copy()
    for c in fin:
        df[c] = df[c].fillna(df[c].median())

    # ── PCA ──────────────────────────────────────────────────────────────────
    X = df[fin].copy()
    for c in fin:
        if c in FLIP_COLS:
            X[c] = -X[c]                       # higher = better throughout
    X_norm = pd.DataFrame(
        StandardScaler().fit_transform(X),
        columns=fin, index=df.index,
    )
    pca_scores = PCA(n_components=min(4, len(fin))).fit_transform(X_norm)
    for i in range(pca_scores.shape[1]):
        df[f"PC{i+1}"] = pca_scores[:, i]

    # ── Pension axis: PC1 oriented so higher = lower pension burden ───────────
    pen_good = (
        X_norm["Pension Exposure Ratio"]
        if "Pension Exposure Ratio" in X_norm.columns
        else pd.Series(0.0, index=df.index)
    )
    sign = 1 if np.corrcoef(df["PC1"], pen_good)[0, 1] >= 0 else -1
    df["PC1_pension_axis"] = df["PC1"] * sign

    # ── Liquidity axis: average of cash + net investment (both z-scored) ─────
    liq_cols = [
        c for c in ["Cash & Investment Coverage", "Net Investment Capacity"]
        if c in X_norm.columns
    ]
    df["liquidity_axis"] = (
        StandardScaler().fit_transform(X_norm[liq_cols]).mean(axis=1)
        if liq_cols else np.zeros(len(df))
    )

    # ── Overall fiscal health composite ──────────────────────────────────────
    df["fiscal_health"] = X_norm.mean(axis=1)

    # ── Pre-compute z-scores for the bar chart in City vs City ────────────────
    z_scores = pd.DataFrame(
        StandardScaler().fit_transform(df[fin]),
        columns=[f"_z_{c}" for c in fin],
        index=df.index,
    )
    df = pd.concat([df, z_scores], axis=1)

    # ── 2×2 typology (median split) ──────────────────────────────────────────
    pc1_median = df["PC1_pension_axis"].median()
    liq_median = df["liquidity_axis"].median()

    def _assign_quad(row: pd.Series) -> str:
        p = "Low pension burden"  if row["PC1_pension_axis"] >= pc1_median else "High pension burden"
        l = "High liquidity"      if row["liquidity_axis"]   >= liq_median else "Low liquidity"
        return f"{p} / {l}"

    df["pca_2x2_type"] = df.apply(_assign_quad, axis=1)

    # ── Climate network membership count ─────────────────────────────────────
    nc = [c for c in NET_COLS if c in df.columns]
    df["climate_network_count"] = df[nc].sum(axis=1).astype(int)

    return df


@st.cache_data(show_spinner="Loading climate actions…")
def load_actions(raw: bytes) -> pd.DataFrame:
    xl  = pd.ExcelFile(io.BytesIO(raw))
    sht = (
        "Municipality Actions"
        if "Municipality Actions" in xl.sheet_names
        else xl.sheet_names[0]
    )
    df = xl.parse(sht)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Normalise column name: "action name" (space) → "action_name"
    df = df.rename(columns={"action name": "action_name"})

    # Standardise text fields
    for col in ("city", "state", "sector", "action_name", "action"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "city"   in df.columns: df["city"]   = df["city"].str.title()
    if "state"  in df.columns: df["state"]  = df["state"].str.title()
    if "sector" in df.columns: df["sector"] = df["sector"].str.title()

    # Keep only the three focus sectors
    if "sector" in df.columns:
        df = df[df["sector"].isin(FOCUS_SECTORS)].copy()

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL RENDER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def render_city_card(row: pd.Series, accent: str) -> None:
    """Full-width city summary card for the City vs City tab."""
    lbl, bcls, _ = QUAD_META.get(row.get("pca_2x2_type", ""), ("—", "q2", ""))
    cl  = int(row.get("commission_authority_level ", 0) or 0)
    rnc = int(row.get("regional_network_count", 0) or 0)
    cnc = int(row.get("climate_network_count",  0) or 0)
    sus = row.get(SUS_COL, "N/A")
    sus_str = f"{sus:.0f}/48" if isinstance(sus, (int, float)) else "N/A"

    st.markdown(f"""
    <div class="ccard cc-{bcls}">
      <div class="cname">{row["city"]}</div>
      <div class="cstate">{row.get("State", "")}</div>
      <span class="badge b{bcls}">{lbl}</span>
      <div style="margin-top:13px">
        {mrow("Population",          f"{int(row.get('population', 0) or 0):,}")}
        {mrow("Median Household Income", f"${int(row.get('Median Income', 0) or 0):,}")}
        {mrow("Per Capita Income",   f"${int(row.get('Per Capita', 0) or 0):,}")}
        {mrow("Sustainability Score",sus_str)}
        {mrow("Pension Health",      f"{row.get('PC1_pension_axis', 0):.2f}")}
        {mrow("Cash Liquidity",      f"{row.get('liquidity_axis', 0):.2f}")}
        {mrow("Overall Fiscal Health (vs. peers)", f"{row.get('fiscal_health', 0):.2f}")}
        {mrow("Regional Networks",   str(rnc))}
        {mrow("Climate Memberships", str(cnc))}
        {mrow("Commission Authority",COMM_LBL.get(cl, f"Level {cl}"))}
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">Climate network memberships</div>', unsafe_allow_html=True)
    st.markdown(net_pills_html(row), unsafe_allow_html=True)


def render_full_profile(row: pd.Series, accent: str,
                        fin: list[str], key_suffix: str) -> None:
    """Detailed city profile used in the City Profiles tab."""
    st.markdown(
        f'<h2 style="color:{accent}">{row["city"]} '
        f'<span style="color:#1e3045;font-size:.6em;font-family:IBM Plex Sans">'
        f'{row.get("State", "")}</span></h2>',
        unsafe_allow_html=True,
    )

    # ── Demographics ──────────────────────────────────────────────────────────
    d1, d2, d3, d4 = st.columns(4)
    for col_w, v, lbl in [
        (d1, f"{int(row.get('population', 0) or 0):,}", "Population"),
        (d2, str(row.get("median_age", "—")),           "Median Age"),
        (d3, f"${int(row.get('Median Income', 0) or 0):,}", "Median Income"),
        (d4, f"${int(row.get('Per Capita', 0) or 0):,}",    "Per Capita Income"),
    ]:
        with col_w:
            st.markdown(kpi_card(v, lbl, accent, "1.05rem"), unsafe_allow_html=True)

    # ── Sustainability scores ─────────────────────────────────────────────────
    st.markdown('<div class="sec-lbl" style="margin-top:14px">Sustainability scores</div>', unsafe_allow_html=True)
    for sc_col, sc_lbl, sc_max in [*SUS_SUBS, (SUS_COL, "Total Score", 48)]:
        if sc_col in row.index:
            val = float(row.get(sc_col, 0) or 0)
            pct = val / sc_max * 100
            st.markdown(
                f'{mrow(sc_lbl, f"{val:.0f}/{sc_max}")}'
                f'{progress_bar(pct, accent)}',
                unsafe_allow_html=True,
            )

    # ── Fiscal cluster ────────────────────────────────────────────────────────
    st.markdown('<div class="sec-lbl" style="margin-top:14px">Fiscal position</div>', unsafe_allow_html=True)
    qlbl, qbcls, _ = QUAD_META.get(row.get("pca_2x2_type", ""), ("—", "q2", ""))
    st.markdown(
        f'<span class="badge b{qbcls}" style="font-size:.74rem;padding:4px 13px">{qlbl}</span>',
        unsafe_allow_html=True,
    )
    for fld, flbl in [
        ("PC1_pension_axis", "Pension Health Score"),
        ("liquidity_axis",   "Cash Liquidity Score"),
        ("fiscal_health",    "Overall Fiscal Health (vs. peers)"),
    ]:
        st.markdown(mrow(flbl, f"{row.get(fld, 0):.2f}"), unsafe_allow_html=True)

    # ── Governance & networks ─────────────────────────────────────────────────
    st.markdown('<div class="sec-lbl" style="margin-top:14px">Governance & networks</div>', unsafe_allow_html=True)
    cl  = int(row.get("commission_authority_level ", 0) or 0)
    rnc = int(row.get("regional_network_count",     0) or 0)
    cnc = int(row.get("climate_network_count",      0) or 0)
    for fv, fl in [
        (COMM_LBL.get(cl, f"Level {cl}"), "Commission authority"),
        (str(rnc), "Regional networks"),
        (str(cnc), "Climate memberships"),
    ]:
        st.markdown(mrow(fl, fv), unsafe_allow_html=True)
    st.markdown(net_pills_html(row), unsafe_allow_html=True)

    # ── Financial profile vs dataset median (bar view) ───────────────────────
    z_fin = [f"_z_{c}" for c in fin]
    if z_fin and all(zc in row.index for zc in z_fin):
        st.markdown(
            '<div class="sec-lbl" style="margin-top:16px">Financial profile vs. peers</div>',
            unsafe_allow_html=True,
        )
        labels  = [c.split("(")[0].strip()[:22] for c in fin]
        city_z  = [float(row[zc]) for zc in z_fin]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=row["city"],
            x=labels,
            y=city_z,
            marker_color=accent,
            opacity=0.85,
        ))
        fig.add_hline(
            y=0,
            line_color="#6b7280",
            line_width=1,
            line_dash="dot",
            annotation_text="Peer average",
            annotation_position="top left",
        )
        fig.update_layout(
            **base_chart_layout(height=320, margin=dict(l=45, r=15, t=20, b=100)),
            yaxis=dark_axis(
                title="Deviation from peer average",
                zeroline=True,
                zerolinecolor="#1a3050",
            ),
            xaxis=dict(
                tickangle=-38,
                tickfont=dict(size=8),
                gridcolor="#0f1e30",
            ),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"fin_profile_bar_{key_suffix}")


def render_action_list(
    city_nm: str,
    acts: pd.DataFrame,
    sel_sectors: list[str],
    *,
    list_state_key: str,
) -> None:
    """Render filtered action cards for one city (paginated when the list is long)."""
    filtered = (
        acts[acts["sector"].isin(sel_sectors)]
        if sel_sectors and "sector" in acts.columns
        else acts
    )

    n_all = len(filtered)
    st.markdown(
        f'<div class="sec-lbl">{city_nm} &nbsp;·&nbsp; {n_all} actions</div>',
        unsafe_allow_html=True,
    )

    if filtered.empty:
        no_data = len(acts) == 0
        if no_data:
            st.markdown(
                '<div class="info-banner">No climate actions on record for this city. '
                'This may mean the city has not published a sustainability plan yet, '
                'or the document is not yet in the dataset.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No actions match the selected sectors.")
        return

    page_key = f"_alpage_{list_state_key}"
    sig_key = f"_alsig_{list_state_key}"
    sig_val = (tuple(sorted(sel_sectors or [])), n_all)
    if st.session_state.get(sig_key) != sig_val:
        st.session_state[sig_key] = sig_val
        st.session_state[page_key] = 0

    n_pages = max(1, (n_all + ACTION_PAGE_SIZE - 1) // ACTION_PAGE_SIZE)
    page = int(st.session_state.get(page_key, 0))
    page = max(0, min(page, n_pages - 1))
    st.session_state[page_key] = page

    start = page * ACTION_PAGE_SIZE
    end = min(start + ACTION_PAGE_SIZE, n_all)
    page_df = filtered.iloc[start:end]

    if n_pages > 1:
        st.caption(f"Showing actions {start + 1}–{end} of {n_all} · Page {page + 1} of {n_pages}")
        p1, p2, p3 = st.columns([1, 2, 1])
        with p1:
            if st.button("◀ Prev", key=f"{list_state_key}_prev", disabled=page <= 0):
                st.session_state[page_key] = page - 1
                st.rerun()
        with p3:
            if st.button(
                "Next ▶",
                key=f"{list_state_key}_next",
                disabled=page >= n_pages - 1,
            ):
                st.session_state[page_key] = page + 1
                st.rerun()

    for _, row in page_df.iterrows():
        name   = row.get("action_name", "")
        text   = row.get("action", "")
        sector = str(row.get("sector", "Other"))
        st.markdown(
            f'<div class="acard">'
            f'<div class="acard-name">{name}</div>'
            f'<div class="acard-text">{text}</div>'
            f'{sector_tag_html(sector)}'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_focus_sector_pie(
    acts: pd.DataFrame,
    title_line: str,
    accent: str,
    plot_key: str,
) -> None:
    """Donut chart: counts of Energy, Transport, Waste actions (FOCUS_SECTORS order)."""
    st.markdown(
        f'<p style="color:{accent};font-weight:600;margin-bottom:4px">{title_line}</p>',
        unsafe_allow_html=True,
    )
    if acts.empty or "sector" not in acts.columns:
        vc = pd.Series(0, index=FOCUS_SECTORS, dtype=int)
    else:
        vc = acts["sector"].value_counts().reindex(FOCUS_SECTORS, fill_value=0).astype(int)
    total = int(vc.sum())
    st.caption(
        " · ".join(f"{s}: {int(vc[s])}" for s in FOCUS_SECTORS)
        + f" — {total} total in these sectors"
    )
    labels = [s for s in FOCUS_SECTORS if int(vc[s]) > 0]
    vals = [int(vc[s]) for s in labels]
    if total == 0:
        st.caption("No actions on record for this city in the dataset.")
        return
    fig_p = go.Figure(go.Pie(
        labels=labels,
        values=vals,
        hole=0.52,
        marker_colors=[SEC_COL.get(s, "#475569") for s in labels],
        textinfo="label+value",
        textposition="outside",
        textfont=dict(size=10, color="#e2e8f0"),
        hovertemplate="%{label}: %{value} actions (%{percent})<extra></extra>",
    ))
    fig_p.update_layout(
        **base_chart_layout(height=260, margin=dict(l=8, r=8, t=8, b=8)),
        showlegend=False,
    )
    st.plotly_chart(fig_p, use_container_width=True, key=plot_key)


def pick_fiscal_peer_benchmark(dframe: pd.DataFrame, focal: pd.Series) -> Optional[pd.Series]:
    """
    Choose a benchmark municipality with similar Overall Fiscal Health index
    (`fiscal_health`: peer-average financial indicators) but a meaningfully
    higher sustainability total score.
    """
    if focal is None or SUS_COL not in dframe.columns:
        return None
    fh_t = float(pd.to_numeric(focal.get("fiscal_health"), errors="coerce") or 0.0)
    sus_t = float(pd.to_numeric(focal.get(SUS_COL), errors="coerce") or 0.0)
    fh_std = float(dframe["fiscal_health"].std() or 0.25)
    if np.isnan(fh_std) or fh_std < 1e-6:
        fh_std = 0.25

    m_self = (dframe["city"] == focal["city"]) & (dframe["State"] == focal["State"])
    pool = dframe.loc[~m_self].copy()
    if pool.empty:
        return None

    pool["_dfh"] = (
        pd.to_numeric(pool["fiscal_health"], errors="coerce").fillna(0.0) - fh_t
    ).abs()
    pool["_sus"] = pd.to_numeric(pool[SUS_COL], errors="coerce").fillna(0.0)

    for mult in (0.28, 0.42, 0.6, 0.85, 1.15, 1.55, 2.2, 3.5, 1e9):
        band = mult * fh_std
        near = pool[pool["_dfh"] <= band]
        better = near[near["_sus"] > sus_t + 0.08]
        if not better.empty:
            ix = better.sort_values(["_dfh", "_sus"], ascending=[True, False]).index[0]
            return dframe.loc[ix]

    better_all = pool[pool["_sus"] > sus_t + 0.08]
    if not better_all.empty:
        ix = better_all.sort_values(["_dfh", "_sus"], ascending=[True, False]).index[0]
        return dframe.loc[ix]
    return None


def _comm_level(row: pd.Series) -> int:
    for k in ("commission_authority_level ", "commission_authority_level"):
        if k in row.index:
            return int(pd.to_numeric(row.get(k), errors="coerce") or 0)
    return 0


def render_improvement_benchmark_for_city(
    focal: Optional[pd.Series],
    city_nm: str,
    state_nm: str,
    accent_sel: str,
    accent_bench: str,
    sector_filter: list[str],
    chart_key_suffix: str,
) -> None:
    """Peer-learning panel: focal city vs. a higher-scoring, fiscally similar benchmark."""
    if focal is None:
        st.warning(f"**{city_nm}, {state_nm}** not found in the dataset.")
        return

    bench = pick_fiscal_peer_benchmark(df, focal)
    if bench is None:
        st.info(
            f"No benchmark peer found for **{city_nm}** with a higher sustainability score "
            "in this dataset. Try another city, or this city may already rank at the top "
            "among comparable fiscal peers."
        )
        return

    fh_f = float(focal.get("fiscal_health", 0) or 0)
    fh_b = float(bench.get("fiscal_health", 0) or 0)
    sus_f = float(focal.get(SUS_COL, 0) or 0)
    sus_b = float(bench.get(SUS_COL, 0) or 0)
    dfh = abs(fh_b - fh_f)
    bench_label = f'{bench["city"]}, {bench.get("State", "")}'

    st.markdown(
        f'<div class="info-banner" style="margin-bottom:14px">'
        f"<b>Benchmark peer:</b> {bench_label}<br>"
        f"<span style='opacity:.92'>Matched because both cities sit in a <b>similar "
        f"Overall Fiscal Health</b> band—that index compares each city to the "
        f"<b>dataset average</b> on cash, debt, pensions, capital burden, and related "
        f"financial fields (see <i>How is Overall Fiscal Health calculated?</i> above). "
        f"<b>{bench['city']}</b> index <b>{fh_b:.2f}</b> · <b>{city_nm}</b> "
        f"<b>{fh_f:.2f}</b> (difference {dfh:.2f}; small gap = similar financial "
        f"position among peers here). The benchmark still earns a higher "
        f"<b>sustainability</b> score: <b>{sus_b:.1f}/48</b> vs "
        f"<b>{sus_f:.1f}/48</b>.</span></div>",
        unsafe_allow_html=True,
    )

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            kpi_card(f"{sus_f:.1f}/48", f"{city_nm} — sustainability", accent_sel, "1.25rem"),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            kpi_card(f"{sus_b:.1f}/48", "Benchmark — sustainability", accent_bench, "1.25rem"),
            unsafe_allow_html=True,
        )
        with k3:
            st.markdown(
                kpi_card(f"+{sus_b - sus_f:.1f}", "Score gap (benchmark − selected)", "#34d399", "1.25rem"),
                unsafe_allow_html=True,
            )

    acts_f = get_city_actions(city_nm, state_nm)
    acts_b = get_city_actions(str(bench["city"]), str(bench.get("State", "")))

    st.markdown("### Actions by sector (Energy, Transport, Waste)")
    st.caption("Each pie shows how documented actions in the dataset split across the three sectors.")
    pie_l, pie_r = st.columns(2)
    with pie_l:
        render_focus_sector_pie(
            acts_f,
            f"{city_nm}, {state_nm}",
            accent_sel,
            f"improve_pie_f_{chart_key_suffix}",
        )
    with pie_r:
        render_focus_sector_pie(
            acts_b,
            bench_label,
            accent_bench,
            f"improve_pie_b_{chart_key_suffix}",
        )

    st.markdown("### Side-by-side climate actions")
    st.caption(
        "Documented Energy, Transport, and Waste measures from the actions dataset. "
        "Use the sector filter in this tab to narrow the lists."
    )
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown(
            f'<p style="color:{accent_sel};font-weight:600;margin-bottom:6px">'
            f"{city_nm}, {state_nm}</p>",
            unsafe_allow_html=True,
        )
        render_action_list(f"{city_nm}, {state_nm}", acts_f, sector_filter, list_state_key=f"improve_act_f_{chart_key_suffix}")
    with c_right:
        st.markdown(
            f'<p style="color:{accent_bench};font-weight:600;margin-bottom:6px">'
            f"{bench_label}</p>",
            unsafe_allow_html=True,
        )
        render_action_list(bench_label, acts_b, sector_filter, list_state_key=f"improve_act_b_{chart_key_suffix}")

    st.markdown("### Where policies and programs diverge")
    diff_lines: list[str] = []
    for sc_col, sc_lbl, sc_max in SUS_SUBS:
        if sc_col not in focal.index or sc_col not in bench.index:
            continue
        vf, vb = float(focal.get(sc_col, 0) or 0), float(bench.get(sc_col, 0) or 0)
        if vb > vf + 0.5:
            diff_lines.append(
                f"**{sc_lbl.split('(')[0].strip()}**: benchmark scores **{vb:.0f}/{sc_max}** "
                f"vs **{vf:.0f}/{sc_max}** for {city_nm} — stronger documented maturity in "
                f"this pillar."
            )
    c_f, c_b = _comm_level(focal), _comm_level(bench)
    if c_b > c_f:
        diff_lines.append(
            f"**Governance / formal authority**: benchmark has **{COMM_LBL.get(c_b, str(c_b))}** "
            f"for its sustainability commission vs **{COMM_LBL.get(c_f, str(c_f))}** for {city_nm}."
        )
    cn_f = int(pd.to_numeric(focal.get("climate_network_count"), errors="coerce") or 0)
    cn_b = int(pd.to_numeric(bench.get("climate_network_count"), errors="coerce") or 0)
    if cn_b > cn_f:
        diff_lines.append(
            f"**Climate networks**: benchmark participates in **{cn_b}** tracked memberships "
            f"vs **{cn_f}** for {city_nm}, signalling broader multi-city learning and visibility."
        )
    rn_f = int(pd.to_numeric(focal.get("regional_network_count"), errors="coerce") or 0)
    rn_b = int(pd.to_numeric(bench.get("regional_network_count"), errors="coerce") or 0)
    if rn_b > rn_f:
        diff_lines.append(
            f"**Regional collaboration**: **{rn_b}** regional network ties vs **{rn_f}** — "
            "often associated with shared resources and implementation support."
        )

    if (("sector" in acts_f.columns and not acts_f.empty)
            or ("sector" in acts_b.columns and not acts_b.empty)):
        sf = acts_f["sector"].value_counts() if "sector" in acts_f.columns else pd.Series(dtype=int)
        sb = acts_b["sector"].value_counts() if "sector" in acts_b.columns else pd.Series(dtype=int)
        for sec in FOCUS_SECTORS:
            nf, nb = int(sf.get(sec, 0)), int(sb.get(sec, 0))
            if nb > nf:
                diff_lines.append(
                    f"**{sec} actions**: benchmark lists **{nb}** documented items vs **{nf}** "
                    f"for {city_nm} in this dataset."
                )

    if diff_lines:
        body = "<br>".join(markdown_bold_to_html(line) for line in diff_lines)
        st.markdown(
            '<div style="border-left:4px solid #34d399;padding:10px 14px;margin:12px 0;'
            'background:#0f172a;border-radius:0 8px 8px 0;line-height:1.65">'
            + body
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption(
            "No single dominant gap stands out in the automated comparison; see score "
            "breakdown and recommendations below."
        )

    st.markdown("### Why the benchmark scores higher at similar fiscal capacity")
    st.markdown(
        f"The peer match uses the **Overall Fiscal Health** index described above "
        f"(cash, debt, pensions, capital stress, etc., each compared to this dataset’s "
        f"average—not population or income). **{bench_label}** and **{city_nm}** are only "
        f"**{dfh:.2f}** index points apart, yet the benchmark leads by "
        f"**{sus_b - sus_f:.1f}** sustainability points. That gap usually reflects "
        f"**governance, planning, data practices, and climate programs** in the rubric, "
        f"not a materially “richer” city in this financial snapshot.",
    )

    st.markdown("### Key performance drivers and strategic advantages")
    drivers: list[str] = []
    gov_f = float(focal.get("Governance\n(/9)", 0) or 0)
    gov_b = float(bench.get("Governance\n(/9)", 0) or 0)
    da_f = float(focal.get("Data & Analytics\n(/21)", 0) or 0)
    da_b = float(bench.get("Data & Analytics\n(/21)", 0) or 0)
    ap_f = float(focal.get("Action Planning\n(/18)", 0) or 0)
    ap_b = float(bench.get("Action Planning\n(/18)", 0) or 0)
    pillars = [
        ("Governance", gov_b - gov_f, gov_b),
        ("Data & analytics", da_b - da_f, da_b),
        ("Action planning", ap_b - ap_f, ap_b),
    ]
    pillars.sort(key=lambda x: x[1], reverse=True)
    for name, gap, val_b in pillars:
        if gap > 0.3:
            drivers.append(
                f"**{name}** (benchmark **{val_b:.0f}** pts in this block vs **{val_b - gap:.0f}** "
                f"for {city_nm}): largest relative lift in the rubric."
            )
    if not drivers:
        drivers.append(
            "Gains are spread across pillars; the benchmark still leads on the **total** score "
            "after matching fiscal capacity — review sub-scores above for nuance."
        )
    if cn_b > cn_f:
        drivers.append(
            "**Institutional embedding**: climate network participation often correlates with "
            "structured reporting, targets, and staff capacity — areas reflected in the score."
        )
    st.markdown("\n".join(f"- {d}" for d in drivers))

    fig_sub = go.Figure()
    sub_lbls = ["Governance", "Data & analytics", "Action planning"]
    sub_f = [gov_f, da_f, ap_f]
    sub_b = [gov_b, da_b, ap_b]
    fig_sub.add_trace(go.Bar(name=city_nm, x=sub_lbls, y=sub_f, marker_color=accent_sel, opacity=0.88))
    fig_sub.add_trace(
        go.Bar(name=bench["city"], x=sub_lbls, y=sub_b, marker_color=accent_bench, opacity=0.88)
    )
    _lay = base_chart_layout(height=300, margin=dict(l=40, r=15, t=28, b=45))
    _leg = dict(_lay.get("legend") or {})
    _leg.update(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    )
    _lay["legend"] = _leg
    fig_sub.update_layout(
        **_lay,
        barmode="group",
        yaxis=dark_axis(title="Rubric points (raw)"),
        xaxis=dict(tickfont=dict(size=11), gridcolor="#0f1e30"),
    )
    st.plotly_chart(fig_sub, use_container_width=True, key=f"bench_subscores_{chart_key_suffix}")

    st.markdown("### Recommendations / suggested actions")
    recs: list[str] = []
    if gov_b > gov_f + 0.5:
        recs.append(
            f"**Governance**: Align commission scope and decision rights with **{bench['city']}**-style "
            "practice (clear mandate, reporting cadence, documented authority level)."
        )
    if da_b > da_f + 0.5:
        recs.append(
            "**Data & analytics**: Publish inventories and metrics (energy, fleet, buildings) "
            "and tie capital decisions to measurable baselines — typical of higher rubric scores."
        )
    if ap_b > ap_f + 0.5:
        recs.append(
            "**Action planning**: Refresh CAP milestones, interdepartmental owners, and "
            "funding pathways; mirror the breadth of initiatives seen in the benchmark’s action list."
        )
    if c_b > c_f:
        recs.append(
            f"**Formal authority**: Explore moving the sustainability commission toward "
            f"**{COMM_LBL.get(c_b, str(c_b)).lower()}**, matching the benchmark’s level."
        )
    if cn_b > cn_f:
        recs.append(
            "**Networks**: Join or deepen participation in national/regional climate networks "
            "where the benchmark is active — use their toolkits and peer cohorts."
        )
    for sec in FOCUS_SECTORS:
        nf = int(acts_f["sector"].value_counts().get(sec, 0)) if "sector" in acts_f.columns else 0
        nb = int(acts_b["sector"].value_counts().get(sec, 0)) if "sector" in acts_b.columns else 0
        if nb > nf:
            recs.append(
                f"**{sec}**: Add or document **{sec.lower()}** programs to close the gap "
                f"({nf} vs {nb} listed actions)."
            )
    if not recs:
        recs.append(
            "Focus on **incremental rubric gains** across governance, data, and planning — "
            "the benchmark leads overall; small systematic upgrades in each pillar often compound."
        )
    st.markdown("\n".join(f"{i}. {r}" for i, r in enumerate(recs, start=1)))


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — fixed data source + cascading state → city pickers
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏛️ Municipal Dashboard")
    st.markdown(
        '<p style="color:#ffffff;font-size:.68rem;margin-top:-6px">'
        "Fiscal Typology · Sustainability · Climate Actions</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### Data source")
    st.caption("Using fixed Excel files from the frontend folder.")

# ── Load fiscal data ──────────────────────────────────────────────────────────
try:
    with open(DISK_FISCAL, "rb") as fh:
        fiscal_bytes = fh.read()
except FileNotFoundError:
    st.error(f"Fiscal file not found: {DISK_FISCAL}")
    st.stop()

try:
    df = load_fiscal(fiscal_bytes)
except Exception as exc:
    st.error(f"Could not read fiscal data: {exc}")
    st.stop()

# ── Load actions (fixed file from disk) ───────────────────────────────────────
action_bytes = None
try:
    with open(DISK_ACTIONS, "rb") as fh:
        action_bytes = fh.read()
except FileNotFoundError:
    st.warning(f"Actions file not found: {DISK_ACTIONS}")

df_act: Optional[pd.DataFrame] = None
if action_bytes:
    try:
        df_act = load_actions(action_bytes)
        # Filter to only the cities present in the fiscal dataset
        valid_pairs = frozenset(
            zip(df["city"].str.strip().str.lower(),
                df["State"].str.strip().str.lower())
        )
        keep = df_act.apply(
            lambda r: (r["city"].lower(), r["state"].lower()) in valid_pairs, axis=1
        )
        df_act = df_act[keep].reset_index(drop=True)
    except Exception as exc:
        st.warning(f"Could not load actions data: {exc}")

# ── Pre-compute derived lists used in multiple tabs ───────────────────────────
fin_avail  = [c for c in FIN_COLS if c in df.columns]
z_cols     = [f"_z_{c}" for c in fin_avail]
fin_labels = [c.split("(")[0].strip()[:26] for c in fin_avail]
pc1_median = df["PC1_pension_axis"].median()
liq_median = df["liquidity_axis"].median()
sus_mean   = df[SUS_COL].mean() if SUS_COL in df.columns else 0.0
rnc_mean   = df["regional_network_count"].mean() if "regional_network_count" in df else 0.0
all_states = sorted(df["State"].dropna().unique().tolist())
compare_states = [s for s in all_states if s not in {"Indiana", "Iowa"}]
highlight_states = [s for s in all_states if s not in {"Indiana", "Iowa"}]
FOCUS_STATES_ORDER = ("Illinois", "Michigan", "Minnesota", "Wisconsin")
focus_states_list = [s for s in FOCUS_STATES_ORDER if s in df["State"].dropna().unique().tolist()]
if not focus_states_list:
    focus_states_list = [compare_states[0]] if compare_states else []

# Dashboard view mode (compare vs single-city focus for all tabs)
VIEW_MODE_COMPARE = "compare_two"
VIEW_MODE_CITY_A = "city_a_only"
VIEW_MODE_CITY_B = "city_b_only"
VIEW_MODE_CUSTOM = "single_custom"
VIEW_MODE_OPTIONS: list[tuple[str, str]] = [
    (VIEW_MODE_COMPARE, "Compare City A & City B"),
    (VIEW_MODE_CITY_A, "City A only (one city everywhere)"),
    (VIEW_MODE_CITY_B, "City B only (one city everywhere)"),
    (VIEW_MODE_CUSTOM, "Single city — Illinois, Michigan, Minnesota, or Wisconsin"),
]

def cities_for(state: str) -> list[str]:
    return sorted(df[df["State"] == state]["city"].dropna().unique().tolist())

# ── Cascading state → city dropdowns ─────────────────────────────────────────
def _default(key: str, value: str) -> None:
    if key not in st.session_state:
        st.session_state[key] = value

_default("state_a", "Illinois" if "Illinois" in compare_states else compare_states[0])
_default("state_b", "Illinois" if "Illinois" in compare_states else compare_states[0])
if st.session_state.get("state_a") not in compare_states:
    st.session_state["state_a"] = compare_states[0]
if st.session_state.get("state_b") not in compare_states:
    st.session_state["state_b"] = compare_states[0]

def on_state_a_change() -> None:
    c = cities_for(st.session_state["state_a"])
    st.session_state["city_a"] = c[0] if c else ""

def on_state_b_change() -> None:
    c = cities_for(st.session_state["state_b"])
    st.session_state["city_b"] = c[0] if c else ""


def on_focus_custom_state_change() -> None:
    c = cities_for(st.session_state["focus_custom_state"])
    st.session_state["focus_custom_city"] = c[0] if c else ""

cities_a_init = cities_for(st.session_state["state_a"])
cities_b_init = cities_for(st.session_state["state_b"])
_default("city_a", "Brookfield" if "Brookfield" in cities_a_init
         else (cities_a_init[0] if cities_a_init else ""))
_default("city_b", "Naperville" if "Naperville" in cities_b_init
         else (cities_b_init[1] if len(cities_b_init) > 1
               else (cities_b_init[0] if cities_b_init else "")))

_default("dashboard_view_mode", VIEW_MODE_COMPARE)
_fc0 = cities_for(focus_states_list[0])
_default("focus_custom_state", focus_states_list[0])
_default("focus_custom_city", _fc0[0] if _fc0 else "")

with st.sidebar:
    st.divider()
    st.markdown("### Compare cities")

    # ── City A ────────────────────────────────────────────────────────────────
    st.markdown('<span class="sb-lbl">City A 🔵</span>', unsafe_allow_html=True)
    st.selectbox("State A", compare_states, key="state_a",
                 on_change=on_state_a_change, label_visibility="collapsed")
    c_a = cities_for(st.session_state["state_a"])
    if st.session_state.get("city_a") not in c_a:
        st.session_state["city_a"] = c_a[0] if c_a else ""
    st.selectbox("City A", c_a, key="city_a", label_visibility="collapsed")

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    # ── City B ────────────────────────────────────────────────────────────────
    st.markdown('<span class="sb-lbl">City B 🟡</span>', unsafe_allow_html=True)
    st.selectbox("State B", compare_states, key="state_b",
                 on_change=on_state_b_change, label_visibility="collapsed")
    c_b = cities_for(st.session_state["state_b"])
    if st.session_state.get("city_b") not in c_b:
        st.session_state["city_b"] = c_b[0] if c_b else ""
    st.selectbox("City B", c_b, key="city_b", label_visibility="collapsed")

    st.divider()
    st.markdown("### Dashboard view")
    st.caption("Single-city modes show only that municipality across all tabs.")
    _mode_labels = {k: v for k, v in VIEW_MODE_OPTIONS}
    st.selectbox(
        "View mode",
        options=[x[0] for x in VIEW_MODE_OPTIONS],
        format_func=lambda k: _mode_labels[k],
        key="dashboard_view_mode",
        label_visibility="collapsed",
    )
    if st.session_state.get("dashboard_view_mode") == VIEW_MODE_CUSTOM:
        st.markdown('<span class="sb-lbl">Focus state (IL / MI / MN / WI)</span>', unsafe_allow_html=True)
        st.selectbox(
            "Focus state",
            focus_states_list,
            key="focus_custom_state",
            on_change=on_focus_custom_state_change,
            label_visibility="collapsed",
        )
        _cf_list = cities_for(st.session_state["focus_custom_state"])
        if st.session_state.get("focus_custom_city") not in _cf_list:
            st.session_state["focus_custom_city"] = _cf_list[0] if _cf_list else ""
        st.markdown('<span class="sb-lbl">Focus city</span>', unsafe_allow_html=True)
        st.selectbox(
            "Focus city",
            _cf_list,
            key="focus_custom_city",
            label_visibility="collapsed",
        )

    st.divider()
    st.markdown("### Map options")
    show_pop    = st.checkbox("Scale bubbles by population", value=True,  key="show_pop")
    show_labels = st.checkbox("Show all city labels",        value=False, key="show_lbl")
    shade       = st.checkbox("Shade quadrant areas",        value=True,  key="shade")
    hl_states   = st.multiselect("Highlight state(s)",
                                  highlight_states, default=[], key="hl_st")

# ── Effective cities (respect dashboard view mode) ─────────────────────────────
_vm = st.session_state.get("dashboard_view_mode", VIEW_MODE_COMPARE)
if _vm == VIEW_MODE_CITY_A:
    city1, state1 = st.session_state["city_a"], st.session_state["state_a"]
    city2, state2 = city1, state1
    single_city_mode = True
elif _vm == VIEW_MODE_CITY_B:
    city1, state1 = st.session_state["city_b"], st.session_state["state_b"]
    city2, state2 = city1, state1
    single_city_mode = True
elif _vm == VIEW_MODE_CUSTOM:
    city1 = str(st.session_state.get("focus_custom_city") or "")
    state1 = str(st.session_state.get("focus_custom_state") or "")
    city2, state2 = city1, state1
    single_city_mode = True
else:
    city1, state1 = st.session_state["city_a"], st.session_state["state_a"]
    city2, state2 = st.session_state["city_b"], st.session_state["state_b"]
    single_city_mode = False

def get_row(city: str, state: str) -> Optional[pd.Series]:
    mask = (df["city"] == city) & (df["State"] == state)
    return df[mask].iloc[0] if mask.any() else None

r1 = get_row(city1, state1)
r2 = get_row(city2, state2)

# ── Actions for selected cities (match on city + state) ──────────────────────
def get_city_actions(city: str, state: str) -> pd.DataFrame:
    if df_act is None:
        return pd.DataFrame()
    return df_act[
        (df_act["city"].str.lower()  == city.lower()) &
        (df_act["state"].str.lower() == state.lower())
    ].copy()

a1 = get_city_actions(city1, state1)
a2 = get_city_actions(city2, state2)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# Municipal Fiscal & Sustainability Dashboard")
if single_city_mode:
    st.markdown(
        f'<p style="color:#ffffff;font-size:.76rem;margin-top:-10px">'
        f"{len(df)} cities &nbsp;·&nbsp; {df['State'].nunique()} states &nbsp;·&nbsp; "
        f'<b>Single-city view:</b> <span style="color:#93c5fd">{city1}, {state1}</span>'
        f"</p>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<p style="color:#ffffff;font-size:.76rem;margin-top:-10px">'
        f"{len(df)} cities &nbsp;·&nbsp; {df['State'].nunique()} states &nbsp;·&nbsp; "
        f'Comparing <b style="color:#ffffff">{city1}, {state1}</b>'
        f' &nbsp;vs&nbsp; '
        f'<b style="color:#fbbf24">{city2}, {state2}</b>'
        f"</p>",
        unsafe_allow_html=True,
    )

# ── KPI strip ─────────────────────────────────────────────────────────────────
sus_a = float(r1.get(SUS_COL, 0) or 0) if r1 is not None and SUS_COL in r1 else 0.0
sus_b = float(r2.get(SUS_COL, 0) or 0) if r2 is not None and SUS_COL in r2 else 0.0
if single_city_mode:
    k_cols = st.columns(3)
    for col_w, (v, lbl) in zip(k_cols, [
        (str(len(df)),                               "Cities in dataset"),
        (str(df["State"].nunique()),                 "States in dataset"),
        (f"{sus_a:.1f}/48",                          f"{city1} — sustainability"),
    ]):
        with col_w:
            st.markdown(kpi_card(v, lbl), unsafe_allow_html=True)
else:
    k_cols = st.columns(4)
    for col_w, (v, lbl) in zip(k_cols, [
        (str(len(df)),                               "Cities"),
        (str(df["State"].nunique()),                 "States"),
        (f"{sus_a:.1f}/48",                          f"{city1} Sustainability"),
        (f"{sus_b:.1f}/48",                          f"{city2} Sustainability"),
    ]):
        with col_w:
            st.markdown(kpi_card(v, lbl), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
T1, T2, T3, T4, T5 = st.tabs([
    "🗺️  Typology Map",
    "⚖️  City vs City",
    "⚡  Actions Explorer",
    "🔬  City Profiles",
    "How Can Cities Improve?",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1  ·  TYPOLOGY MAP
# ──────────────────────────────────────────────────────────────────────────────
with T1:
    st.markdown(
        '<div class="info-banner">'
        '<b>How cluster names are assigned:</b><br>'
        '• <b>Resilient (Q1)</b>: lower pension burden and higher cash liquidity.<br>'
        '• <b>Stable (Q2)</b>: lower pension burden but tighter cash liquidity.<br>'
        '• <b>Pressured (Q3)</b>: higher pension burden, but currently stronger liquidity.<br>'
        '• <b>Vulnerable (Q4)</b>: higher pension burden and lower liquidity.'
        '</div>',
        unsafe_allow_html=True,
    )

    xpad = (df["PC1_pension_axis"].max() - df["PC1_pension_axis"].min()) * .1
    ypad = (df["liquidity_axis"].max()   - df["liquidity_axis"].min())   * .1
    xr = [df["PC1_pension_axis"].min() - xpad, df["PC1_pension_axis"].max() + xpad]
    yr = [df["liquidity_axis"].min()   - ypad, df["liquidity_axis"].max()   + ypad]

    fig_map = go.Figure()

    # Quadrant shading
    if shade:
        quad_shapes = [
            (xr[0], pc1_median, liq_median, yr[1], "rgba(52,211,153,.05)",  "Q1  RESILIENT",  "#34d399",
             xr[0] + (pc1_median - xr[0]) * .06, yr[1] - (yr[1] - liq_median) * .09),
            (pc1_median, xr[1], liq_median, yr[1], "rgba(96,165,250,.05)",  "Q2  STABLE",     "#60a5fa",
             pc1_median + (xr[1] - pc1_median) * .55, yr[1] - (yr[1] - liq_median) * .09),
            (xr[0], pc1_median, yr[0], liq_median, "rgba(251,191,36,.05)",  "Q3  PRESSURED",  "#fbbf24",
             xr[0] + (pc1_median - xr[0]) * .06, yr[0] + (liq_median - yr[0]) * .12),
            (pc1_median, xr[1], yr[0], liq_median, "rgba(248,113,113,.05)", "Q4  VULNERABLE", "#f87171",
             pc1_median + (xr[1] - pc1_median) * .55, yr[0] + (liq_median - yr[0]) * .12),
        ]
        for x0, x1, y0, y1, fill, qlbl, tc, ax, ay in quad_shapes:
            fig_map.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                              fillcolor=fill, line_width=0, layer="below")
            fig_map.add_annotation(x=ax, y=ay, text=qlbl,
                                   font=dict(color=tc, size=8, family="IBM Plex Mono"),
                                   showarrow=False, xanchor="left")

    fig_map.add_hline(y=liq_median, line_dash="dot", line_color="#18243a", line_width=1.5)
    fig_map.add_vline(x=pc1_median, line_dash="dot", line_color="#18243a", line_width=1.5)

    # All cities grouped by quadrant
    for quad, (qn, _, qcol) in QUAD_META.items():
        sub = df[df["pca_2x2_type"] == quad]
        if sub.empty:
            continue
        sizes = (
            np.clip(np.sqrt(sub["population"].fillna(50_000) / 1_000) * 3.2, 5, 26)
            if show_pop else np.full(len(sub), 9.0)
        )
        opacity = np.where(
            sub["State"].isin(hl_states) if hl_states else np.ones(len(sub), bool),
            0.85, 0.40 if hl_states else 0.62,
        ).tolist()
        sus_vals = sub.get(SUS_COL, pd.Series(np.nan, index=sub.index)).fillna(0).round(1)
        fig_map.add_trace(go.Scatter(
            x=sub["PC1_pension_axis"],
            y=sub["liquidity_axis"],
            mode="markers+text" if show_labels else "markers",
            name=qn,
            text=sub["city"] if show_labels else None,
            textposition="top center",
            textfont=dict(size=6.5, color="#ffffff"),
            marker=dict(
                size=sizes, color=qcol, opacity=opacity,
                line=dict(width=1, color="rgba(255,255,255,.1)"),
            ),
            customdata=np.stack([
                sub["city"], sub["State"],
                sus_vals,
                sub["fiscal_health"].round(2),
                sub["population"].fillna(0).astype(int),
                sub["pca_2x2_type"],
                sub["climate_network_count"],
            ], axis=1),
            hovertemplate=(
                "<b>%{customdata[0]}</b>, %{customdata[1]}<br>"
                "Cluster: %{customdata[5]}<br>"
                "Sustainability: %{customdata[2]}/48<br>"
                "Fiscal Health: %{customdata[3]}<br>"
                "Population: %{customdata[4]:,}<br>"
                "Climate Networks: %{customdata[6]}<br>"
                "Pension Health: %{x:.2f}  ·  Cash Liquidity: %{y:.2f}"
                "<extra></extra>"
            ),
        ))

    # Highlighted selected cities
    _pairs: list[tuple] = []
    if r1 is not None:
        _pairs.append((r1, f"{city1}, {state1}", "#93c5fd", "star"))
    if not single_city_mode and r2 is not None:
        _pairs.append((r2, f"{city2}, {state2}", "#fbbf24", "star-diamond"))
    for row_d, nm, col_hex, sym in _pairs:
        fig_map.add_trace(go.Scatter(
                x=[row_d["PC1_pension_axis"]],
                y=[row_d["liquidity_axis"]],
                mode="markers+text",
                name=nm,
                text=[nm.split(",")[0]],
                textposition="top right",
                textfont=dict(size=11, color=col_hex, family="IBM Plex Sans"),
                marker=dict(size=21, color=col_hex, symbol=sym,
                            line=dict(width=2, color="white")),
            ))

    fig_map.update_layout(
        **base_chart_layout(height=570, margin=dict(l=55, r=25, t=28, b=58)),
        xaxis=dark_axis(
            title="← Worse pension position  |  Stronger pension position →",
            range=xr, title_font=dict(size=10),
        ),
        yaxis=dark_axis(
            title="← Lower cash reserves  |  Higher cash reserves →",
            range=yr, title_font=dict(size=10),
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True, key="map_main")

    # Bottom row: cluster donut + sustainability box
    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown("### Cities per cluster")
        cnt = df["pca_2x2_type"].value_counts().reset_index()
        cnt.columns = ["Cluster", "Count"]
        fig_donut = go.Figure(go.Pie(
            labels=[qname(c) for c in cnt["Cluster"]],
            values=cnt["Count"],
            hole=.55,
            marker_colors=[qcolor(c) for c in cnt["Cluster"]],
            textfont=dict(size=10),
            hovertemplate="%{label}<br>%{value} cities (%{percent})<extra></extra>",
        ))
        fig_donut.update_layout(
            **base_chart_layout(height=240, margin=dict(l=5, r=5, t=5, b=5)),
        )
        st.plotly_chart(fig_donut, use_container_width=True, key="donut")

    with bc2:
        if SUS_COL in df.columns:
            st.markdown("### Sustainability scores by city (selected cluster)")
            cluster_opts = [c for c in QUAD_META.keys() if c in df["pca_2x2_type"].unique()]
            state_opts = ["All states"] + sorted(
                s for s in df.loc[df["pca_2x2_type"].isin(cluster_opts), "State"].dropna().unique().tolist()
                if s not in {"Indiana", "Iowa"}
            )
            f1, f2 = st.columns(2)
            with f1:
                sel_cluster = st.selectbox(
                    "Choose cluster",
                    cluster_opts,
                    format_func=qname,
                    key="sus_cluster_select",
                )
            with f2:
                sel_state = st.selectbox(
                    "Choose state",
                    state_opts,
                    key="sus_state_select",
                )

            mask = df["pca_2x2_type"] == sel_cluster
            if sel_state != "All states":
                mask &= df["State"] == sel_state

            sub_cluster = df.loc[mask, ["city", "State", SUS_COL]].dropna()
            sub_cluster = sub_cluster.sort_values(SUS_COL, ascending=False)

            if sub_cluster.empty:
                st.caption("No sustainability scores available for this cluster/state selection.")
            else:
                sel_color = qcolor(sel_cluster)
                fig_city_cluster = go.Figure(go.Bar(
                    x=sub_cluster[SUS_COL],
                    y=sub_cluster["city"],
                    orientation="h",
                    marker_color=sel_color,
                    opacity=0.9,
                    text=[f"{v:.1f}" for v in sub_cluster[SUS_COL]],
                    textposition="outside",
                    hovertemplate="%{y}<br>Sustainability: %{x:.1f}/48<extra></extra>",
                ))
                fig_city_cluster.update_layout(
                    **base_chart_layout(
                        height=max(260, min(820, 34 * len(sub_cluster) + 60)),
                        margin=dict(l=10, r=40, t=10, b=30),
                    ),
                    xaxis=dark_axis(title="Score (/48)", range=[0, 48]),
                    yaxis=dict(
                        tickfont=dict(size=9, color="#ffffff"),
                        automargin=True,
                        categoryorder="array",
                        categoryarray=sub_cluster["city"][::-1],
                    ),
                    showlegend=False,
                )
                st.plotly_chart(
                    fig_city_cluster,
                    use_container_width=True,
                    key="bar_city_sus_cluster",
                )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2  ·  CITY vs CITY
# ──────────────────────────────────────────────────────────────────────────────
with T2:
    if r1 is None or r2 is None:
        missing = [
            f"**{city1}, {state1}**" if r1 is None else None,
            f"**{city2}, {state2}**" if r2 is None else None,
        ]
        st.warning(
            "City not found in dataset: "
            + " and ".join(m for m in missing if m)
            + ". Please check the sidebar selection."
        )
    elif single_city_mode:
        st.caption("Single-city view — comparing this municipality to itself is hidden; see scores vs. peers below.")
        render_city_card(r1, "#93c5fd")
        st.divider()
        st.markdown("### Sustainability & fiscal profile (normalized)")
        r_lbls, v1 = [], []
        for sc_col, sc_lbl, sc_max in SUS_SUBS:
            if sc_col in df.columns:
                r_lbls.append(sc_lbl)
                v1.append(float(r1.get(sc_col, 0) or 0) / sc_max)
        for fc, fl in [
            ("PC1_pension_axis", "Pension Health"),
            ("liquidity_axis",   "Cash Liquidity"),
            ("fiscal_health",    "Fiscal index (vs. peers)"),
        ]:
            mn, mx = df[fc].min(), df[fc].max()
            r_lbls.append(fl)
            v1.append((float(r1.get(fc, mn)) - mn) / (mx - mn + 1e-9))
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name=f"{city1}, {state1}",
            x=r_lbls,
            y=v1,
            marker_color="#93c5fd",
            opacity=0.84,
        ))
        fig_compare.update_layout(
            **base_chart_layout(height=370, margin=dict(l=45, r=15, t=30, b=85)),
            yaxis=dark_axis(
                title="Normalized score (0-1)",
                range=[0, 1],
                zeroline=True,
                zerolinecolor="#1a3050",
            ),
            xaxis=dict(
                tickangle=-25,
                tickfont=dict(size=9),
                gridcolor="#0f1e30",
            ),
        )
        st.plotly_chart(fig_compare, use_container_width=True, key="head_to_head_bar_single")

        st.markdown("### Financial indicators vs. peer average")
        st.caption("Positive = better than dataset average for that indicator.")
        if z_cols and all(zc in df.columns for zc in z_cols):
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name=city1,
                x=fin_labels,
                y=[float(r1[zc]) for zc in z_cols],
                marker_color="#93c5fd",
                opacity=0.82,
            ))
            fig_bar.update_layout(
                **base_chart_layout(height=320, margin=dict(l=45, r=15, t=15, b=100)),
                yaxis=dark_axis(
                    title="Deviation from average",
                    zeroline=True,
                    zerolinecolor="#1a3050",
                ),
                xaxis=dict(
                    tickangle=-38,
                    tickfont=dict(size=8),
                    gridcolor="#0f1e30",
                ),
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="fin_bar_single")
    else:
        # Side-by-side city cards
        cc1, cc2 = st.columns(2)
        with cc1:
            render_city_card(r1, "#93c5fd")
        with cc2:
            render_city_card(r2, "#fbbf24")

        st.divider()

        # ── Head-to-head comparison (normalized bar chart) ────────────────────
        st.markdown("### Head-to-head comparison")
        r_lbls, v1, v2 = [], [], []
        for sc_col, sc_lbl, sc_max in SUS_SUBS:
            if sc_col in df.columns:
                r_lbls.append(sc_lbl)
                v1.append(float(r1.get(sc_col, 0) or 0) / sc_max)
                v2.append(float(r2.get(sc_col, 0) or 0) / sc_max)
        for fc, fl in [
            ("PC1_pension_axis", "Pension Health"),
            ("liquidity_axis",   "Cash Liquidity"),
            ("fiscal_health",    "Fiscal index (vs. peers)"),
        ]:
            mn, mx = df[fc].min(), df[fc].max()
            r_lbls.append(fl)
            v1.append((float(r1.get(fc, mn)) - mn) / (mx - mn + 1e-9))
            v2.append((float(r2.get(fc, mn)) - mn) / (mx - mn + 1e-9))

        fig_compare = go.Figure()
        for vals, nm, col_hex in [
            (v1, f"{city1}, {state1}", "#93c5fd"),
            (v2, f"{city2}, {state2}", "#fbbf24"),
        ]:
            fig_compare.add_trace(go.Bar(
                name=nm,
                x=r_lbls,
                y=vals,
                marker_color=col_hex,
                opacity=.84,
            ))
        fig_compare.update_layout(
            **base_chart_layout(height=370, margin=dict(l=45, r=15, t=30, b=85)),
            barmode="group",
            yaxis=dark_axis(
                title="Normalized score (0-1)",
                range=[0, 1],
                zeroline=True,
                zerolinecolor="#1a3050",
            ),
            xaxis=dict(
                tickangle=-25,
                tickfont=dict(size=9),
                gridcolor="#0f1e30",
            ),
        )
        st.plotly_chart(fig_compare, use_container_width=True, key="head_to_head_bar")

        # ── Financial indicators bar ───────────────────────────────────────────
        st.markdown("### Financial indicators compared to peer average")
        st.caption(
            "Bars show how far each city sits above or below the dataset average "
            "for each financial indicator. Positive = better position."
        )
        if z_cols and all(zc in df.columns for zc in z_cols):
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name=city1, x=fin_labels,
                y=[float(r1[zc]) for zc in z_cols],
                marker_color="#93c5fd", opacity=.82,
            ))
            fig_bar.add_trace(go.Bar(
                name=city2, x=fin_labels,
                y=[float(r2[zc]) for zc in z_cols],
                marker_color="#fbbf24", opacity=.82,
            ))
            fig_bar.update_layout(
                **base_chart_layout(height=320, margin=dict(l=45, r=15, t=15, b=100)),
                barmode="group",
                yaxis=dark_axis(
                    title="Deviation from average",
                    zeroline=True, zerolinecolor="#1a3050",
                ),
                xaxis=dict(
                    tickangle=-38, tickfont=dict(size=8),
                    gridcolor="#0f1e30",
                ),
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="fin_bar")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3  ·  ACTIONS EXPLORER
# ──────────────────────────────────────────────────────────────────────────────
with T3:
    if df_act is None:
        st.info(
            "⬅️  Upload the **Climate Actions** Excel file in the sidebar to explore "
            "Energy, Transport, and Waste actions for each city.",
            icon="⚡",
        )
    else:
        total_a1 = len(a1)
        total_a2 = len(a2)

        # ── Sector bar chart ───────────────────────────────────────────────────
        st.markdown("### Actions by sector")
        s1 = (a1["sector"].value_counts().reindex(FOCUS_SECTORS, fill_value=0)
              if not a1.empty and "sector" in a1.columns
              else pd.Series(0, index=FOCUS_SECTORS))
        s2 = (a2["sector"].value_counts().reindex(FOCUS_SECTORS, fill_value=0)
              if not a2.empty and "sector" in a2.columns
              else pd.Series(0, index=FOCUS_SECTORS))

        fig_sec = go.Figure()
        if single_city_mode:
            fig_sec.add_trace(go.Bar(
                name=f"{city1}, {state1}",
                x=FOCUS_SECTORS,
                y=s1.values,
                marker_color="#93c5fd",
                opacity=0.85,
                text=s1.values,
                textposition="outside",
                textfont=dict(size=9, color="#93c5fd"),
            ))
            fig_sec.update_layout(
                **base_chart_layout(height=290, margin=dict(l=35, r=15, t=20, b=45)),
                yaxis=dark_axis(title="Number of actions"),
                xaxis=dict(tickfont=dict(size=11), gridcolor="#0f1e30"),
            )
            st.plotly_chart(fig_sec, use_container_width=True, key="sec_bar_single")
        else:
            for vals, nm, col_hex in [
                (s1.values, city1, "#93c5fd"),
                (s2.values, city2, "#fbbf24"),
            ]:
                fig_sec.add_trace(go.Bar(
                    name=nm, x=FOCUS_SECTORS, y=vals,
                    marker_color=col_hex, opacity=.85,
                    text=vals, textposition="outside",
                    textfont=dict(size=9, color=col_hex),
                ))
            fig_sec.update_layout(
                **base_chart_layout(height=290, margin=dict(l=35, r=15, t=20, b=45)),
                barmode="group",
                yaxis=dark_axis(title="Number of actions"),
                xaxis=dict(tickfont=dict(size=11), gridcolor="#0f1e30"),
            )
            st.plotly_chart(fig_sec, use_container_width=True, key="sec_bar")

        # ── Mini sector pies ───────────────────────────────────────────────────
        if single_city_mode:
            st.markdown(f"**{city1}, {state1}** — {total_a1} total actions")
            if not a1.empty and "sector" in a1.columns:
                sc = a1["sector"].value_counts().reset_index()
                sc.columns = ["sector", "count"]
                fig_pie = go.Figure(go.Pie(
                    labels=sc["sector"], values=sc["count"], hole=.52,
                    marker_colors=[SEC_COL.get(s, "#475569") for s in sc["sector"]],
                    textfont=dict(size=9),
                    hovertemplate="%{label}: %{value}<extra></extra>",
                ))
                fig_pie.update_layout(
                    **base_chart_layout(height=220, margin=dict(l=5, r=5, t=5, b=5)),
                )
                st.plotly_chart(fig_pie, use_container_width=True, key="pie_single_city")
            else:
                st.caption("No actions on record for this city.")
        else:
            pc1w, pc2w = st.columns(2)
            for colw, city_nm, acts in [(pc1w, city1, a1), (pc2w, city2, a2)]:
                with colw:
                    st.markdown(f"**{city_nm}** — {len(acts)} total actions")
                    if not acts.empty and "sector" in acts.columns:
                        sc = acts["sector"].value_counts().reset_index()
                        sc.columns = ["sector", "count"]
                        fig_pie = go.Figure(go.Pie(
                            labels=sc["sector"], values=sc["count"], hole=.52,
                            marker_colors=[SEC_COL.get(s, "#475569") for s in sc["sector"]],
                            textfont=dict(size=9),
                            hovertemplate="%{label}: %{value}<extra></extra>",
                        ))
                        fig_pie.update_layout(
                            **base_chart_layout(height=200, margin=dict(l=5, r=5, t=5, b=5)),
                        )
                        st.plotly_chart(fig_pie, use_container_width=True,
                                        key=f"pie_{city_nm}")
                    else:
                        st.caption("No actions on record for this city.")

        st.divider()

        # ── Sector filter + side-by-side action lists ──────────────────────────
        sel_secs = st.multiselect(
            "Filter by sector",
            FOCUS_SECTORS,
            default=FOCUS_SECTORS,
            key="t3_filter",
        )

        # Export button
        if not a1.empty or not a2.empty:
            combined = a1 if single_city_mode else pd.concat([a1, a2], ignore_index=True)
            csv_bytes = combined.to_csv(index=False).encode()
            dl_lbl = (
                f"⬇️  Download actions for {city1} (.csv)"
                if single_city_mode
                else "⬇️  Download actions for both cities (.csv)"
            )
            fn = (
                f"actions_{city1}_{state1}.csv".replace(" ", "_")
                if single_city_mode
                else f"actions_{city1}_{city2}.csv"
            )
            st.download_button(
                dl_lbl,
                data=csv_bytes,
                file_name=fn,
                mime="text/csv",
                key="dl_actions",
            )

        st.markdown("### Climate actions")
        if single_city_mode:
            render_action_list(
                f"{city1}, {state1}", a1, sel_secs, list_state_key="t3_actions_city_a",
            )
        else:
            la1, la2 = st.columns(2)
            with la1:
                render_action_list(
                    f"{city1}, {state1}", a1, sel_secs, list_state_key="t3_actions_city_a",
                )
            with la2:
                render_action_list(
                    f"{city2}, {state2}", a2, sel_secs, list_state_key="t3_actions_city_b",
                )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4  ·  CITY PROFILES
# ──────────────────────────────────────────────────────────────────────────────
with T4:
    if single_city_mode:
        if r1 is not None:
            render_full_profile(r1, "#93c5fd", fin_avail, "a")
        else:
            st.warning(f"**{city1}, {state1}** not found in the dataset.")
    else:
        p1, p2 = st.columns(2)
        with p1:
            if r1 is not None:
                render_full_profile(r1, "#93c5fd", fin_avail, "a")
            else:
                st.warning(f"**{city1}, {state1}** not found in the dataset.")
        with p2:
            if r2 is not None:
                render_full_profile(r2, "#fbbf24", fin_avail, "b")
            else:
                st.warning(f"**{city2}, {state2}** not found in the dataset.")

    # ── All-city bubble: sustainability vs fiscal health ──────────────────────
    st.divider()
    st.markdown("### All cities — sustainability vs fiscal health score")
    st.caption(
        "Bubble size = population. "
        "Hover any city to see details. "
        "Selected cities are shown as stars."
    )
    if SUS_COL in df.columns:
        fig_bub = go.Figure()
        for quad, (_, _, qcol) in QUAD_META.items():
            sub = df[df["pca_2x2_type"] == quad]
            if sub.empty:
                continue
            sizes = np.clip(
                np.sqrt(sub["population"].fillna(10_000) / 1_000) * 2.4, 4, 28
            )
            fig_bub.add_trace(go.Scatter(
                x=sub["fiscal_health"], y=sub[SUS_COL],
                mode="markers", name=qname(quad),
                marker=dict(size=sizes, color=qcol, opacity=.67,
                            line=dict(width=1, color="rgba(255,255,255,.07)")),
                customdata=np.stack([
                    sub["city"], sub["State"],
                    sub["population"].fillna(0).astype(int),
                ], axis=1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b>, %{customdata[1]}<br>"
                    "Population: %{customdata[2]:,}<br>"
                    "Fiscal Health: %{x:.2f}<br>"
                    "Sustainability: %{y}/48"
                    "<extra></extra>"
                ),
            ))
        _bub_stars = [(r1, city1, "#93c5fd")]
        if not single_city_mode:
            _bub_stars.append((r2, city2, "#fbbf24"))
        for row_d, nm, col_hex in _bub_stars:
            if row_d is not None and SUS_COL in row_d.index:
                fig_bub.add_trace(go.Scatter(
                    x=[row_d["fiscal_health"]], y=[row_d[SUS_COL]],
                    mode="markers+text", name=f"{nm}, {state1}" if row_d is r1 else f"{nm}, {state2}",
                    text=[nm], textposition="top right",
                    textfont=dict(size=10, color=col_hex),
                    marker=dict(size=20, color=col_hex, symbol="star",
                                line=dict(width=2, color="white")),
                ))
        fig_bub.update_layout(
            **base_chart_layout(height=420, margin=dict(l=50, r=25, t=15, b=50)),
            xaxis=dark_axis(title="Overall Fiscal Health (peer index)"),
            yaxis=dark_axis(title="Sustainability Score (/48)"),
        )
        st.plotly_chart(fig_bub, use_container_width=True, key="bubble_all")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5  ·  HOW CAN CITIES IMPROVE?  (peer benchmarking)
# ──────────────────────────────────────────────────────────────────────────────
with T5:
    if single_city_mode:
        st.markdown(
            '<div class="info-banner">'
            "<b>Peer learning for your selected city</b><br>"
            "Benchmarking uses the same rules as before: a higher-scoring municipality with "
            "<b>similar Overall Fiscal Health</b>, plus sector actions and recommendations."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="info-banner">'
            "<b>Peer learning & benchmarking</b><br>"
            "Each sub-tab finds a municipality with a <b>similar Overall Fiscal Health</b> "
            "index (see below) but a <b>higher sustainability score</b>, then compares climate "
            "actions and score pillars so you can see what stronger peers do differently."
            "</div>",
            unsafe_allow_html=True,
        )
    with st.expander("How is Overall Fiscal Health calculated?", expanded=False):
        st.markdown(FISCAL_HEALTH_EXPLAINER_MD)
    sel_secs_improve = st.multiselect(
        "Sectors to show in action lists",
        FOCUS_SECTORS,
        default=FOCUS_SECTORS,
        key="t5_improve_sectors",
    )
    if single_city_mode:
        render_improvement_benchmark_for_city(
            r1, city1, state1, "#93c5fd", "#34d399", sel_secs_improve, "improve_single",
        )
    else:
        sub_a, sub_b = st.tabs(["City A", "City B"])
        with sub_a:
            render_improvement_benchmark_for_city(
                r1, city1, state1, "#93c5fd", "#34d399", sel_secs_improve, "improve_a",
            )
        with sub_b:
            render_improvement_benchmark_for_city(
                r2, city2, state2, "#fbbf24", "#34d399", sel_secs_improve, "improve_b",
            )