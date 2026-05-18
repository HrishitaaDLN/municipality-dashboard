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
import json
import unicodedata
import os
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

# User-facing name for the horizontal typology axis (`PC1_pension_axis` in data — composite, not “pension only”)
PC1_AXIS_LABEL = "Long-term pressure"

# ── 2×2 typology quadrants ────────────────────────────────────────────────────
QUAD_META: dict[str, tuple[str, str, str]] = {
    # key → (display name, css-class suffix, hex colour)
    "Lower long-term pressure / High liquidity":  ("Q1 · Resilient",  "q1", "#34d399"),
    "Lower long-term pressure / Low liquidity":   ("Q2 · Stable",     "q2", "#60a5fa"),
    "Higher long-term pressure / High liquidity": ("Q3 · Pressured",  "q3", "#fbbf24"),
    "Higher long-term pressure / Low liquidity":  ("Q4 · Vulnerable", "q4", "#f87171"),
}

# ── Fallback data path (when no file is uploaded) ─────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DISK_FISCAL  = BASE_DIR / "midwest data (1).xlsx"
DISK_ACTIONS = BASE_DIR / "municipality_actions.xlsx"
PROJECT_RESULTS_DIR = BASE_DIR.parent / "Extract financial data" / "Results"
DISK_CITY_SUMMARY = PROJECT_RESULTS_DIR / "city_summary.csv"
DISK_ALL_PROJECTS = PROJECT_RESULTS_DIR / "all_projects.csv"

# ── Friendly axis/metric labels for municipality audiences ────────────────────
FRIENDLY: dict[str, str] = {
    "PC1_pension_axis": f"{PC1_AXIS_LABEL} (composite)",
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


# ── About page (methodology + typology definition) ───────────────────────────
ABOUT_PAGE_MD = """
### Municipal Fiscal & Sustainability Dashboard

This dashboard is designed to help municipalities understand how **financial capacity**, **sustainability readiness**, and **climate action activity** fit together. It brings fiscal indicators, sustainability rubric scores, planned climate actions, implemented project records, and peer-learning recommendations into one guided workflow.

---

### What You Can Do

1. **Start with a state-level view**  
   Choose Illinois, Michigan, Minnesota, or Wisconsin to see how cities in that state are distributed across fiscal clusters and sustainability score ranges.

2. **Understand city clusters**  
   Cities are grouped using two fiscal dimensions: **long-term liability** and **liquidity**. The goal is not to label cities as “good” or “bad,” but to show their relative position among peers.

3. **Select one or two cities**  
   After selecting a state, choose a single city or compare two cities from that same state. The dashboard then shows profiles, actions, implemented projects, and improvement ideas for those selections.

4. **Review planned and implemented work**  
   The Actions Explorer separates **planned actions** from sustainability reports and **implemented projects** from financial project records, grouped by **Energy**, **Transport**, and **Waste**.

5. **Generate peer-based recommendations**  
   The recommendation section finds a same-state benchmark peer with similar Overall Fiscal Health and a higher sustainability score. Gemini can then generate discussion-oriented recommendations from the data sent to it.

---

### How The Fiscal Clusters Work

The state explorer uses a 2×2 fiscal typology:

- **Long-term liability** reflects debt-service burden, pension liability / pension exposure, and related long-run balance-sheet pressure.
- **Liquidity** reflects near-term fiscal flexibility, especially Cash & Investment Coverage and Net Investment Capacity.

The dashed lines on the map are **peer-relative medians** from the uploaded dataset. They create four groups:

| Cluster | Interpretation |
|---------|----------------|
| **Q1 · Resilient** | Lower long-term liability and higher liquidity |
| **Q2 · Stable** | Lower long-term liability but lower liquidity |
| **Q3 · Pressured** | Higher long-term liability but higher liquidity |
| **Q4 · Vulnerable** | Higher long-term liability and lower liquidity |

These are **analytical peer groups**, not credit ratings, legal judgments, or official classifications.

---

### Data Used

- **Fiscal and sustainability data** come from the Midwest municipal fiscal workbook.
- **Planned actions** come from municipal sustainability action records.
- **Implemented projects** come from financial project extraction files and are matched only to cities that also exist in the Midwest fiscal dataset.
- **Peer recommendations** are AI-generated from the selected city, benchmark city, rubric scores, fiscal context, and action/project evidence.

---

### How To Interpret Results

Use this dashboard as a **decision-support and discussion tool**. It helps identify peer patterns, action gaps, and possible next steps, but it does not replace local policy review, engineering analysis, financial advising, or legal judgment.

---

### Start Exploring

Click **Continue to state explorer** below to begin with a state-level view, then select a city or two-city comparison for deeper analysis.
""".strip()

# ── Short definitions for Streamlit `help=` (?) and KPI hover titles ─────────
HELP: dict[str, str] = {
    "state_pick": "Only states that appear in your uploaded fiscal spreadsheet are listed.",
    "city_a": "Primary municipality for the profile, action list, and peer recommendations.",
    "city_b": "Second municipality from the same state for side-by-side profile, actions, and recommendations.",
    "dashboard_view": "Choose whether to analyze one city or compare two cities from the selected state.",
    "focus_state": "Guided state selection: Illinois, Michigan, Minnesota, or Wisconsin (if present in your data).",
    "focus_city": "Any city in the selected state that exists in the fiscal file.",
    "map_population": "When on, each dot’s size reflects population; when off, dots are equal size.",
    "map_labels": "Shows every city name on the map (can look crowded). Hover still works when off.",
    "map_shade": "Light background colors match the four quadrants split by the dashed median lines.",
    "map_highlight": "Cities outside the chosen state(s) fade so your selection stands out.",
    "t1_map_states": "State explorer: plot dots for the selected state. Dashed medians still use every city in your upload.",
    "sus_cluster": "Same 2×2 typology group as the map (median split on long-term pressure axis × liquidity axis).",
    "sus_state": "Limit the bar chart to one state, or show all states in the list.",
    "sector_filter": "Actions data only includes Energy, Transport, and Waste in this app.",
    "download_actions": "Downloads the action rows for the city or cities shown in this tab.",
    "improve_sectors": "Filters benchmark action lists and sector pies to the sectors you select.",
    "page_prev": "Previous page of action cards (12 per page).",
    "page_next": "Next page of action cards (12 per page).",
    "kpi_cities": "Unique municipalities in **Illinois, Michigan, Minnesota, and Wisconsin** in your file "
    "(city + state rows; duplicates collapsed).",
    "kpi_states": "How many of those **four** states appear in your fiscal upload (not Indiana/Iowa or others).",
    "kpi_sustain": "Sustainability rubric total from your spreadsheet (out of 48).",
    "imp_sus_sel": "Your city’s sustainability rubric total (from the spreadsheet, out of 48).",
    "imp_sus_bench": "The auto-picked peer city’s sustainability total for the same rubric.",
    "imp_gap": "Benchmark score minus your city’s score on the rubric total (larger = more headroom vs. this peer).",
}

# How benchmark recommendations are built (shown in “How Can Cities Improve?”)
RECOMMENDATIONS_METHOD_MD = """
**What these are**  
The **numbered peer recommendations** on this tab are produced by **Gemini** from the JSON we send (actions + rubric + fiscal context). The table below describes an **older rule-based checklist** that was removed from the UI; it is kept here only as a **reference** for how automated gap language was originally triggered.

**When a numbered line appears**  
The app checks a few **if / then** rules. Each bullet is added **only** when its trigger is true:

| Trigger | What we compare | What the text suggests |
|--------|-------------------|-------------------------|
| **Governance** | Rubric **Governance (/9)** | Benchmark is **> 0.5 points** higher than your city. |
| **Data & analytics** | **Data & Analytics (/21)** | Same **> 0.5** gap rule. |
| **Action planning** | **Action Planning (/18)** | Same **> 0.5** gap rule. |
| **Formal authority** | Sustainability **commission authority level** | Benchmark’s level is **higher** than yours (see dataset coding). |
| **Regional collaboration** | **Regional network count** in the fiscal sheet | Benchmark has **more** ties than your city. |
| **Energy / Transport / Waste** | Rows in **municipality_actions.xlsx** for that sector | Benchmark has **more listed actions** than your city in the same sector. |

**If the list is empty**  
None of the thresholds above fired, so you see one **generic** line about small gains across pillars—the benchmark still scores higher overall from the rubric total.

**Limits**  
Recommendations **do not** read action text word-for-word; they only react to **counts and rubric fields** present in your files. Update the spreadsheet to improve the quality of triggers.

**Optional — Gemini peer recommendations**  
The **Generate peer-based recommendations** control (same tab) sends full Energy/Transport/Waste action rows plus rubric/fiscal JSON for both cities to **Google Gemini**; that path is separate from this rule table. Configure a key with **`GOOGLE_API_KEY`** / **`GEMINI_API_KEY`** in the environment or Streamlit secrets (same entries as local **`.streamlit/secrets.toml`**).
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


def kpi_card(
    value: str,
    label: str,
    color: str = "#60a5fa",
    size: str = "1.6rem",
    tip: Optional[str] = None,
) -> str:
    if tip:
        safe = html.escape(tip).replace('"', "&quot;")
        label_html = (
            f'<span class="kpi-tip" title="{safe}" style="cursor:help;'
            f"border-bottom:1px dotted #64748b\">{html.escape(label)}</span>"
        )
    else:
        label_html = html.escape(label)
    return (
        f'<div class="kpi">'
        f'<div class="kpi-val" style="font-size:{size};color:{color}">{html.escape(str(value))}</div>'
        f'<div class="kpi-lbl">{label_html}</div>'
        f"</div>"
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

    # ── Horizontal typology axis: PC1 oriented so higher ⇔ lower long-term pressure
    #    (correlates PC1 with X_norm["Pension Exposure Ratio"] after sign flips).
    pen_good = (
        X_norm["Pension Exposure Ratio"]
        if "Pension Exposure Ratio" in X_norm.columns
        else pd.Series(0.0, index=df.index)
    )
    sign = 1 if np.corrcoef(df["PC1"], pen_good)[0, 1] >= 0 else -1
    df["PC1_pension_axis"] = df["PC1"] * sign

    # ── Liquidity axis: match reference notebook — take standardized cash +
    #    net investment columns, re-standardize that 2-column matrix only, mean.
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

    # ── Pre-compute z-scores (financial bars vs. peer average) ───────────────
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
        p = (
            "Lower long-term pressure"
            if row["PC1_pension_axis"] >= pc1_median
            else "Higher long-term pressure"
        )
        l = "High liquidity" if row["liquidity_axis"] >= liq_median else "Low liquidity"
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


@st.cache_data(show_spinner="Loading implemented projects…")
def load_implemented_projects(summary_raw: bytes, projects_raw: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load financial-record project extracts and normalize them to Energy/Transport/Waste sectors."""
    summary = pd.read_csv(io.BytesIO(summary_raw))
    projects = pd.read_csv(io.BytesIO(projects_raw))
    summary.columns = [str(c).strip().lower() for c in summary.columns]
    projects.columns = [str(c).strip().lower() for c in projects.columns]

    for frame in (summary, projects):
        for col in ("state", "city"):
            if col in frame.columns:
                frame[col] = frame[col].astype(str).str.strip().str.title()

    domain_to_sector = {
        "Energy Efficiency": "Energy",
        "Transport Emissions": "Transport",
        "Waste Management": "Waste",
    }
    if "domain" in projects.columns:
        projects["sector"] = projects["domain"].map(domain_to_sector)
        projects = projects[projects["sector"].isin(FOCUS_SECTORS)].copy()
    else:
        projects["sector"] = ""
        projects = projects.iloc[0:0].copy()

    for col in ("project_name", "domain", "status", "amount_raw", "description"):
        if col in projects.columns:
            projects[col] = projects[col].fillna("").astype(str).str.strip()
    if "status" in projects.columns:
        projects = projects[projects["status"].str.lower() != "planned"].copy()
    if "amount_usd" in projects.columns:
        projects["amount_usd"] = pd.to_numeric(projects["amount_usd"], errors="coerce")

    return summary.reset_index(drop=True), projects.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL RENDER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def render_full_profile(row: pd.Series, accent: str,
                        fin: list[str], key_suffix: str) -> None:
    """Detailed city profile used in the City Profiles tab."""
    st.markdown(
        f'<h2 style="color:{accent}">{row["city"]} '
        f'<span style="color:#7dd3fc;font-size:.6em;font-family:IBM Plex Sans;font-weight:500">'
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
        ("PC1_pension_axis", FRIENDLY["PC1_pension_axis"]),
        ("liquidity_axis",   FRIENDLY["liquidity_axis"]),
        ("fiscal_health",    "Overall Fiscal Health (vs. peers)"),
    ]:
        st.markdown(mrow(flbl, f"{row.get(fld, 0):.2f}"), unsafe_allow_html=True)

    # ── Governance & networks ─────────────────────────────────────────────────
    st.markdown('<div class="sec-lbl" style="margin-top:14px">Governance & networks</div>', unsafe_allow_html=True)
    cl  = int(row.get("commission_authority_level ", 0) or 0)
    rnc = int(row.get("regional_network_count",     0) or 0)
    for fv, fl in [
        (COMM_LBL.get(cl, f"Level {cl}"), "Commission authority"),
        (str(rnc), "Regional networks"),
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
        st.plotly_chart(fig, width="stretch", key=f"fin_profile_bar_{key_suffix}")


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
            if st.button(
                "◀ Prev",
                key=f"{list_state_key}_prev",
                disabled=page <= 0,
                help=HELP["page_prev"],
            ):
                st.session_state[page_key] = page - 1
                st.rerun()
        with p3:
            if st.button(
                "Next ▶",
                key=f"{list_state_key}_next",
                disabled=page >= n_pages - 1,
                help=HELP["page_next"],
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


def render_project_list(
    city_nm: str,
    projects: pd.DataFrame,
    sel_sectors: list[str],
    *,
    list_state_key: str,
) -> None:
    """Render implemented project cards from financial records."""
    filtered = (
        projects[projects["sector"].isin(sel_sectors)]
        if sel_sectors and "sector" in projects.columns
        else projects
    )

    st.markdown(
        f'<div class="sec-lbl">{city_nm} &nbsp;·&nbsp; {len(filtered)} implemented projects</div>',
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.markdown(
            '<div class="info-banner">No implemented Energy, Transport, or Waste projects '
            'from the financial project extract are on record for this city.</div>',
            unsafe_allow_html=True,
        )
        return

    total_pages = max(1, int(np.ceil(len(filtered) / ACTION_PAGE_SIZE)))
    page_key = f"{list_state_key}_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    st.session_state[page_key] = min(max(int(st.session_state[page_key]), 0), total_pages - 1)

    if total_pages > 1:
        nav_l, nav_m, nav_r = st.columns([1, 2, 1])
        with nav_l:
            if st.button("← Prev", key=f"{list_state_key}_prev", disabled=st.session_state[page_key] <= 0):
                st.session_state[page_key] -= 1
        with nav_m:
            st.caption(f"Page {st.session_state[page_key] + 1} of {total_pages}")
        with nav_r:
            if st.button("Next →", key=f"{list_state_key}_next", disabled=st.session_state[page_key] >= total_pages - 1):
                st.session_state[page_key] += 1

    start = st.session_state[page_key] * ACTION_PAGE_SIZE
    for _, row in filtered.iloc[start:start + ACTION_PAGE_SIZE].iterrows():
        name = html.escape(str(row.get("project_name", "") or "Unnamed project"))
        desc = html.escape(str(row.get("description", "") or "No description available."))
        sector = str(row.get("sector", "Other"))
        status = html.escape(str(row.get("status", "") or "Status unknown"))
        amount = row.get("amount_usd", np.nan)
        if pd.notna(amount):
            amount_txt = f"${float(amount):,.0f}"
        else:
            amount_raw = str(row.get("amount_raw", "") or "").strip()
            amount_txt = html.escape(amount_raw) if amount_raw else "Amount not stated"
        st.markdown(
            f'<div class="acard">'
            f'<div class="acard-name">{name}</div>'
            f'<div class="acard-text">{desc}</div>'
            f'<div class="acard-text" style="margin-top:8px"><b>Status:</b> {status} &nbsp;·&nbsp; '
            f'<b>Amount:</b> {amount_txt}</div>'
            f'{sector_tag_html(sector)}'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_focus_sector_pie(
    acts: pd.DataFrame,
    title_line: str,
    accent: str,
    plot_key: str,
    item_label: str = "actions",
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
        st.caption(f"No {item_label} on record for this city in the dataset.")
        return
    # One string per slice (e.g. "Transport 88"): label+value with textposition="outside"
    # can clip the number for some angles; combined labels stay readable.
    pie_labels = [f"{s} {v}" for s, v in zip(labels, vals)]
    fig_p = go.Figure(go.Pie(
        labels=pie_labels,
        values=vals,
        hole=0.52,
        marker_colors=[SEC_COL.get(s, "#475569") for s in labels],
        textinfo="label",
        textposition="outside",
        textfont=dict(size=10, color="#e2e8f0"),
        hovertemplate=f"%{{label}}<br>%{{value}} {item_label} (%{{percent}})<extra></extra>",
    ))
    fig_p.update_layout(
        **base_chart_layout(height=280, margin=dict(l=22, r=22, t=10, b=26)),
        showlegend=False,
    )
    st.plotly_chart(fig_p, width="stretch", key=plot_key)


# Default Gemini model for peer-based recommendations (override with env GEMINI_MODEL)
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
_GEMINI_ACTION_ROW_CAP = 450


def _actions_records(acts: pd.DataFrame) -> tuple[list[dict], bool]:
    """Serialize Energy/Transport/Waste action rows for LLM context (long text trimmed)."""
    if acts is None or acts.empty:
        return [], False
    cols = [c for c in ("sector", "action_name", "action") if c in acts.columns]
    if not cols:
        return [], False
    sub = acts[cols].copy()
    truncated = len(sub) > _GEMINI_ACTION_ROW_CAP
    if truncated:
        sub = sub.iloc[:_GEMINI_ACTION_ROW_CAP].copy()
    for c in sub.columns:
        if c == "action" and sub[c].dtype == object:
            sub[c] = sub[c].astype(str).str.slice(0, 1500)
        elif sub[c].dtype == object:
            sub[c] = sub[c].astype(str).str.slice(0, 400)
    return sub.to_dict(orient="records"), truncated


def _rubric_fiscal_snapshot(row: pd.Series) -> dict:
    """Sustainability pillars + key fiscal indices for one municipality."""
    snap: dict = {}
    for sc_col, sc_lbl, sc_max in SUS_SUBS:
        if sc_col in row.index:
            key = sc_lbl.split("(")[0].strip().replace("\n", " ")
            snap[f"{key}_out_of_{sc_max}"] = float(
                pd.to_numeric(row.get(sc_col), errors="coerce") or 0.0
            )
    if SUS_COL in row.index:
        snap["total_sustainability_out_of_48"] = float(
            pd.to_numeric(row.get(SUS_COL), errors="coerce") or 0.0
        )
    for k_py, k_out in [
        ("fiscal_health", "overall_fiscal_health_peer_index"),
        ("PC1_pension_axis", "long_term_pressure_axis_composite"),
        ("liquidity_axis", "liquidity_axis"),
    ]:
        if k_py in row.index:
            snap[k_out] = float(pd.to_numeric(row.get(k_py), errors="coerce") or 0.0)
    if "population" in row.index:
        snap["population"] = int(pd.to_numeric(row.get("population"), errors="coerce") or 0)
    if "pca_2x2_type" in row.index:
        snap["fiscal_typology_quadrant"] = str(row.get("pca_2x2_type") or "")
    cl = 0
    for k in ("commission_authority_level ", "commission_authority_level"):
        if k in row.index:
            cl = int(pd.to_numeric(row.get(k), errors="coerce") or 0)
            break
    snap["sustainability_commission_authority"] = COMM_LBL.get(cl, str(cl))
    if "regional_network_count" in row.index:
        snap["regional_network_count"] = int(
            pd.to_numeric(row.get("regional_network_count"), errors="coerce") or 0
        )
    return snap


def _resolve_gemini_api_key() -> str:
    """Environment variables → Streamlit secrets. Returns '' if unset (never raises)."""
    for env_k in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        v = (os.environ.get(env_k) or "").strip()
        if v:
            return v
    try:
        sec = getattr(st, "secrets", None)
        if sec is not None:
            for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
                v = sec.get(k)
                if v:
                    return str(v).strip()
    except Exception:
        pass
    return ""


def _gemini_build_peer_prompt(payload: dict) -> str:
    return (
        "You are helping U.S. municipal sustainability and finance staff.\n\n"
        "Two cities are paired: **similar Overall Fiscal Health** (peer fiscal index from the spreadsheet) "
        "but the **benchmark city has a higher total sustainability score**. The **selected city** should "
        "learn from the **benchmark (peer) city**.\n\n"
        "**Your task:** Write **8–14 numbered recommendations** for the **selected city**. Each recommendation "
        "must be **strictly grounded in the benchmark city’s evidence** in the JSON below—its listed "
        "Energy/Transport/Waste actions and/or its higher Governance, Data & Analytics, or Action Planning scores "
        "relative to the selected city. **Do not** invent benchmark programs that are not supported by that "
        "evidence. You may contrast with the selected city’s data, but **every actionable idea must reflect what "
        "the benchmark actually documents or scores well on**.\n\n"
        "Style: short bullets, concrete, no legal advice, no long quotations. If an action field is empty, say so "
        "and lean on rubric-score gaps instead.\n\n"
        "**Data (JSON):**\n```json\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
        + "\n```"
    )


def _gemini_generate_peer_recommendations(prompt: str, api_key: str) -> str:
    try:
        import google.generativeai as genai  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError(
            "Install **google-generativeai**: `pip install google-generativeai` "
            "(see requirements.txt)."
        ) from e

    key = (api_key or "").strip()
    if not key:
        raise RuntimeError(
            "Missing API key: set **GOOGLE_API_KEY** or **GEMINI_API_KEY** in the environment "
            "or Streamlit secrets (same keys as in local **.streamlit/secrets.toml**)."
        )
    model_id = os.environ.get("GEMINI_MODEL", GEMINI_DEFAULT_MODEL).strip() or GEMINI_DEFAULT_MODEL
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_id)
    resp = model.generate_content(
        prompt,
        generation_config={"temperature": 0.35, "max_output_tokens": 4096},
    )
    if not resp.candidates:
        raise RuntimeError("Gemini returned no candidates (blocked or empty).")
    text = getattr(resp, "text", None) or ""
    if not text.strip():
        # Fallback: stitch parts if .text missing
        parts: list[str] = []
        for cand in resp.candidates:
            for part in getattr(cand.content, "parts", []) or []:
                if hasattr(part, "text") and part.text:
                    parts.append(part.text)
        text = "\n".join(parts) if parts else "(Empty model response)"
    return text


def _slug_peer_rec_pdf_filename(city_nm: str, state_nm: str) -> str:
    """ASCII-safe filename: City_State_peer_recommendations.pdf."""

    def _slug(part: str, cap: int = 40) -> str:
        p = str(part or "").strip()
        p = unicodedata.normalize("NFKD", p)
        p = p.encode("ascii", "ignore").decode("ascii")
        p = re.sub(r"[^\w\-.]+", "_", p, flags=re.ASCII)
        p = re.sub(r"_+", "_", p).strip("_") or "city"
        return p[:cap]

    return f"{_slug(city_nm)}_{_slug(state_nm)}_peer_recommendations.pdf"


def _peer_rec_md_line_to_reportlab_xml(line: str) -> str:
    """Turn one line of Gemini markdown into ReportLab Paragraph XML (**bold** only)."""
    parts = line.split("**")
    out: list[str] = []
    for i, ch in enumerate(parts):
        esc = html.escape(ch, quote=False)
        if i % 2 == 1:
            out.append(f"<b>{esc}</b>")
        else:
            out.append(esc)
    return "".join(out)


def _build_peer_recommendations_pdf_bytes(
    city_nm: str,
    state_nm: str,
    bench_label: str,
    sus_f: float,
    sus_b: float,
    fh_f: float,
    fh_b: float,
    body_markdown: str,
) -> bytes:
    try:
        from datetime import date

        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer
    except ImportError as e:
        raise RuntimeError(
            "Install **reportlab**: `pip install reportlab` (see requirements.txt)."
        ) from e

    buf = io.BytesIO()
    margin = 54
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        rightMargin=margin,
        leftMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
        title=f"Peer recommendations — {city_nm}",
        author="Municipal Sustainability Dashboard",
    )
    page_w = LETTER[0]
    rule_w = page_w - 2 * margin

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="PeerPdfTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
        leading=22,
    )
    subtitle_style = ParagraphStyle(
        name="PeerPdfSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=13,
        textColor=colors.HexColor("#1e3a5f"),
        spaceAfter=14,
        leading=16,
    )
    meta_style = ParagraphStyle(
        name="PeerPdfMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        textColor=colors.HexColor("#475569"),
        spaceAfter=5,
        leading=12,
    )
    body_style = ParagraphStyle(
        name="PeerPdfBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=9,
        leading=15,
        alignment=TA_LEFT,
    )
    foot_style = ParagraphStyle(
        name="PeerPdfFoot",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        textColor=colors.HexColor("#64748b"),
        spaceBefore=14,
        leading=12,
    )

    story: list = []
    story.append(Paragraph("Peer-based recommendations", title_style))
    story.append(
        Paragraph(
            f"{html.escape(city_nm, quote=False)}, {html.escape(state_nm, quote=False)}",
            subtitle_style,
        )
    )
    story.append(
        Paragraph(
            f"<b>Benchmark peer:</b> {html.escape(bench_label, quote=False)}",
            meta_style,
        )
    )
    story.append(
        Paragraph(
            f"Sustainability (out of 48): {sus_f:.1f} (selected) vs {sus_b:.1f} (benchmark) &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Overall fiscal health index: {fh_f:.2f} vs {fh_b:.2f}",
            meta_style,
        )
    )
    story.append(
        Paragraph(f"Generated {date.today().isoformat()}", meta_style),
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        HRFlowable(
            width=rule_w,
            thickness=0.75,
            color=colors.HexColor("#94a3b8"),
            spaceAfter=14,
        )
    )

    for raw_line in (body_markdown or "").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            story.append(Spacer(1, 4))
            continue
        story.append(Paragraph(_peer_rec_md_line_to_reportlab_xml(line), body_style))

    story.append(
        Paragraph(
            "Draft for discussion only — not legal, financial, or engineering advice. "
            "Text was produced by an AI model from the peer comparison data summarized above.",
            foot_style,
        )
    )
    doc.build(story)
    return buf.getvalue()


def pick_fiscal_peer_benchmark(dframe: pd.DataFrame, focal: pd.Series) -> Optional[pd.Series]:
    """
    Choose a same-state benchmark municipality with similar Overall Fiscal Health index
    (`fiscal_health`: peer-average financial indicators) but a meaningfully
    higher sustainability total score.
    """
    if focal is None or SUS_COL not in dframe.columns:
        return None
    focal_state = str(focal.get("State", "") or "")
    state_pool = dframe[dframe["State"] == focal_state].copy()
    if state_pool.empty:
        return None
    fh_t = float(pd.to_numeric(focal.get("fiscal_health"), errors="coerce") or 0.0)
    sus_t = float(pd.to_numeric(focal.get(SUS_COL), errors="coerce") or 0.0)
    fh_std = float(state_pool["fiscal_health"].std() or 0.25)
    if np.isnan(fh_std) or fh_std < 1e-6:
        fh_std = 0.25

    m_self = (state_pool["city"] == focal["city"]) & (state_pool["State"] == focal["State"])
    pool = state_pool.loc[~m_self].copy()
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
            return state_pool.loc[ix]

    better_all = pool[pool["_sus"] > sus_t + 0.08]
    if not better_all.empty:
        ix = better_all.sort_values(["_dfh", "_sus"], ascending=[True, False]).index[0]
        return state_pool.loc[ix]
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
            f"No same-state benchmark peer found for **{city_nm}, {state_nm}** with a higher "
            "sustainability score in this dataset. Try another city, or this city may already "
            "rank at the top among comparable peers in the selected state."
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
        f"financial fields. "
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
            kpi_card(
                f"{sus_f:.1f}/48",
                f"{city_nm} — sustainability",
                accent_sel,
                "1.25rem",
                tip=HELP["imp_sus_sel"],
            ),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            kpi_card(
                f"{sus_b:.1f}/48",
                "Benchmark — sustainability",
                accent_bench,
                "1.25rem",
                tip=HELP["imp_sus_bench"],
            ),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            kpi_card(
                f"+{sus_b - sus_f:.1f}",
                "Score gap (benchmark − selected)",
                "#34d399",
                "1.25rem",
                tip=HELP["imp_gap"],
            ),
            unsafe_allow_html=True,
        )

    with st.expander(f"View benchmark city profile: {bench_label}", expanded=False):
        render_full_profile(bench, accent_bench, fin_avail, f"bench_profile_{chart_key_suffix}")

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
    st.plotly_chart(fig_sub, width="stretch", key=f"bench_subscores_{chart_key_suffix}")

    gemini_key = f"peer_gemini_recs_{chart_key_suffix}"
    resolved_key = _resolve_gemini_api_key()
    api_ready = bool(resolved_key)

    af, tf = _actions_records(acts_f)
    ab, tb = _actions_records(acts_b)
    peer_payload = {
        "match_summary": {
            "selected_city": f"{city_nm}, {state_nm}",
            "benchmark_city": bench_label,
            "overall_fiscal_health_peer_index_selected": fh_f,
            "overall_fiscal_health_peer_index_benchmark": fh_b,
            "abs_fiscal_health_gap": dfh,
            "total_sustainability_out_of_48_selected": sus_f,
            "total_sustainability_out_of_48_benchmark": sus_b,
            "gap_points_benchmark_minus_selected": sus_b - sus_f,
        },
        "selected_city": {
            "rubric_governance_data_analytics_action_planning_and_fiscal": _rubric_fiscal_snapshot(focal),
            "documented_actions_energy_transport_waste": af,
            "action_rows_omitted_after_cap": tf,
        },
        "benchmark_city_peer": {
            "rubric_governance_data_analytics_action_planning_and_fiscal": _rubric_fiscal_snapshot(bench),
            "documented_actions_energy_transport_waste": ab,
            "action_rows_omitted_after_cap": tb,
        },
    }

    if api_ready:
        st.caption(
            "Note: these recommendations are AI-generated from the city, benchmark, action, rubric, "
            "and fiscal data sent to Gemini. Treat them as discussion starters, not final policy, "
            "legal, financial, or engineering advice."
        )
        if st.button(
            "Generate peer-based recommendations",
            key=f"btn_{gemini_key}",
            help="Calls Google Gemini when GOOGLE_API_KEY or GEMINI_API_KEY is set (environment or Streamlit secrets).",
        ):
            try:
                prompt = _gemini_build_peer_prompt(peer_payload)
                use_key = _resolve_gemini_api_key()
                with st.spinner("Calling Gemini…"):
                    st.session_state[gemini_key] = _gemini_generate_peer_recommendations(prompt, use_key)
                st.session_state.pop(f"{gemini_key}_err", None)
            except Exception as exc:  # noqa: BLE001 — surface API/config errors in UI
                st.session_state[gemini_key] = None
                st.session_state[f"{gemini_key}_err"] = str(exc)
        err_key = f"{gemini_key}_err"
        if st.session_state.get(err_key) and not st.session_state.get(gemini_key):
            st.error(st.session_state[err_key])
        elif st.session_state.get(gemini_key):
            rec_md = st.session_state[gemini_key]
            st.markdown(rec_md)
            try:
                pdf_bytes = _build_peer_recommendations_pdf_bytes(
                    city_nm=city_nm,
                    state_nm=state_nm,
                    bench_label=bench_label,
                    sus_f=sus_f,
                    sus_b=sus_b,
                    fh_f=fh_f,
                    fh_b=fh_b,
                    body_markdown=str(rec_md),
                )
                st.download_button(
                    "Download recommendations as PDF",
                    data=pdf_bytes,
                    file_name=_slug_peer_rec_pdf_filename(city_nm, state_nm),
                    mime="application/pdf",
                    key=f"dl_peer_pdf_{gemini_key}",
                    help="Clean one-page layout with benchmark context; file name uses the selected city.",
                )
            except Exception as pdf_exc:  # noqa: BLE001 — surface layout/deps errors without losing markdown
                st.caption(f"PDF export unavailable: {pdf_exc}")
    else:
        st.caption(
            "**Peer recommendations (Gemini) are optional** — the rest of the dashboard works without a key. "
            "To use that feature, set **GOOGLE_API_KEY** or **GEMINI_API_KEY** in the environment or Streamlit "
            "secrets (same as local **.streamlit/secrets.toml**). **GEMINI_MODEL** is optional "
            "(default `gemini-2.5-flash`)."
        )


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

# ── Load implemented projects from financial extraction outputs ───────────────
df_city_summary: Optional[pd.DataFrame] = None
df_projects: Optional[pd.DataFrame] = None
try:
    with open(DISK_CITY_SUMMARY, "rb") as fh:
        city_summary_bytes = fh.read()
    with open(DISK_ALL_PROJECTS, "rb") as fh:
        all_projects_bytes = fh.read()
    df_city_summary, df_projects = load_implemented_projects(city_summary_bytes, all_projects_bytes)
    valid_pairs = frozenset(
        zip(df["city"].str.strip().str.lower(), df["State"].str.strip().str.lower())
    )
    summary_pairs = frozenset(
        zip(df_city_summary["city"].str.strip().str.lower(), df_city_summary["state"].str.strip().str.lower())
    )
    keep_projects = df_projects.apply(
        lambda r: (
            str(r["city"]).lower(),
            str(r["state"]).lower(),
        ) in valid_pairs and (
            str(r["city"]).lower(),
            str(r["state"]).lower(),
        ) in summary_pairs,
        axis=1,
    )
    df_projects = df_projects[keep_projects].reset_index(drop=True)
except FileNotFoundError:
    st.warning(f"Implemented project files not found: {DISK_CITY_SUMMARY} / {DISK_ALL_PROJECTS}")
except Exception as exc:
    st.warning(f"Could not load implemented project data: {exc}")

# ── Pre-compute derived lists used in multiple tabs ───────────────────────────
fin_avail  = [c for c in FIN_COLS if c in df.columns]
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


def municipality_row_count(frame: pd.DataFrame) -> int:
    """Rows in frame; collapse duplicate (city, State) pairs so counts match unique municipalities."""
    if frame is None or frame.empty:
        return 0
    if "city" in frame.columns and "State" in frame.columns:
        return int(frame.drop_duplicates(subset=["city", "State"]).shape[0])
    return len(frame)


_core_states_in_file = frozenset(s for s in FOCUS_STATES_ORDER if s in set(df["State"].dropna()))
if _core_states_in_file:
    _df_kpi_scope = df[df["State"].isin(_core_states_in_file)]
    KPI_CITIES = municipality_row_count(_df_kpi_scope)
    KPI_STATES = int(_df_kpi_scope["State"].nunique())
else:
    KPI_CITIES = municipality_row_count(df)
    KPI_STATES = int(df["State"].nunique())

# Plotly toolbar: scroll-wheel zoom + double-click reset (typology map)
PLOTLY_TYPOLOGY_CONFIG: dict = {"scrollZoom": True, "displayModeBar": True, "doubleClick": "reset"}


# ══════════════════════════════════════════════════════════════════════════════
# GUIDED FLOW — About → State Explorer → City Selection → Analysis
# ══════════════════════════════════════════════════════════════════════════════

GUIDED_ABOUT = "about"
GUIDED_STATE = "state"
GUIDED_CITY = "city"
GUIDED_COMPARE_SINGLE = "single"
GUIDED_COMPARE_TWO = "two"


def cities_for(state: str) -> list[str]:
    return sorted(df[df["State"] == state]["city"].dropna().unique().tolist())


def _default(key: str, value: str) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def _first_or_blank(items: list[str]) -> str:
    return items[0] if items else ""


def _second_or_first(items: list[str]) -> str:
    if len(items) > 1:
        return items[1]
    return _first_or_blank(items)


guided_states = [s for s in FOCUS_STATES_ORDER if s in df["State"].dropna().unique().tolist()]
if not guided_states:
    guided_states = [s for s in all_states if s not in {"Indiana", "Iowa"}] or all_states

_default("guided_step", GUIDED_ABOUT)
_default("guided_state", "Illinois" if "Illinois" in guided_states else _first_or_blank(guided_states))
if st.session_state.get("guided_state") not in guided_states:
    st.session_state["guided_state"] = _first_or_blank(guided_states)

_guided_city_init = cities_for(st.session_state["guided_state"])
_default("guided_compare_mode", GUIDED_COMPARE_SINGLE)
_default("guided_city_a", _first_or_blank(_guided_city_init))
_default("guided_city_b", _second_or_first(_guided_city_init))


def _sync_guided_cities_for_state() -> None:
    city_opts = cities_for(st.session_state["guided_state"])
    if st.session_state.get("guided_city_a") not in city_opts:
        st.session_state["guided_city_a"] = _first_or_blank(city_opts)
    if st.session_state.get("guided_city_b") not in city_opts:
        st.session_state["guided_city_b"] = _second_or_first(city_opts)


def _guided_state_changed() -> None:
    _sync_guided_cities_for_state()
    st.session_state["guided_step"] = GUIDED_STATE


def _go_to_state_explorer() -> None:
    st.session_state["guided_step"] = GUIDED_STATE


def _go_to_city_selection() -> None:
    _sync_guided_cities_for_state()
    st.session_state["guided_step"] = GUIDED_CITY


def _go_to_about() -> None:
    st.session_state["guided_step"] = GUIDED_ABOUT


def _go_back_to_state() -> None:
    st.session_state["guided_step"] = GUIDED_STATE


_sync_guided_cities_for_state()


def get_row(city: str, state: str) -> Optional[pd.Series]:
    mask = (df["city"] == city) & (df["State"] == state)
    return df[mask].iloc[0] if mask.any() else None


def get_city_actions(city: str, state: str) -> pd.DataFrame:
    if df_act is None:
        return pd.DataFrame()
    return df_act[
        (df_act["city"].str.lower() == city.lower()) &
        (df_act["state"].str.lower() == state.lower())
    ].copy()


def get_city_projects(city: str, state: str) -> pd.DataFrame:
    if df_projects is None:
        return pd.DataFrame()
    return df_projects[
        (df_projects["city"].str.lower() == city.lower()) &
        (df_projects["state"].str.lower() == state.lower())
    ].copy()


def _render_guided_header(
    *,
    selected_state: str,
    city1: str = "",
    city2: str = "",
    single_city_mode: bool = True,
) -> None:
    st.markdown("# Municipal Fiscal & Sustainability Dashboard")
    if city1:
        if single_city_mode:
            suffix = (
                f'<b>Selected city:</b> <span style="color:#93c5fd">{city1}, {selected_state}</span>'
            )
        else:
            suffix = (
                f'Comparing <b style="color:#93c5fd">{city1}, {selected_state}</b>'
                f' &nbsp;vs&nbsp; <b style="color:#fbbf24">{city2}, {selected_state}</b>'
            )
    elif selected_state:
        suffix = f'<b>State explorer:</b> <span style="color:#93c5fd">{selected_state}</span>'
    else:
        suffix = "Start with the About page"
    st.markdown(
        f'<p style="color:#ffffff;font-size:.76rem;margin-top:-10px">'
        f"{KPI_CITIES} cities &nbsp;·&nbsp; {KPI_STATES} states &nbsp;·&nbsp; {suffix}"
        f"</p>",
        unsafe_allow_html=True,
    )


def _render_guided_kpis(
    selected_state: str,
    r1: Optional[pd.Series] = None,
    r2: Optional[pd.Series] = None,
    *,
    single_city_mode: bool = True,
) -> None:
    state_df = df[df["State"] == selected_state] if selected_state else df.iloc[0:0]
    avg_sus = float(pd.to_numeric(state_df.get(SUS_COL), errors="coerce").mean() or 0.0) if SUS_COL in state_df else 0.0
    city_a_cluster = qname(str(r1.get("pca_2x2_type", "—"))) if r1 is not None else "—"
    city_b_cluster = qname(str(r2.get("pca_2x2_type", "—"))) if r2 is not None else "—"
    if r1 is None:
        k_cols = st.columns(3)
        values = [
            (str(municipality_row_count(state_df)), f"{selected_state} cities", "kpi_cities"),
            (str(state_df["pca_2x2_type"].nunique() if not state_df.empty else 0), "Clusters represented", "sus_cluster"),
            (f"{avg_sus:.1f}/48", "Average sustainability", "kpi_sustain"),
        ]
    elif single_city_mode:
        sus_a = float(pd.to_numeric(r1.get(SUS_COL), errors="coerce") or 0.0) if SUS_COL in r1 else 0.0
        k_cols = st.columns(3)
        values = [
            (str(municipality_row_count(state_df)), f"{selected_state} cities", "kpi_cities"),
            (f"{sus_a:.1f}/48", f"{r1['city']} sustainability", "kpi_sustain"),
            (city_a_cluster, "City cluster", "sus_cluster"),
        ]
    else:
        sus_a = float(pd.to_numeric(r1.get(SUS_COL), errors="coerce") or 0.0) if SUS_COL in r1 else 0.0
        sus_b = float(pd.to_numeric(r2.get(SUS_COL), errors="coerce") or 0.0) if r2 is not None and SUS_COL in r2 else 0.0
        k_cols = st.columns(4)
        values = [
            (str(municipality_row_count(state_df)), f"{selected_state} cities", "kpi_cities"),
            (f"{avg_sus:.1f}/48", "State average", "kpi_sustain"),
            (f"{sus_a:.1f}/48", f"{r1['city']} sustainability", "kpi_sustain"),
            (f"{sus_b:.1f}/48", f"{r2['city']} sustainability", "kpi_sustain"),
        ]
    for col_w, (v, lbl, hk) in zip(k_cols, values):
        with col_w:
            st.markdown(kpi_card(v, lbl, tip=HELP.get(hk, "")), unsafe_allow_html=True)
    if r1 is not None and not single_city_mode and r2 is not None:
        st.markdown(
            f"""
            <div class="info-banner" style="margin-top:10px;padding:10px 14px">
              <b>City clusters</b><br>
              <span style="color:#93c5fd;font-weight:700">{html.escape(str(r1["city"]))}</span>:
              <span class="badge b{qcss(str(r1.get("pca_2x2_type", "")))}" style="font-size:.72rem;padding:3px 10px;margin-left:6px">
                {html.escape(city_a_cluster)}
              </span>
              <span style="display:inline-block;width:18px"></span>
              <span style="color:#fbbf24;font-weight:700">{html.escape(str(r2["city"]))}</span>:
              <span class="badge b{qcss(str(r2.get("pca_2x2_type", "")))}" style="font-size:.72rem;padding:3px 10px;margin-left:6px">
                {html.escape(city_b_cluster)}
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_cluster_explainer() -> None:
    st.markdown(
        '<div class="info-banner">'
        "<b>How to read the clusters</b><br>"
        "<b>Resilient (Q1):</b> lower long-term liability and higher liquidity.<br>"
        "<b>Stable (Q2):</b> lower long-term liability, but tighter liquidity.<br>"
        "<b>Pressured (Q3):</b> higher long-term liability, but currently stronger liquidity.<br>"
        "<b>Vulnerable (Q4):</b> higher long-term liability and lower liquidity.<br><br>"
        "The dashed medians are calculated from all cities in the uploaded file, so each state is still "
        "being compared to the same peer-relative typology."
        "</div>",
        unsafe_allow_html=True,
    )


def _render_state_cluster_scores(selected_state: str) -> None:
    state_df = df[df["State"] == selected_state].copy()
    if state_df.empty:
        st.warning(f"No cities found for **{selected_state}**.")
        return

    st.markdown("### City scores by cluster")
    st.caption("Bars show each city's sustainability rubric total, grouped by its fiscal typology cluster.")
    for quad, (cluster_name, _, cluster_color) in QUAD_META.items():
        sub = state_df[state_df["pca_2x2_type"] == quad].copy()
        if sub.empty:
            continue
        if SUS_COL in sub.columns:
            sub[SUS_COL] = pd.to_numeric(sub[SUS_COL], errors="coerce").fillna(0)
            sub = sub.sort_values(SUS_COL, ascending=False)
            values = sub[SUS_COL]
            text = [f"{v:.1f}" for v in values]
        else:
            sub = sub.sort_values("city")
            values = pd.Series([0.0] * len(sub), index=sub.index)
            text = ["N/A"] * len(sub)
        st.markdown(f"#### {cluster_name}")
        fig = go.Figure(go.Bar(
            x=values,
            y=sub["city"],
            orientation="h",
            marker_color=cluster_color,
            opacity=0.9,
            text=text,
            textposition="outside",
            hovertemplate="%{y}<br>Sustainability: %{x:.1f}/48<extra></extra>",
        ))
        fig.update_layout(
            **base_chart_layout(
                height=max(220, min(680, 32 * len(sub) + 70)),
                margin=dict(l=10, r=42, t=10, b=30),
            ),
            xaxis=dark_axis(title="Sustainability score (/48)", range=[0, 48]),
            yaxis=dict(
                tickfont=dict(size=9, color="#ffffff"),
                automargin=True,
                categoryorder="array",
                categoryarray=sub["city"][::-1],
            ),
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch", key=f"guided_scores_{selected_state}_{quad}")


def _render_state_sustainability_quartiles(selected_state: str) -> None:
    state_df = df[df["State"] == selected_state].copy()
    if state_df.empty or SUS_COL not in state_df.columns:
        return

    scores = pd.to_numeric(state_df[SUS_COL], errors="coerce")
    sub = state_df.loc[scores.notna(), ["city", "State", SUS_COL]].copy()
    sub[SUS_COL] = scores.loc[scores.notna()].astype(float)
    if sub.empty:
        return

    st.markdown("### Sustainability score quartiles")
    q_col, exp_col = st.columns([2.6, 1])
    with exp_col:
        with st.expander("What does this plot mean?", expanded=False):
            st.markdown(
                "This groups cities in the selected state into **four equal-sized score bands**.\n\n"
                "- **Quartile 1**: lower-scoring cities in this state.\n"
                "- **Quartile 4**: higher-scoring cities in this state.\n"
                "- The **box** shows the middle spread of scores in that quartile.\n"
                "- The **highlighted line/point** is the **median**, or typical city score in that quartile.\n\n"
                "For municipalities, this helps show whether a city is near the lower, middle, or higher end "
                "of sustainability performance within its own state."
            )

    try:
        sub["_quartile_num"] = pd.qcut(sub[SUS_COL], q=4, labels=False, duplicates="drop") + 1
    except ValueError:
        sub["_quartile_num"] = 1
    sub["_quartile_num"] = pd.to_numeric(sub["_quartile_num"], errors="coerce").fillna(1).astype(int)
    sub["_quartile"] = sub["_quartile_num"].map(lambda n: f"Quartile {int(n)}")

    with q_col:
        fig = go.Figure()
        quartiles = [f"Quartile {i}" for i in sorted(sub["_quartile_num"].unique())]
        for q in quartiles:
            qsub = sub[sub["_quartile"] == q]
            fig.add_trace(go.Box(
                x=[q] * len(qsub),
                y=qsub[SUS_COL],
                name=q,
                boxpoints="all",
                jitter=0.35,
                pointpos=0,
                marker=dict(size=7, color="#60a5fa", opacity=0.72),
                line=dict(color="#93c5fd"),
                fillcolor="rgba(96,165,250,.18)",
                hovertext=qsub["city"],
                hovertemplate="<b>%{hovertext}</b><br>%{x}<br>Score: %{y:.1f}/48<extra></extra>",
                showlegend=False,
            ))
            med = float(qsub[SUS_COL].median())
            fig.add_trace(go.Scatter(
                x=[q],
                y=[med],
                mode="markers+text",
                marker=dict(size=12, color="#fbbf24", symbol="diamond", line=dict(width=1, color="#ffffff")),
                text=[f"Median {med:.1f}"],
                textposition="top center",
                textfont=dict(size=10, color="#fbbf24"),
                hovertemplate=f"{q}<br>Median: {med:.1f}/48<extra></extra>",
                showlegend=False,
            ))
        fig.update_layout(
            **base_chart_layout(height=360, margin=dict(l=45, r=20, t=20, b=45)),
            xaxis=dict(title="Quartile", tickfont=dict(size=11, color="#ffffff"), gridcolor="#0f1e30"),
            yaxis=dark_axis(title="Sustainability score (/48)", range=[0, 48]),
        )
        st.plotly_chart(fig, width="stretch", key=f"guided_quartile_box_{selected_state}")


def _render_state_typology_map(selected_state: str) -> None:
    state_df = df[df["State"] == selected_state].copy()
    if state_df.empty:
        return
    st.markdown("### Fiscal typology map for this state")
    st.caption(
        "This map groups cities by long-term liability and liquidity using peer-relative median lines."
    )
    with st.expander("How are these clusters calculated?", expanded=False):
        st.markdown(
            "**Long-term liability** uses the fiscal liability side of the workbook, especially "
            "**debt service burden** and **pension liability / pension exposure** signals. "
            "Cities to the right have a lighter long-term liability position relative to peers.\n\n"
            "**Liquidity** uses near-term balance-sheet flexibility: **Cash & Investment Coverage** "
            "and **Net Investment Capacity**. Cities higher on the chart have stronger liquidity "
            "relative to peers.\n\n"
            "**Clusters** are assigned by the median lines: **Q1 · Resilient** = lower liability + "
            "higher liquidity; **Q2 · Stable** = lower liability + lower liquidity; "
            "**Q3 · Pressured** = higher liability + higher liquidity; **Q4 · Vulnerable** = "
            "higher liability + lower liquidity."
        )
    m1, m2 = st.columns(2)
    with m1:
        show_pop = st.checkbox(
            "Scale bubbles by population",
            value=True,
            key="guided_show_pop",
            help=HELP["map_population"],
        )
    with m2:
        show_labels = st.checkbox(
            "Show all city labels",
            value=False,
            key="guided_show_lbl",
            help=HELP["map_labels"],
        )
    fig = go.Figure()
    for quad, (cluster_name, _, cluster_color) in QUAD_META.items():
        sub = state_df[state_df["pca_2x2_type"] == quad]
        if sub.empty:
            continue
        sizes = (
            np.clip(np.sqrt(sub["population"].fillna(50_000) / 1_000) * 3.2, 6, 26)
            if show_pop else np.full(len(sub), 10.0)
        )
        sus_vals = sub.get(SUS_COL, pd.Series(np.nan, index=sub.index)).fillna(0).round(1)
        fig.add_trace(go.Scatter(
            x=sub["PC1_pension_axis"],
            y=sub["liquidity_axis"],
            mode="markers+text" if show_labels else "markers",
            name=cluster_name,
            text=sub["city"] if show_labels else None,
            textposition="top center",
            textfont=dict(size=7, color="#ffffff"),
            marker=dict(
                size=sizes,
                color=cluster_color,
                opacity=0.78,
                line=dict(width=1, color="rgba(255,255,255,.14)"),
            ),
            customdata=np.stack([
                sub["city"],
                sub["State"],
                sus_vals,
                sub["fiscal_health"].round(2),
                pd.Series([qname(str(v)) for v in sub["pca_2x2_type"]], index=sub.index),
            ], axis=1),
            hovertemplate=(
                "<b>%{customdata[0]}</b>, %{customdata[1]}<br>"
                "Cluster: %{customdata[4]}<br>"
                "Sustainability: %{customdata[2]}/48<br>"
                "Fiscal Health: %{customdata[3]}<br>"
                "Long-term liability: %{x:.2f} · Liquidity: %{y:.2f}"
                "<extra></extra>"
            ),
        ))
    fig.add_hline(y=liq_median, line_dash="dot", line_color="#18243a", line_width=1.5)
    fig.add_vline(x=pc1_median, line_dash="dot", line_color="#18243a", line_width=1.5)
    fig.update_layout(
        title=dict(
            text=f"{selected_state}: city clusters by long-term liability and liquidity",
            font=dict(size=13, color="#f8fafc", family="IBM Plex Sans"),
            x=0.5,
            xanchor="center",
        ),
        **base_chart_layout(height=470, margin=dict(l=55, r=25, t=48, b=58)),
        xaxis=dark_axis(title="Long-term liability"),
        yaxis=dark_axis(title="Liquidity"),
    )
    st.plotly_chart(fig, width="stretch", key=f"guided_typology_{selected_state}", config=PLOTLY_TYPOLOGY_CONFIG)


def _render_state_explorer(selected_state: str) -> None:
    st.markdown("## Step 2: Choose a state and review city clusters")
    st.selectbox(
        "State",
        guided_states,
        key="guided_state",
        on_change=_guided_state_changed,
        help=HELP["state_pick"],
    )
    selected_state = st.session_state["guided_state"]
    _render_guided_kpis(selected_state)
    st.markdown("<br>", unsafe_allow_html=True)
    _render_cluster_explainer()
    _render_state_cluster_scores(selected_state)
    _render_state_sustainability_quartiles(selected_state)
    with st.expander("Show typology map for this state", expanded=False):
        _render_state_typology_map(selected_state)


def _render_city_selection(selected_state: str) -> tuple[str, str, bool]:
    city_opts = cities_for(selected_state)
    _sync_guided_cities_for_state()
    st.markdown("## Select one city or compare two cities")
    st.caption("City choices are restricted to the state you selected above.")
    st.radio(
        "Analysis mode",
        [GUIDED_COMPARE_SINGLE, GUIDED_COMPARE_TWO],
        format_func=lambda v: "Single city" if v == GUIDED_COMPARE_SINGLE else "Compare two cities",
        key="guided_compare_mode",
        horizontal=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("City", city_opts, key="guided_city_a", help=HELP["city_a"])
    single = st.session_state.get("guided_compare_mode") == GUIDED_COMPARE_SINGLE
    if single:
        st.session_state["guided_city_b"] = st.session_state["guided_city_a"]
    else:
        with c2:
            st.selectbox("Comparison city", city_opts, key="guided_city_b", help=HELP["city_b"])
        if st.session_state["guided_city_a"] == st.session_state["guided_city_b"] and len(city_opts) > 1:
            st.warning("Choose two different cities for a cleaner comparison.")
    return st.session_state["guided_city_a"], st.session_state["guided_city_b"], single


with st.sidebar:
    st.markdown("## Navigation")
    step_label = {
        GUIDED_ABOUT: "1. About",
        GUIDED_STATE: "2. State explorer",
        GUIDED_CITY: "3. City analysis",
    }.get(st.session_state.get("guided_step"), "1. About")
    st.caption(f"Current step: **{step_label}**")
    st.button("About", key="nav_about", on_click=_go_to_about, width="stretch")
    st.button("State explorer", key="nav_state", on_click=_go_to_state_explorer, width="stretch")
    if st.session_state.get("guided_step") == GUIDED_CITY:
        st.button("City analysis", key="nav_city", on_click=_go_to_city_selection, width="stretch")


selected_state = st.session_state["guided_state"]
city1 = st.session_state.get("guided_city_a", "")
city2 = st.session_state.get("guided_city_b", city1)
single_city_mode = st.session_state.get("guided_compare_mode") == GUIDED_COMPARE_SINGLE
if single_city_mode:
    city2 = city1

r1 = get_row(city1, selected_state)
r2 = get_row(city2, selected_state)
a1 = get_city_actions(city1, selected_state)
a2 = get_city_actions(city2, selected_state)
p1_impl = get_city_projects(city1, selected_state)
p2_impl = get_city_projects(city2, selected_state)

_render_guided_header(
    selected_state=selected_state,
    city1=city1 if st.session_state.get("guided_step") == GUIDED_CITY else "",
    city2=city2,
    single_city_mode=single_city_mode,
)

if st.session_state["guided_step"] == GUIDED_ABOUT:
    st.markdown(ABOUT_PAGE_MD)
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Continue to state explorer", type="primary", on_click=_go_to_state_explorer)
    st.stop()

if st.session_state["guided_step"] == GUIDED_STATE:
    _render_state_explorer(selected_state)
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Continue to city analysis", type="primary", on_click=_go_to_city_selection)
    st.stop()

st.markdown("## Step 3: City analysis")
st.caption("Change the selected city or same-state comparison city here if needed.")
city1, city2, single_city_mode = _render_city_selection(selected_state)
r1 = get_row(city1, selected_state)
r2 = get_row(city2, selected_state)
a1 = get_city_actions(city1, selected_state)
a2 = get_city_actions(city2, selected_state)
p1_impl = get_city_projects(city1, selected_state)
p2_impl = get_city_projects(city2, selected_state)

st.divider()
_render_guided_kpis(selected_state, r1, r2, single_city_mode=single_city_mode)

profile_tab, actions_tab, rec_tab = st.tabs([
    "City Profiles",
    "Actions Explorer",
    "How Can Cities Improve?",
])

with profile_tab:
    if single_city_mode:
        if r1 is not None:
            render_full_profile(r1, "#93c5fd", fin_avail, "guided_a")
        else:
            st.warning(f"**{city1}, {selected_state}** not found in the dataset.")
    else:
        p1, p2 = st.columns(2)
        with p1:
            if r1 is not None:
                render_full_profile(r1, "#93c5fd", fin_avail, "guided_a")
            else:
                st.warning(f"**{city1}, {selected_state}** not found in the dataset.")
        with p2:
            if r2 is not None:
                render_full_profile(r2, "#fbbf24", fin_avail, "guided_b")
            else:
                st.warning(f"**{city2}, {selected_state}** not found in the dataset.")

with actions_tab:
    if df_act is None:
        st.info(
            "The Climate Actions Excel file is not available, so action lists cannot be shown.",
            icon="⚡",
        )
    else:
        total_a1 = len(a1)
        total_a2 = len(a2)
        action_view = st.selectbox(
            "Action type",
            ["Planned actions", "Implemented projects"],
            key="guided_action_view",
            help="Switch between planned sustainability-report actions and implemented financial-record projects.",
        )

        if action_view == "Planned actions":
            st.markdown("### Planned actions by sector")
            s1 = (
                a1["sector"].value_counts().reindex(FOCUS_SECTORS, fill_value=0)
                if not a1.empty and "sector" in a1.columns
                else pd.Series(0, index=FOCUS_SECTORS)
            )
            s2 = (
                a2["sector"].value_counts().reindex(FOCUS_SECTORS, fill_value=0)
                if not a2.empty and "sector" in a2.columns
                else pd.Series(0, index=FOCUS_SECTORS)
            )
            fig_sec = go.Figure()
            if single_city_mode:
                fig_sec.add_trace(go.Bar(
                    name=f"{city1}, {selected_state}",
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
                st.plotly_chart(fig_sec, width="stretch", key="guided_sec_bar_single")
            else:
                for vals, nm, col_hex in [
                    (s1.values, city1, "#93c5fd"),
                    (s2.values, city2, "#fbbf24"),
                ]:
                    fig_sec.add_trace(go.Bar(
                        name=nm,
                        x=FOCUS_SECTORS,
                        y=vals,
                        marker_color=col_hex,
                        opacity=.85,
                        text=vals,
                        textposition="outside",
                        textfont=dict(size=9, color=col_hex),
                    ))
                fig_sec.update_layout(
                    **base_chart_layout(height=290, margin=dict(l=35, r=15, t=20, b=45)),
                    barmode="group",
                    yaxis=dark_axis(title="Number of actions"),
                    xaxis=dict(tickfont=dict(size=11), gridcolor="#0f1e30"),
                )
                st.plotly_chart(fig_sec, width="stretch", key="guided_sec_bar")

            st.divider()
            sel_secs = st.multiselect(
                "Filter by sector",
                FOCUS_SECTORS,
                default=FOCUS_SECTORS,
                key="guided_actions_filter",
                help=HELP["sector_filter"],
            )
            if not a1.empty or not a2.empty:
                combined = a1 if single_city_mode else pd.concat([a1, a2], ignore_index=True)
                csv_bytes = combined.to_csv(index=False).encode()
                st.download_button(
                    f"Download actions for {city1} (.csv)" if single_city_mode else "Download actions for both cities (.csv)",
                    data=csv_bytes,
                    file_name=(
                        f"actions_{city1}_{selected_state}.csv".replace(" ", "_")
                        if single_city_mode
                        else f"actions_{city1}_{city2}.csv".replace(" ", "_")
                    ),
                    mime="text/csv",
                    key="guided_dl_actions",
                    help=HELP["download_actions"],
                )
            st.markdown("### Planned climate actions")
            if single_city_mode:
                st.markdown(f"**{city1}, {selected_state}** — {total_a1} total actions")
                render_action_list(
                    f"{city1}, {selected_state}",
                    a1,
                    sel_secs,
                    list_state_key="guided_actions_city_a",
                )
            else:
                la1, la2 = st.columns(2)
                with la1:
                    st.markdown(f"**{city1}, {selected_state}** — {total_a1} total actions")
                    render_action_list(
                        f"{city1}, {selected_state}",
                        a1,
                        sel_secs,
                        list_state_key="guided_actions_city_a",
                    )
                with la2:
                    st.markdown(f"**{city2}, {selected_state}** — {total_a2} total actions")
                    render_action_list(
                        f"{city2}, {selected_state}",
                        a2,
                        sel_secs,
                        list_state_key="guided_actions_city_b",
                    )

        else:
            st.markdown("### Implemented projects from financial records")
            if df_projects is None:
                st.info("Implemented project extracts are not available.")
            else:
                pi1, pi2 = st.columns(2) if not single_city_mode else (st.container(), None)
                with pi1:
                    render_focus_sector_pie(
                        p1_impl,
                        f"{city1}, {selected_state}",
                        "#93c5fd",
                        "guided_impl_pie_a",
                        item_label="projects",
                    )
                if not single_city_mode and pi2 is not None:
                    with pi2:
                        render_focus_sector_pie(
                            p2_impl,
                            f"{city2}, {selected_state}",
                            "#fbbf24",
                            "guided_impl_pie_b",
                            item_label="projects",
                        )

                impl_secs = st.multiselect(
                    "Filter implemented projects by sector",
                    FOCUS_SECTORS,
                    default=FOCUS_SECTORS,
                    key="guided_impl_filter",
                    help=HELP["sector_filter"],
                )
                if single_city_mode:
                    render_project_list(
                        f"{city1}, {selected_state}",
                        p1_impl,
                        impl_secs,
                        list_state_key="guided_projects_city_a",
                    )
                else:
                    ip1, ip2 = st.columns(2)
                    with ip1:
                        render_project_list(
                            f"{city1}, {selected_state}",
                            p1_impl,
                            impl_secs,
                            list_state_key="guided_projects_city_a",
                        )
                    with ip2:
                        render_project_list(
                            f"{city2}, {selected_state}",
                            p2_impl,
                            impl_secs,
                            list_state_key="guided_projects_city_b",
                        )

with rec_tab:
    if single_city_mode:
        st.markdown(
            '<div class="info-banner">'
            "<b>Peer learning for your selected city</b><br>"
            "For this city, the app finds a municipality with similar Overall Fiscal Health and a "
            "higher sustainability score, then compares climate actions and score pillars."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="info-banner">'
            "<b>Peer learning & benchmarking</b><br>"
            "For each selected city, the app finds a municipality with similar Overall Fiscal Health "
            "and a higher sustainability score, then compares climate actions and score pillars."
            "</div>",
            unsafe_allow_html=True,
        )
    sel_secs_improve = st.multiselect(
        "Sectors to show in action lists",
        FOCUS_SECTORS,
        default=FOCUS_SECTORS,
        key="guided_improve_sectors",
        help=HELP["improve_sectors"],
    )
    if single_city_mode:
        render_improvement_benchmark_for_city(
            r1,
            city1,
            selected_state,
            "#93c5fd",
            "#34d399",
            sel_secs_improve,
            "guided_improve_single",
        )
    else:
        sub_a, sub_b = st.tabs([city1, city2])
        with sub_a:
            render_improvement_benchmark_for_city(
                r1,
                city1,
                selected_state,
                "#93c5fd",
                "#34d399",
                sel_secs_improve,
                "guided_improve_a",
            )
        with sub_b:
            render_improvement_benchmark_for_city(
                r2,
                city2,
                selected_state,
                "#fbbf24",
                "#34d399",
                sel_secs_improve,
                "guided_improve_b",
            )

st.stop()

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
    st.selectbox(
        "State A",
        compare_states,
        key="state_a",
        on_change=on_state_a_change,
        label_visibility="collapsed",
        help=HELP["state_pick"],
    )
    c_a = cities_for(st.session_state["state_a"])
    if st.session_state.get("city_a") not in c_a:
        st.session_state["city_a"] = c_a[0] if c_a else ""
    st.selectbox(
        "City A",
        c_a,
        key="city_a",
        label_visibility="collapsed",
        help=HELP["city_a"],
    )

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    # ── City B ────────────────────────────────────────────────────────────────
    st.markdown('<span class="sb-lbl">City B 🟡</span>', unsafe_allow_html=True)
    st.selectbox(
        "State B",
        compare_states,
        key="state_b",
        on_change=on_state_b_change,
        label_visibility="collapsed",
        help=HELP["state_pick"],
    )
    c_b = cities_for(st.session_state["state_b"])
    if st.session_state.get("city_b") not in c_b:
        st.session_state["city_b"] = c_b[0] if c_b else ""
    st.selectbox(
        "City B",
        c_b,
        key="city_b",
        label_visibility="collapsed",
        help=HELP["city_b"],
    )

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
        help=HELP["dashboard_view"],
    )
    if st.session_state.get("dashboard_view_mode") == VIEW_MODE_CUSTOM:
        st.markdown('<span class="sb-lbl">Focus state (IL / MI / MN / WI)</span>', unsafe_allow_html=True)
        st.selectbox(
            "Focus state",
            focus_states_list,
            key="focus_custom_state",
            on_change=on_focus_custom_state_change,
            label_visibility="collapsed",
            help=HELP["focus_state"],
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
            help=HELP["focus_city"],
        )

    st.divider()
    st.markdown("### Map options")
    show_pop    = st.checkbox(
        "Scale bubbles by population",
        value=True,
        key="show_pop",
        help=HELP["map_population"],
    )
    show_labels = st.checkbox(
        "Show all city labels",
        value=False,
        key="show_lbl",
        help=HELP["map_labels"],
    )
    shade       = st.checkbox(
        "Shade quadrant areas",
        value=True,
        key="shade",
        help=HELP["map_shade"],
    )
    hl_states   = st.multiselect(
        "Highlight state(s)",
        highlight_states,
        default=[],
        key="hl_st",
        help=HELP["map_highlight"],
    )
    _typology_map_state_opts = [
        s for s in FOCUS_STATES_ORDER if s in df["State"].dropna().unique().tolist()
    ]
    if _typology_map_state_opts:
        st.multiselect(
            "Typology map — states to plot",
            options=_typology_map_state_opts,
            default=_typology_map_state_opts,
            key="t1_map_states",
            help=HELP["t1_map_states"],
        )

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
        f"{KPI_CITIES} cities &nbsp;·&nbsp; {KPI_STATES} states &nbsp;·&nbsp; "
        f'<b>Single-city view:</b> <span style="color:#93c5fd">{city1}, {state1}</span>'
        f"</p>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<p style="color:#ffffff;font-size:.76rem;margin-top:-10px">'
        f"{KPI_CITIES} cities &nbsp;·&nbsp; {KPI_STATES} states &nbsp;·&nbsp; "
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
    for col_w, (v, lbl, hk) in zip(k_cols, [
        (str(KPI_CITIES),       "Cities (IL · MI · MN · WI)", "kpi_cities"),
        (str(KPI_STATES),       "States in core region",    "kpi_states"),
        (f"{sus_a:.1f}/48",     f"{city1} — sustainability", "kpi_sustain"),
    ]):
        with col_w:
            st.markdown(
                kpi_card(v, lbl, tip=HELP[hk]),
                unsafe_allow_html=True,
            )
else:
    k_cols = st.columns(4)
    for col_w, (v, lbl, hk) in zip(k_cols, [
        (str(KPI_CITIES),       "Cities (IL · MI · MN · WI)", "kpi_cities"),
        (str(KPI_STATES),       "States in core region",    "kpi_states"),
        (f"{sus_a:.1f}/48",     f"{city1} Sustainability", "kpi_sustain"),
        (f"{sus_b:.1f}/48",     f"{city2} Sustainability", "kpi_sustain"),
    ]):
        with col_w:
            st.markdown(
                kpi_card(v, lbl, tip=HELP[hk]),
                unsafe_allow_html=True,
            )

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
T_about, T5, T4, T3, T1 = st.tabs([
    "About",
    "How Can Cities Improve?",
    "🔬  City Profiles",
    "⚡  Actions Explorer",
    "🗺️  Typology Map",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB  ·  ABOUT
# ──────────────────────────────────────────────────────────────────────────────
with T_about:
    st.markdown(ABOUT_PAGE_MD)

# ──────────────────────────────────────────────────────────────────────────────
# TAB  ·  HOW CAN CITIES IMPROVE?  (peer benchmarking; next to About)
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
    with st.popover("❓ Peer benchmark — vocabulary"):
        st.markdown(
            "**Benchmark peer:** another city in this file with **similar Overall Fiscal Health** "
            "(the composite fiscal index) but a **higher sustainability score**.\n\n"
            "**Recommendations:** use **Generate peer-based recommendations (Gemini 2.5 Flash)** to send full "
            "action text plus rubric/fiscal snapshots to Gemini—outputs should reflect **benchmark-only** "
            "evidence. These are **ideas for discussion**, not directives."
        )
    with st.expander(
        "How is Overall Fiscal Health calculated?",
        expanded=False,
    ):
        st.markdown(FISCAL_HEALTH_EXPLAINER_MD)
    st.caption(
        "Peer matching uses the Overall Fiscal Health index—open the section above for the full definition."
    )
    sel_secs_improve = st.multiselect(
        "Sectors to show in action lists",
        FOCUS_SECTORS,
        default=FOCUS_SECTORS,
        key="t5_improve_sectors",
        help=HELP["improve_sectors"],
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

# ──────────────────────────────────────────────────────────────────────────────
# TAB  ·  TYPOLOGY MAP
# ──────────────────────────────────────────────────────────────────────────────
with T1:
    st.markdown(
        '<div class="info-banner">'
        f'<b>What “{PC1_AXIS_LABEL.lower()}” and “liquidity” mean on this map</b><br>'
        f"<b>{PC1_AXIS_LABEL} (horizontal):</b> a <b>composite</b> fiscal-stress axis from <b>PC1</b> of all standardized "
        "balance-sheet columns, oriented using the spreadsheet’s <b>Pension Exposure Ratio</b> (when present) as a "
        f"<b>direction anchor</b>—<b>further right = a lighter long-term load vs. cities in this file</b>, "
        "not a legal rating. (Still blends debt and other signals—not “pension only.”)<br>"
        "<b>Liquidity (vertical):</b> the average of <b>Cash &amp; Investment Coverage</b> and "
        "<b>Net Investment Capacity</b> only (standardized as a pair)—<b>higher = more cash / "
        "net-investment cushion vs. peers</b>, not overall city wealth.<br><br>"
        '<b>How cluster names are assigned:</b><br>'
        '• <b>Resilient (Q1)</b>: lower long-term pressure and higher cash liquidity.<br>'
        '• <b>Stable (Q2)</b>: lower long-term pressure but tighter cash liquidity.<br>'
        '• <b>Pressured (Q3)</b>: higher long-term pressure, but currently stronger liquidity.<br>'
        '• <b>Vulnerable (Q4)</b>: higher long-term pressure and lower liquidity.<br><br>'
        '<i>These labels use <b>median splits within your uploaded file</b> (peer-relative), '
        "not an official credit rating or legal standard.</i>"
        '</div>',
        unsafe_allow_html=True,
    )

    _typ_opts = [s for s in FOCUS_STATES_ORDER if s in df["State"].dropna().unique().tolist()]
    _t1_sel = st.session_state.get("t1_map_states")
    if _typ_opts:
        if not _t1_sel:
            _t1_sel = _typ_opts
        _t1_sel = [s for s in _t1_sel if s in _typ_opts] or _typ_opts
        df_map = df[df["State"].isin(_t1_sel)].copy()
        _t1_sel_set = frozenset(_t1_sel)
    else:
        df_map = df
        _t1_sel_set = None

    with st.popover("❓ What this map shows (plain language)"):
        st.markdown(
            f"### {PC1_AXIS_LABEL} vs liquidity (plain language)\n\n"
            f"**{PC1_AXIS_LABEL} (horizontal)** — A **summary stress score** from the first PCA axis of **all** "
            "standardized fiscal columns, oriented using **Pension Exposure Ratio** (when present) only as a "
            "**direction anchor**. **Right** = a **lighter** long-term fiscal load than other cities **in this "
            "spreadsheet**; **left** = a **heavier** load in that *combined* sense. It is **not** a single "
            "funded-ratio verdict by itself.\n\n"
            "**Liquidity (vertical)** — **Cash & investment coverage** plus **net investment capacity**, averaged "
            "after scaling those **two** columns together. **Up** = more **near-term cash / investment headroom** "
            "vs. peers in the file—not a measure of how “rich” the local economy is.\n\n"
            "### Technical detail\n\n"
            f"**Horizontal ({PC1_AXIS_LABEL.lower()}):** built from **PC1** of the financial PCA, oriented so "
            "**further right = relatively lower long-term fiscal pressure** among cities in this file.\n\n"
            "**Vertical (liquidity axis):** from **cash & investment coverage** and "
            "**net investment capacity** only—**higher = stronger liquidity** vs. peers.\n\n"
            "**Dashed lines:** the **median** of each axis using **all cities in your upload** "
            "(not a legal threshold). They split the full sample into four groups "
            "(Resilient / Stable / Pressured / Vulnerable).\n\n"
            "**State filter (sidebar):** limits **which dots** appear on this map; medians stay the same.\n\n"
            "**Zoom:** use the **mode bar** (top-right of the chart) for box/lasso zoom, **scroll** with the "
            "pointer over the map to zoom in/out, or **double-click** to reset the axes.\n\n"
            "**Colors:** each dot’s color is its group. **Stars** mark your selected city or cities "
            "(only if that city’s state is included in the map filter)."
        )

    _px = df_map["PC1_pension_axis"].dropna().to_numpy()
    _py = df_map["liquidity_axis"].dropna().to_numpy()
    if _px.size >= 12:
        xm0, xm1 = map(float, np.percentile(_px, [2, 98]))
        ym0, ym1 = map(float, np.percentile(_py, [2, 98]))
    elif _px.size > 0:
        xm0, xm1 = float(_px.min()), float(_px.max())
        ym0, ym1 = float(_py.min()), float(_py.max())
    else:
        xm0, xm1, ym0, ym1 = -0.1, 0.1, -0.1, 0.1
    xspan = max(xm1 - xm0, 0.06)
    yspan = max(ym1 - ym0, 0.06)
    xpad = xspan * 0.14
    ypad = yspan * 0.14
    _pm = float(pc1_median)
    _lm = float(liq_median)
    _msx = max(0.04, xspan * 0.12)
    _msy = max(0.04, yspan * 0.12)
    xr = [min(xm0 - xpad, _pm - _msx), max(xm1 + xpad, _pm + _msx)]
    yr = [min(ym0 - ypad, _lm - _msy), max(ym1 + ypad, _lm + _msy)]

    fig_map = go.Figure()

    # Quadrant shading (x increases → lower long-term pressure; y increases → liquidity)
    # Upper-left = higher pressure + high liq (Q3); upper-right = Q1; lower-left = Q4; lower-right = Q2
    if shade:
        quad_shapes = [
            (xr[0], pc1_median, liq_median, yr[1], "rgba(251,191,36,.05)",  "Q3  PRESSURED",  "#fbbf24",
             xr[0] + (pc1_median - xr[0]) * .12, yr[1] - (yr[1] - liq_median) * .09),
            (pc1_median, xr[1], liq_median, yr[1], "rgba(52,211,153,.05)",  "Q1  RESILIENT",  "#34d399",
             pc1_median + (xr[1] - pc1_median) * .55, yr[1] - (yr[1] - liq_median) * .09),
            (xr[0], pc1_median, yr[0], liq_median, "rgba(248,113,113,.05)", "Q4  VULNERABLE", "#f87171",
             xr[0] + (pc1_median - xr[0]) * .06, yr[0] + (liq_median - yr[0]) * .12),
            (pc1_median, xr[1], yr[0], liq_median, "rgba(96,165,250,.05)",  "Q2  STABLE",     "#60a5fa",
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

    # Cities on the map (subset by sidebar “Typology map — states to plot” when available)
    for quad, (qn, _, qcol) in QUAD_META.items():
        sub = df_map[df_map["pca_2x2_type"] == quad]
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
                f"{PC1_AXIS_LABEL}: %{{x:.2f}}  ·  Liquidity axis: %{{y:.2f}}"
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
        if _t1_sel_set is not None and str(row_d.get("State", "")) not in _t1_sel_set:
            continue
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
        title=dict(
            text=f"2×2 Fiscal typology: {PC1_AXIS_LABEL.lower()} × liquidity (median split)",
            font=dict(size=13, color="#f8fafc", family="IBM Plex Sans"),
            x=0.5,
            xanchor="center",
        ),
        **base_chart_layout(height=570, margin=dict(l=55, r=25, t=48, b=58)),
        xaxis=dark_axis(
            title=f"{PC1_AXIS_LABEL} — higher = lighter long-term load (vs. peers)",
            range=xr, title_font=dict(size=10),
        ),
        yaxis=dark_axis(
            title="Liquidity axis — higher = stronger cash + net investment (vs. peers)",
            range=yr, title_font=dict(size=10),
        ),
    )
    st.plotly_chart(
        fig_map,
        width="stretch",
        key="map_main",
        config=PLOTLY_TYPOLOGY_CONFIG,
    )
    st.caption(
        "Tip: **scroll** the mouse wheel over the map to zoom, use the **mode bar** (top-right of the chart) "
        "for box zoom / pan, or **double-click** the plot to reset axes."
    )
    if _typ_opts and _t1_sel is not None and len(_t1_sel) < len(_typ_opts):
        st.caption(
            f"Showing **{len(df_map)}** cities in **{', '.join(_t1_sel)}**; dashed medians still reflect "
            "**all** cities in your fiscal file."
        )
    elif _typ_opts:
        st.caption(
            "Dashed medians divide **every city in your upload**; cluster colors use the same global assignment."
        )

    # Bottom row: cluster donut + sustainability box
    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown("### Cities per cluster")
        cnt = df_map["pca_2x2_type"].value_counts().reset_index()
        cnt.columns = ["Cluster", "Count"]
        if cnt.empty:
            st.caption("No cities match the current map state filter.")
        else:
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
            st.plotly_chart(fig_donut, width="stretch", key="donut")

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
                    help=HELP["sus_cluster"],
                )
            with f2:
                sel_state = st.selectbox(
                    "Choose state",
                    state_opts,
                    key="sus_state_select",
                    help=HELP["sus_state"],
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
                    width="stretch",
                    key="bar_city_sus_cluster",
                )

# ──────────────────────────────────────────────────────────────────────────────
# TAB  ·  ACTIONS EXPLORER
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

        with st.popover("❓ Actions tab — what counts?"):
            st.markdown(
                "Counts and lists are **derived from each city's sustainability report** in three sectors: "
                "**energy**, **waste**, and **transport**."
            )

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
            st.plotly_chart(fig_sec, width="stretch", key="sec_bar_single")
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
            st.plotly_chart(fig_sec, width="stretch", key="sec_bar")

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
                st.plotly_chart(fig_pie, width="stretch", key="pie_single_city")
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
                        st.plotly_chart(fig_pie, width="stretch",
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
            help=HELP["sector_filter"],
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
                help=HELP["download_actions"],
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
# TAB  ·  CITY PROFILES
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
