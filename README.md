# Municipal Fiscal & Sustainability Dashboard

An interactive **[Streamlit](https://streamlit.io/)** app for exploring **Midwest municipalities** through three lenses: **fiscal typology** (balance-sheet style indicators), **sustainability scores** (governance, data, and action planning), and **documented climate actions** (energy, transport, waste). It is designed for planners, finance staff, and sustainability offices who want maps, comparisons, and peer benchmarks in one place.

---

## Features at a glance

| Area | What you get |
|------|----------------|
| **About** | Methodology for the **2×2 fiscal typology** (PCA **long-term pressure** axis, liquidity axis, median split) and how it relates to **Overall Fiscal Health**. |
| **Typology map** | Interactive 2×2 plot with **corrected quadrant shading** (labels match corners), optional population-sized bubbles, and sustainability bars by cluster. |
| **City vs. city** | Side-by-side profile cards, normalized sustainability + fiscal metrics, and financial-indicator bars vs. the peer average. |
| **Actions explorer** | Sector counts, mini pies, filterable side-by-side action cards, and CSV export for the two selected cities. |
| **City profiles** | Deep profiles for each selected city (sustainability, fiscal metrics, and typology context). |
| **How can cities improve?** | For **City A** and **City B** separately: auto-matched **benchmark peers** with similar **Overall Fiscal Health** but higher sustainability scores; sector pies; action comparisons; narrative gaps and recommendations. |

The UI uses a **dark, high-contrast theme** tuned in `.streamlit/config.toml` for long sessions and presentations.

---

## Requirements

- **Python 3.10+** (the app is routinely used with **Python 3.12**; any recent 3.x should work).
- Packages listed in **`requirements.txt`**:

  `streamlit`, `plotly`, `pandas`, `numpy`, `scikit-learn`, `openpyxl`, `google-generativeai`

**Optional — AI peer recommendations (How Can Cities Improve?)** Set **`GOOGLE_API_KEY`** / **`GEMINI_API_KEY`** via **`frontend/.streamlit/secrets.toml`** locally (see **§4b**), the host environment, or Streamlit Cloud **Secrets**. Optional **`GEMINI_MODEL`** (default **`gemini-2.5-flash`**).

## Quick start

### 1. Clone or copy this folder

```bash
cd frontend
```

### 2. Create and activate a virtual environment (recommended)

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the data files

Place these Excel workbooks **next to `dashboard.py`** (same folder as this README):

| File | Role |
|------|------|
| **`midwest data (1).xlsx`** | Fiscal + sustainability fields for each municipality. The loader uses **`Sheet2`** if it exists, otherwise the **first sheet**. |
| **`municipality_actions.xlsx`** | Climate actions. Preferred sheet name: **`Municipality Actions`**; otherwise the **first sheet**. Rows should include (at minimum) city, state, sector, action name, and action text. Only **Energy**, **Transport**, and **Waste** sectors are loaded for the actions views. |

If the fiscal file is missing, the app stops with an error. If the actions file is missing, fiscal and typology views still work; action-heavy tabs show a clear notice.

### 4b. Gemini API key (Streamlit secrets — recommended)

1. Open **`frontend/.streamlit/secrets.toml`** (created for you with an empty key).
2. Set **`GOOGLE_API_KEY = "your-key-here"`** (get a key from [Google AI Studio](https://aistudio.google.com/apikey)).
3. Save the file and run Streamlit from the **`frontend`** folder as usual. **`secrets.toml` is gitignored** so it is not committed; use **`secrets.toml.example`** only as a copy reference if you delete the real file.

### 5. Run the app

```bash
streamlit run dashboard.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Open it in your browser.

> **Tip (Windows):** If you have several Python versions installed, run Streamlit **with the same interpreter** you used for `pip install`, for example:
>
> `py -3.12 -m streamlit run dashboard.py`

---

## How to use the app

1. In the **sidebar**, under **Compare cities**, pick **State** and **City** for **City A** and **City B** (used when you are comparing two places).
2. Under **Dashboard view**, choose **Compare City A & City B** (default), **City A only**, **City B only**, or **Single city — Illinois, Michigan, Minnesota, or Wisconsin**. The last option shows **Focus state** and **Focus city** dropdowns (only those four states). **Single-city modes** show that municipality alone on **every** tab (typology highlight, profiles, actions, peer improvement, etc.).
3. Use **Map options** to toggle bubble scaling, labels, quadrant shading, and state highlighting.
4. Open **How Can Cities Improve?** for peer benchmarks; in single-city mode there is one panel (no City A / B sub-tabs). Expand **How is Overall Fiscal Health calculated?** for the fiscal index, and **How are recommendations for … generated?** (under each city’s charts) for the rules behind suggested actions.

---

## Key metrics (short definitions)

- **Overall Fiscal Health (`fiscal_health`)** — A **peer index**: standardized financial indicators (cash, debt, pensions, capital burden, etc.) are aligned so “higher is better,” averaged per city vs. the **dataset**. Values near **0** are typical for the sample; small differences between cities mean similar fiscal “shape” for benchmarking.
- **Sustainability total** — Reported as **x/48** from the spreadsheet, with sub-pillars for governance, data & analytics, and action planning where columns exist.
- **Typology quadrants** — Cities split by **median** **long-term pressure** (composite PC1 axis, oriented with pension exposure in the data) and **liquidity** (cash & net investment columns), not by a single pension funded ratio.

Full on-screen copy lives in the app (especially under **How Can Cities Improve?**).

---

## Project layout

```text
frontend/
├── README.md                 ← this file
├── requirements.txt
├── dashboard.py              ← Streamlit entrypoint (single app module)
├── .streamlit/
│   └── config.toml           ← dark theme + brand colors
├── midwest data (1).xlsx     ← you supply (not always in repo)
└── municipality_actions.xlsx ← you supply (not always in repo)
```

---

