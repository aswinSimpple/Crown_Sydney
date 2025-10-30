import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict, Any

# =============================
# Defaults (kept similar to Melbourne)
# =============================
PRODUCTIVITY_RATES_SQM_PER_HR = {
    "General": 250,
    "Toilet": 50,
    "Kitchen": 100,
    "Gaming": 200,
    "Office": 300,
    "External": 500,
    "Premium": 150,
    "Conference": 280,
    "Retail": 220,
    "Elevator/Lift": 250,  # treat as small floor area (no unit counts available)
    "Escalator": 200,
}

# Deep cleans are slower -> lower productivity (can be adjusted in UI later)
DEEP_PRODUCTIVITY_RATES_SQM_PER_HR = {
    k: max(v * 0.6, 10) for k, v in PRODUCTIVITY_RATES_SQM_PER_HR.items()
}

# Per-unit task time library (minutes per unit) used for fixture-density hybrid in Advanced mode
TASK_TIMES_MIN = {
    "toilet_cubicle": 5,
    "urinal": 3,
    "sink": 2,
    "gaming_machine": 5,
    "office_desk": 3,
}

SERVICE_LEVELS = {"Basic": 0.8, "Standard": 1.0, "Premium": 1.2, "Luxury": 1.5}
COMPLEXITY_FACTORS = {"Low": 0.9, "Medium": 1.0, "High": 1.15, "Very High": 1.3}

# Day-of-week weights (edit to taste). Heavier Fri/Sat by default.
DEFAULT_DAY_WEIGHTS = {
    "Mon": 1.0,
    "Tue": 1.0,
    "Wed": 1.0,
    "Thu": 1.1,
    "Fri": 1.6,
    "Sat": 1.6,
    "Sun": 0.9,
}

# Shift split (percentage of each day). Keep PM heavier by default.
DEFAULT_SHIFT_WEIGHTS = {
    "Morning (06-14)": 0.35,
    "Afternoon (14-22)": 0.45,
    "Night (22-06)": 0.20,
}

# =============================
# Helpers
# =============================
SYD_EXPECTED_COLUMNS = [
    "Package",
    "Building",
    "Level",
    "Location",
    "SQM",
    "Minimum Standards Reference",
    "Detailed Clean Frequency (# per week) Standard",
    "NON-PEAK Freq (# per week) Standard",
    "PEAK Freq (# per week) Standard",
]

COLUMN_ALIASES = {
    "package": "package",
    "building": "building",
    "lvl": "level",
    "level": "level",
    "location": "location",
    "sqm": "sqm",
    "minimum standards reference": "min_standards_ref",
    # detailed frequency variations
    "detailed": "freq_detailed_raw",
    "detailed clean frequency (# per week) standard": "freq_detailed_raw",
    "detailed clean frequency (# per week)standard": "freq_detailed_raw",
    # non-peak / peak variations
    "non-peak freq (# per week) standard": "freq_nonpeak_per_wk",
    "non peak freq (# per week) standard": "freq_nonpeak_per_wk",
    "nonpeak freq (# per week) standard": "freq_nonpeak_per_wk",
    "peak freq (# per week) standard": "freq_peak_per_wk",
}

AREA_KEYWORDS = [
    ("Toilet", ["toilet", "washroom", "restroom", "bathroom", "wc"]),
    ("Kitchen", ["kitchen", "food", "cafeteria", "restaurant"]),
    ("Gaming", ["gaming", "casino", "slot", "tables"]),
    ("Office", ["office", "admin", "meeting", "boardroom"]),
    ("External", ["parking", "loading", "outdoor", "exterior", "external"]),
    ("Premium", ["vip", "premium", "suite", "towers"]),
    ("Conference", ["conference", "event", "exhibition", "ballroom"]),
    ("Retail", ["retail", "shop", "store", "boutique"]),
    ("Elevator/Lift", ["lift", "elevator"]),
    ("Escalator", ["escalator"]),
]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.astype(str).str.strip()
    cols = cols.str.replace("\r\n", " ", regex=False)
    cols = cols.str.replace("\n", " ", regex=False)
    cols = cols.str.replace("\r", " ", regex=False)
    cols = cols.str.replace("  ", " ", regex=False)
    df.columns = cols
    return df


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        key = c.lower().strip()
        if key in COLUMN_ALIASES:
            mapping[c] = COLUMN_ALIASES[key]
        else:
            # fuzzy contains matching
            for k, v in COLUMN_ALIASES.items():
                if k in key:
                    mapping[c] = v
                    break
    df = df.rename(columns=mapping)
    return df


def classify_area_type(location: Any, package: Any) -> str:
    loc = str(location).lower()
    pkg = str(package).lower()
    for area, keywords in AREA_KEYWORDS:
        if area == "Premium" and ("towers" in pkg or any(w in loc for w in keywords)):
            return area
        if any(w in loc for w in keywords) or area.lower() in pkg:
            return area
    # fallback
    return "General"


def parse_number(x) -> float:
    if pd.isna(x):
        return np.nan
    try:
        return float(str(x).strip())
    except Exception:
        return np.nan


def parse_detailed_field(raw: Any) -> Tuple[float, float, str]:
    """Return (freq_per_wk, minutes_per_clean_override, interpretation).
    - If raw is numeric-like → frequency per week, no minutes override.
    - If raw contains tokens like 'm', 'min', 'v' → treat as minutes per clean, default freq=1.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return (0.0, np.nan, "empty")

    s = str(raw).strip().lower()
    # try pure numeric first
    try:
        val = float(s)
        return (val, np.nan, "frequency_per_week")
    except Exception:
        pass

    # look for minute-like tokens
    if any(tok in s for tok in ["m", "min", "mins", "minute", "minutes", "v"]):
        # extract first number
        import re
        m = re.search(r"([0-9]+(\.[0-9]+)?)", s)
        if m:
            mins = float(m.group(1))
            return (1.0, mins, "minutes_per_clean")
        else:
            return (1.0, np.nan, "minutes_flag_no_number")

    # fallback: nothing parsable
    return (0.0, np.nan, "unparsed")


def hours_per_clean_area(sqm: float, area_type: str, prod_rates: Dict[str, float]) -> float:
    rate = prod_rates.get(area_type, prod_rates.get("General", 250))
    if rate <= 0:
        return 0.0
    return float(sqm) / float(rate)


def effective_weekly_frequency(peak_days: int, nonpeak_freq: float, peak_freq: float) -> float:
    peak_days = max(0, min(7, int(peak_days)))
    nonpeak_days = 7 - peak_days
    f_np = nonpeak_freq if pd.notna(nonpeak_freq) else 0.0
    f_pk = peak_freq if pd.notna(peak_freq) else f_np
    return (peak_days * f_pk + nonpeak_days * f_np) / 7.0


def apply_multipliers(hours: float, service_level: str, complexity: str, special_mult: float = 1.0) -> float:
    sl = SERVICE_LEVELS.get(service_level or "Standard", 1.0)
    cx = COMPLEXITY_FACTORS.get(complexity or "Medium", 1.0)
    return hours * sl * cx * special_mult


def distribute_weekly_hours(weekly_hours: float, day_weights: Dict[str, float]) -> Dict[str, float]:
    total_w = sum(day_weights.values())
    if total_w <= 0:
        total_w = 1.0
    return {d: weekly_hours * (w / total_w) for d, w in day_weights.items()}


def split_shifts(daily_hours: float, shift_weights: Dict[str, float]) -> Dict[str, float]:
    sw_sum = sum(shift_weights.values()) or 1.0
    return {shift: daily_hours * (w / sw_sum) for shift, w in shift_weights.items()}

# =============================
# Monte Carlo (Sydney)
# =============================

def _calc_budget_from_fte(fte_total: float, hourly_wage: float, include_benefits: bool, num_robots: int, smart_facility: bool) -> Dict[str, float]:
    annual_hours = 52 * 38
    base_salary = fte_total * hourly_wage * annual_hours
    labor_cost = base_salary * (1.3 if include_benefits else 1.0)
    supplies_cost = labor_cost * 0.15
    robot_cost = num_robots * 35000
    smart_cost = 50000 if smart_facility else 0
    mgmt_cost = labor_cost * 0.10
    total_budget = labor_cost + supplies_cost + robot_cost + smart_cost + mgmt_cost
    return {
        "labor_cost": labor_cost,
        "supplies_cost": supplies_cost,
        "robot_cost": robot_cost,
        "smart_facility_cost": smart_cost,
        "management_cost": mgmt_cost,
        "total_budget": total_budget,
    }


def run_monte_carlo_sydney(
    df: pd.DataFrame,
    base_scenario: str,
    peak_days_per_week: int,
    prod_cv: float,
    deep_cv: float,
    dens_cv: float,
    productivity_modifier: float,
    weekly_hours_contract: float,
    efficiency_factor: float,
    default_service: str,
    default_complex: str,
    n_iter: int,
    hourly_wage: float,
    include_benefits: bool,
    num_robots: int,
    smart_facility: bool,
    adv_service_map: Dict[str, str] = None,
    adv_complex_map: Dict[str, str] = None,
    special_mult: float = 1.0,
    dens_toilet: float = 0.0,
    dens_gaming: float = 0.0,
    dens_desks: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate uncertainty. If base_scenario=='Advanced', also vary fixture density.
    prod_cv, deep_cv, dens_cv are given as decimals (e.g., 0.15 for ±15%).
    """
    results_fte = np.zeros(n_iter)
    results_budget = np.zeros(n_iter)

    area_types = list(PRODUCTIVITY_RATES_SQM_PER_HR.keys())

    # Ensure dicts
    adv_service_map = adv_service_map or {}
    adv_complex_map = adv_complex_map or {}

    for i in range(n_iter):
        # Sample per-area productivity multipliers
        sampled_prod = {}
        sampled_deep = {}
        for at in area_types:
            m = max(0.6, np.random.normal(1.0, prod_cv))
            md = max(0.6, np.random.normal(1.0, deep_cv))
            sampled_prod[at] = max(1e-6, PRODUCTIVITY_RATES_SQM_PER_HR.get(at, 250) * m)
            sampled_deep[at] = max(1e-6, DEEP_PRODUCTIVITY_RATES_SQM_PER_HR.get(at, 150) * md)

        # Density multipliers (only used if base=Advanced)
        dens_mul = max(0.3, np.random.normal(1.0, dens_cv)) if base_scenario == 'Advanced' else 1.0
        d_toilet = dens_toilet * dens_mul
        d_gaming = dens_gaming * dens_mul
        d_desks  = dens_desks  * dens_mul

        weekly_hours_total = 0.0
        for _, r in df.iterrows():
            at = r["area_type"]
            sqm = float(r["sqm"]) if pd.notna(r["sqm"]) else 0.0
            # Routine hours per clean
            if base_scenario == 'Advanced':
                # Hybrid: area + items using (possibly jittered) densities
                area_based = hours_per_clean_area(sqm, at, sampled_prod) * productivity_modifier
                item_hours = 0.0
                if at == 'Toilet' and d_toilet > 0:
                    num = (sqm / 100.0) * d_toilet
                    mins = num * (0.6*TASK_TIMES_MIN['toilet_cubicle'] + 0.2*TASK_TIMES_MIN['urinal'] + 0.2*TASK_TIMES_MIN['sink'])
                    item_hours = mins / 60.0
                elif at == 'Gaming' and d_gaming > 0:
                    num = (sqm / 100.0) * d_gaming
                    item_hours = (num * TASK_TIMES_MIN['gaming_machine']) / 60.0
                elif at == 'Office' and d_desks > 0:
                    num = (sqm / 100.0) * d_desks
                    item_hours = (num * TASK_TIMES_MIN['office_desk']) / 60.0
                hpc_r = area_based*0.4 + item_hours*0.6 if item_hours>0 else area_based
            else:
                hpc_r = hours_per_clean_area(sqm, at, sampled_prod) * productivity_modifier

            eff_freq = effective_weekly_frequency(peak_days_per_week, r["freq_nonpeak_per_wk"], r["freq_peak_per_wk"])
            wh_r = hpc_r * eff_freq

            # Deep
            if pd.notna(r.get("deep_minutes_per_clean_override", np.nan)):
                hpc_d = float(r["deep_minutes_per_clean_override"]) / 60.0 * productivity_modifier
            else:
                hpc_d = hours_per_clean_area(sqm, at, sampled_deep) * productivity_modifier
            wh_d = hpc_d * (r["freq_detailed_per_wk"] if pd.notna(r["freq_detailed_per_wk"]) else 0.0)

            # Multipliers
            if base_scenario == 'Advanced':
                svc = adv_service_map.get(r['package'], default_service)
                cpx = adv_complex_map.get(r['package'], default_complex)
                wh = apply_multipliers(wh_r, svc, cpx, special_mult) + apply_multipliers(wh_d, svc, cpx, special_mult)
            else:
                wh = apply_multipliers(wh_r, default_service, default_complex) + apply_multipliers(wh_d, default_service, default_complex)

            weekly_hours_total += wh

        fte = weekly_hours_total / (weekly_hours_contract * efficiency_factor)
        results_fte[i] = fte
        budget = _calc_budget_from_fte(fte, hourly_wage, include_benefits, num_robots, smart_facility)
        results_budget[i] = budget["total_budget"]

    return results_fte, results_budget

# =============================
# Streamlit App
# =============================
st.set_page_config(page_title="Crown Sydney FTE Calculator", layout="wide")
st.title("🏙️ Crown Sydney FTE Calculator")
st.caption("SQM/productivity like Melbourne, with a simple Peak Days slider and a Staffing Curve tab.")

with st.sidebar:
    st.header("📁 Data")
    file = st.file_uploader("Upload Crown Sydney Scope (xlsx/csv)", type=["xlsx", "xls", "csv"])  

    st.divider()
    st.header("⚙️ Parameters")
    productivity_modifier = st.slider("Productivity Adjustment", 0.7, 1.3, 1.0, 0.05)
    weekly_hours_contract = st.number_input("Standard Weekly Hours", 35, 45, 40)
    efficiency_factor = st.slider("Efficiency Factor", 0.7, 0.95, 0.85, 0.05)

    st.subheader("Service & Complexity (applies to all unless overridden later)")
    default_service = st.selectbox("Service Level", list(SERVICE_LEVELS.keys()), index=1)
    default_complex = st.selectbox("Complexity", list(COMPLEXITY_FACTORS.keys()), index=1)

    st.divider()
    st.header("📈 Peak Mixing")
    preset = st.radio("Preset", ["Quiet (1)", "Normal (3)", "Event/Holiday (5)"] , index=1, horizontal=True)
    preset_map = {"Quiet (1)": 1, "Normal (3)": 3, "Event/Holiday (5)": 5}
    peak_days_per_week = st.slider("Peak days per week", 0, 7, preset_map[preset])

    st.divider()
    st.header("💵 Budget")
    hourly_wage = st.number_input("Hourly Wage (AUD)", 20.0, 60.0, 25.0, 0.5)
    include_benefits = st.checkbox("Include Benefits & Overhead (+30%)", True)
    num_robots = st.number_input("Robots (units)", 0, 50, 0)
    smart_facility = st.checkbox("Smart Facility Systems", False)
    operational_pattern = st.selectbox("Operational Pattern", ["24/7", "16 hours", "8 hours"], index=0)

# Main workflow
if not file:
    st.info("👈 Upload the Sydney scope to begin. Expected columns include: 'Package', 'Building', 'Level', 'Location', 'SQM', 'Detailed Clean Frequency (# per week) Standard', 'NON-PEAK Freq (# per week) Standard', 'PEAK Freq (# per week) Standard'.")
    st.stop()

# Load
import os
name = getattr(file, 'name', 'uploaded')
ext = os.path.splitext(name)[1].lower()
if ext == '.csv':
    raw = pd.read_csv(file)
else:
    raw = pd.read_excel(file)
raw = clean_columns(raw)
raw = map_columns(raw)

# Verify
required = ["package", "building", "level", "location", "sqm", "freq_detailed_raw", "freq_nonpeak_per_wk", "freq_peak_per_wk"]
missing = [c for c in required if c not in raw.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.dataframe(pd.DataFrame({"Available": raw.columns}))
    st.stop()

# Clean values
df = raw.copy()
df["sqm"] = pd.to_numeric(df["sqm"], errors="coerce").fillna(0)
df["freq_nonpeak_per_wk"] = pd.to_numeric(df["freq_nonpeak_per_wk"], errors="coerce")
df["freq_peak_per_wk"] = pd.to_numeric(df["freq_peak_per_wk"], errors="coerce")

# Interpret Detailed field
freq_det, mins_det, interp_det = [], [], []
for v in df["freq_detailed_raw"].tolist():
    f, m, tag = parse_detailed_field(v)
    freq_det.append(f)
    mins_det.append(m)
    interp_det.append(tag)

df["freq_detailed_per_wk"] = freq_det
df["deep_minutes_per_clean_override"] = mins_det
df["detailed_interpretation"] = interp_det

# Area type

df["area_type"] = [classify_area_type(r["location"], r.get("package", "")) for _, r in df.iterrows()]

# Hours per clean (routine and deep)

df["hours_per_clean_routine"] = [
    hours_per_clean_area(r["sqm"], r["area_type"], PRODUCTIVITY_RATES_SQM_PER_HR) * productivity_modifier
    for _, r in df.iterrows()
]

# Deep via override minutes if present, else deep productivity
hpc_deep = []
for _, r in df.iterrows():
    if pd.notna(r["deep_minutes_per_clean_override"]):
        hpc_deep.append(float(r["deep_minutes_per_clean_override"]) / 60.0)
    else:
        hpc_deep.append(hours_per_clean_area(r["sqm"], r["area_type"], DEEP_PRODUCTIVITY_RATES_SQM_PER_HR) * productivity_modifier)

df["hours_per_clean_deep"] = hpc_deep

# Effective weekly frequency (routine)

df["effective_routine_freq_per_wk"] = [
    effective_weekly_frequency(peak_days_per_week, r["freq_nonpeak_per_wk"], r["freq_peak_per_wk"])
    for _, r in df.iterrows()
]

# Weekly hours split

df["weekly_hours_routine"] = df["hours_per_clean_routine"] * df["effective_routine_freq_per_wk"]
df["weekly_hours_detailed"] = df["hours_per_clean_deep"] * df["freq_detailed_per_wk"]

# Apply global service/complexity multipliers (kept simple like MEL)
special_mult = 1.0

df["weekly_hours_routine"] = [apply_multipliers(h, default_service, default_complex, special_mult) for h in df["weekly_hours_routine"]]
df["weekly_hours_detailed"] = [apply_multipliers(h, default_service, default_complex, special_mult) for h in df["weekly_hours_detailed"]]

# Totals

df["weekly_hours_total"] = df["weekly_hours_routine"] + df["weekly_hours_detailed"]

total_weekly_hours = float(df["weekly_hours_total"].sum())
fte_total = total_weekly_hours / (weekly_hours_contract * efficiency_factor)

# --- Mode Tabs (visual, like Melbourne) ---
st.markdown("---")
mode_tab1, mode_tab2, mode_tab3 = st.tabs(["Standard Calculation", "Advanced Mode", "Risk (Monte Carlo)"])
with mode_tab1:
    st.caption("Standard Calculation uses the parameters in the sidebar. Results are shown below.")

    std_tabs = st.tabs(["Summary", "Package Analysis", "Building Analysis", "Area Type Analysis", "Staffing Curve", "Budget"]) 
    (summary, by_package, by_building, by_areatype, staffing, budget) = std_tabs

    with summary:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total FTE Required", f"{fte_total:.1f}")
        with col2:
            st.metric("Total Areas", f"{len(df):,}")
        with col3:
            st.metric("Total Area (SQM)", f"{df['sqm'].sum():,.0f}")
        with col4:
            st.metric("Peak Days/Week", f"{peak_days_per_week}")

        st.subheader("Top 10 by Weekly Hours")
        top10 = df.sort_values("weekly_hours_total", ascending=False).head(10)[
            ["location", "package", "sqm", "area_type", "effective_routine_freq_per_wk", "freq_detailed_per_wk", "weekly_hours_routine", "weekly_hours_detailed", "weekly_hours_total"]
        ].rename(columns={
            "location": "Location",
            "package": "Package",
            "sqm": "SQM",
            "area_type": "Area Type",
            "effective_routine_freq_per_wk": "Routine Freq (eff/wk)",
            "freq_detailed_per_wk": "Detailed Freq (/wk)",
            "weekly_hours_routine": "Routine Hrs/Wk",
            "weekly_hours_detailed": "Detailed Hrs/Wk",
            "weekly_hours_total": "Total Hrs/Wk",
        })
        top10[["Routine Hrs/Wk","Detailed Hrs/Wk","Total Hrs/Wk"]] = top10[["Routine Hrs/Wk","Detailed Hrs/Wk","Total Hrs/Wk"]].round(2)
        st.dataframe(top10, use_container_width=True, hide_index=True)

    with by_package:
        g = df.groupby("package").agg(
            Total_FTE=("weekly_hours_total", lambda s: s.sum() / (weekly_hours_contract * efficiency_factor)),
            Total_SQM=("sqm", "sum"),
            Areas=("location", "count"),
            Weekly_Hours=("weekly_hours_total", "sum"),
        ).sort_values("Total_FTE", ascending=False)
        st.dataframe(g.round(2), use_container_width=True)
        fig = px.bar(g.reset_index(), x="package", y="Total_FTE", title="FTE by Package", color="Total_FTE", color_continuous_scale="Blues")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with by_building:
        g = df.groupby("building").agg(
            Total_FTE=("weekly_hours_total", lambda s: s.sum() / (weekly_hours_contract * efficiency_factor)),
            Total_SQM=("sqm", "sum"),
            Areas=("location", "count"),
            Weekly_Hours=("weekly_hours_total", "sum"),
        ).sort_values("Total_FTE", ascending=False)
        st.dataframe(g.round(2), use_container_width=True)
        fig = px.bar(g.reset_index(), x="building", y="Total_FTE", title="FTE by Building", color="Total_FTE", color_continuous_scale="Viridis")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with by_areatype:
        g = df.groupby("area_type").agg(
            Total_FTE=("weekly_hours_total", lambda s: s.sum() / (weekly_hours_contract * efficiency_factor)),
            Total_SQM=("sqm", "sum"),
            Areas=("location", "count"),
            Weekly_Hours=("weekly_hours_total", "sum"),
        ).sort_values("Total_FTE", ascending=False)
        st.dataframe(g.round(2), use_container_width=True)
        rates_df = pd.DataFrame.from_dict(PRODUCTIVITY_RATES_SQM_PER_HR, orient="index", columns=["SQM/Hour"]).rename_axis("Area Type").reset_index()
        fig = px.bar(rates_df, x="Area Type", y="SQM/Hour", title="Routine Productivity Rates")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with staffing:
        st.markdown("### Weekly Staffing Curve (Routine vs Detailed)")
        routine_days = df["weekly_hours_routine"].sum()
        detailed_days = df["weekly_hours_detailed"].sum()
        day_split_r = distribute_weekly_hours(routine_days, DEFAULT_DAY_WEIGHTS)
        day_split_d = distribute_weekly_hours(detailed_days, DEFAULT_DAY_WEIGHTS)
        curve = pd.DataFrame({
            "Day": list(DEFAULT_DAY_WEIGHTS.keys()),
            "Routine Hrs": [day_split_r[d] for d in DEFAULT_DAY_WEIGHTS.keys()],
            "Detailed Hrs": [day_split_d[d] for d in DEFAULT_DAY_WEIGHTS.keys()],
        })
        curve["Total Hrs"] = curve["Routine Hrs"] + curve["Detailed Hrs"]
        st.dataframe(curve.round(2), use_container_width=True, hide_index=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Routine", x=curve["Day"], y=curve["Routine Hrs"]))
        fig.add_trace(go.Bar(name="Detailed", x=curve["Day"], y=curve["Detailed Hrs"]))
        fig.update_layout(barmode="stack", title="Weekly Hours by Day")
        st.plotly_chart(fig, use_container_width=True)
        rows = []
        for _, r in curve.iterrows():
            for shift, w in DEFAULT_SHIFT_WEIGHTS.items():
                hrs = r["Total Hrs"] * w
                fte = hrs / (weekly_hours_contract * efficiency_factor)
                rows.append({"Day": r["Day"], "Shift": shift, "Hours": hrs, "FTE": fte})
        shift_tbl = pd.DataFrame(rows)
        st.subheader("Recommended FTE by Day & Shift")
        pivot = shift_tbl.pivot(index="Day", columns="Shift", values="FTE").reindex(list(DEFAULT_DAY_WEIGHTS.keys()))
        st.dataframe(pivot.round(2), use_container_width=True)

    with budget:
        st.subheader("Budget Analysis")
        annual_hours = 52 * 38
        base_salary = fte_total * hourly_wage * annual_hours
        labor_cost = base_salary * (1.3 if include_benefits else 1.0)
        supplies_cost = labor_cost * 0.15
        robot_cost = num_robots * 35000
        smart_cost = 50000 if smart_facility else 0
        mgmt_cost = labor_cost * 0.10
        total_budget = labor_cost + supplies_cost + robot_cost + smart_cost + mgmt_cost
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Annual Budget", f"${total_budget:,.0f}")
        c2.metric("Labor Cost", f"${labor_cost:,.0f}")
        c3.metric("Cost per FTE", f"${(labor_cost / fte_total) if fte_total>0 else 0:,.0f}")
        bdf = pd.DataFrame({
            "Category": ["Labor", "Supplies", "Robots", "Smart Systems", "Management"],
            "Annual Cost": [labor_cost, supplies_cost, robot_cost, smart_cost, mgmt_cost],
        })
        bdf["%"] = (bdf["Annual Cost"] / total_budget * 100).round(1)
        st.dataframe(bdf.style.format({"Annual Cost": "${:,.0f}", "%": "{:.1f}%"}), use_container_width=True, hide_index=True)
        fig = px.bar(bdf, x="Category", y="Annual Cost", title="Annual Budget by Category", text="Annual Cost")
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

with mode_tab2:
    st.markdown("### Advanced Mode — per‑package service & complexity, plus special conditions")
    st.caption("Advanced uses a hybrid timing model: SQM/productivity blended with fixture-density (if provided). Multipliers apply to both routine and detailed hours.")

    # Build package lists
    packages = sorted([p for p in df['package'].dropna().unique().tolist() if str(p).strip()])

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Service Level per Package")
        adv_service = {}
        for p in packages:
            adv_service[p] = st.selectbox(f"{p}", list(SERVICE_LEVELS.keys()), index=1, key=f"adv_service_{p}")
    with colB:
        st.subheader("Complexity per Package")
        adv_complex = {}
        for p in packages:
            adv_complex[p] = st.selectbox(f"{p}", list(COMPLEXITY_FACTORS.keys()), index=1, key=f"adv_complex_{p}")

    st.subheader("Fixture Density (per 100 sqm)")
    cfd1, cfd2, cfd3 = st.columns(3)
    with cfd1:
        dens_toilet = st.number_input("Toilet fixtures", 0, 50, 10, help="Total of cubicles+urinals+sinks per 100 sqm")
    with cfd2:
        dens_gaming = st.number_input("Gaming machines", 0, 50, 7)
    with cfd3:
        dens_desks = st.number_input("Office desks", 0, 50, 4)

    st.subheader("Special Conditions (global multipliers)")
    ca, cb, cc = st.columns(3)
    with ca:
        sc_high_traffic = st.checkbox("High traffic (+15%)", value=False)
        sc_food = st.checkbox("Food service (+20%)", value=False)
    with cb:
        sc_medical = st.checkbox("Medical grade (+30%)", value=False)
        sc_difficult = st.checkbox("Difficult access (+10%)", value=False)
    with cc:
        sc_automated = st.checkbox("Automated equipment (−20%)", value=False)
        sc_green = st.checkbox("Green cleaning (+5%)", value=False)

    special_mult = 1.0
    if sc_high_traffic: special_mult *= 1.15
    if sc_food: special_mult *= 1.20
    if sc_medical: special_mult *= 1.30
    if sc_difficult: special_mult *= 1.10
    if sc_automated: special_mult *= 0.80
    if sc_green: special_mult *= 1.05

    run_adv = st.button("Calculate Advanced FTE", type="primary")

    if run_adv:
        # create working copy
        df_adv = df.copy()
        # Persist settings for MC tab
        st.session_state['adv_service_map'] = adv_service
        st.session_state['adv_complex_map'] = adv_complex
        st.session_state['adv_special_mult'] = special_mult
        st.session_state['adv_dens'] = {"toilet": dens_toilet, "gaming": dens_gaming, "desks": dens_desks}

        # Per‑package multipliers
        svc_map = df_adv['package'].map(adv_service).fillna('Standard')
        cpx_map = df_adv['package'].map(adv_complex).fillna('Medium')
        svc_mult = svc_map.map(SERVICE_LEVELS).fillna(1.0)
        cpx_mult = cpx_map.map(COMPLEXITY_FACTORS).fillna(1.0)
        pkg_mult = (svc_mult * cpx_mult * special_mult)

        # Hybrid hours per clean (routine): 40% area-based + 60% item-based if density applies
        hybrid_hpc = []
        for _, r in df_adv.iterrows():
            area_based = r['hours_per_clean_routine']
            item_hours = 0.0
            sqm = float(r['sqm']) if pd.notna(r['sqm']) else 0.0
            at = r['area_type']
            if at == 'Toilet' and dens_toilet > 0:
                num = (sqm / 100.0) * dens_toilet
                mins = num * (0.6*TASK_TIMES_MIN['toilet_cubicle'] + 0.2*TASK_TIMES_MIN['urinal'] + 0.2*TASK_TIMES_MIN['sink'])
                item_hours = mins / 60.0
            elif at == 'Gaming' and dens_gaming > 0:
                num = (sqm / 100.0) * dens_gaming
                item_hours = (num * TASK_TIMES_MIN['gaming_machine']) / 60.0
            elif at == 'Office' and dens_desks > 0:
                num = (sqm / 100.0) * dens_desks
                item_hours = (num * TASK_TIMES_MIN['office_desk']) / 60.0
            if item_hours > 0:
                hybrid_hpc.append(area_based*0.4 + item_hours*0.6)
            else:
                hybrid_hpc.append(area_based)
        df_adv['hours_per_clean_routine_adv'] = hybrid_hpc

        # Weekly hours (apply effective routine freq and detailed same as standard)
        df_adv['weekly_hours_routine_adv'] = df_adv['hours_per_clean_routine_adv'] * df_adv['effective_routine_freq_per_wk']
        df_adv['weekly_hours_detailed_adv'] = df_adv['hours_per_clean_deep'] * df_adv['freq_detailed_per_wk']

        # Apply multipliers
        df_adv['weekly_hours_routine_adv'] = [apply_multipliers(h, svc_map.iloc[i], cpx_map.iloc[i], special_mult) for i,h in enumerate(df_adv['weekly_hours_routine_adv'])]
        df_adv['weekly_hours_detailed_adv'] = [apply_multipliers(h, svc_map.iloc[i], cpx_map.iloc[i], special_mult) for i,h in enumerate(df_adv['weekly_hours_detailed_adv'])]

        df_adv['weekly_hours_total_adv'] = df_adv['weekly_hours_routine_adv'] + df_adv['weekly_hours_detailed_adv']

        fte_adv = float(df_adv['weekly_hours_total_adv'].sum()) / (weekly_hours_contract * efficiency_factor)
        fte_std = float(df['weekly_hours_total'].sum()) / (weekly_hours_contract * efficiency_factor)

        # Header metrics like MEL Advanced
        st.markdown("### 📊 Advanced Calculation Results")
        mA1, mA2, mA3 = st.columns(3)
        mA1.metric("Standard FTE", f"{fte_std:.1f}")
        mA2.metric("Advanced FTE", f"{fte_adv:.1f}")
        diff = fte_adv - fte_std
        mA3.metric("Difference", f"{diff:+.1f}", f"{(diff/max(fte_std,1e-6))*100:+.1f}%")

        # Sub-tabs inside Advanced like MEL: Summary, Package, Building, Area Type, Budget
        adv_tabs = st.tabs(["Summary", "Package Analysis", "Building Analysis", "Area Type Analysis", "Budget"]) 
        (adv_sum, adv_pkg, adv_bld, adv_area, adv_budget) = adv_tabs

        total_hours_adv = float(df_adv['weekly_hours_total_adv'].sum())
        avg_freq = float(df_adv['effective_routine_freq_per_wk'].mean()) if 'effective_routine_freq_per_wk' in df_adv else 0.0
        # Shift recommendations using default shift weights
        morning_fte = total_hours_adv * DEFAULT_SHIFT_WEIGHTS['Morning (06-14)'] / (weekly_hours_contract*efficiency_factor)
        afternoon_fte = total_hours_adv * DEFAULT_SHIFT_WEIGHTS['Afternoon (14-22)'] / (weekly_hours_contract*efficiency_factor)
        night_fte = total_hours_adv * DEFAULT_SHIFT_WEIGHTS['Night (22-06)'] / (weekly_hours_contract*efficiency_factor)

        with adv_sum:
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Total FTE Required", f"{fte_adv:.1f}")
            a2.metric("Total Areas", f"{len(df_adv):,}")
            a3.metric("Total Area (SQM)", f"{df_adv['sqm'].sum():,.0f}")
            a4.metric("Avg Frequency/Week", f"{avg_freq:.1f}")

            st.markdown("### 🧑‍🤝‍🧑 Shift Recommendations")
            s1, s2, s3 = st.columns(3)
            s1.markdown("**Morning (6am–2pm)**\n\n{:.1f} FTE".format(morning_fte))
            s2.markdown("**Afternoon (2pm–10pm)**\n\n{:.1f} FTE".format(afternoon_fte))
            s3.markdown("**Night (10pm–6am)**\n\n{:.1f} FTE".format(night_fte))

            # Impact pills
            lux_cnt = sum(1 for p,v in adv_service.items() if v in ("Premium","Luxury"))
            sc_cnt = sum([sc_high_traffic, sc_food, sc_medical, sc_difficult, sc_automated, sc_green])
            # simple confidence heuristic
            missing_sqm_ratio = float((df_adv['sqm']<=0).sum())/max(len(df_adv),1)
            missing_freq_ratio = float(df_adv['effective_routine_freq_per_wk'].isna().sum())/max(len(df_adv),1)
            confidence = max(0.5, 1.0 - 0.3*missing_sqm_ratio - 0.2*missing_freq_ratio) * 100
            cA, cB, cC = st.columns(3)
            cA.info(f"Premium/Luxury Packages: {lux_cnt}")
            cB.info(f"Special Conditions Applied: {sc_cnt}")
            cC.info(f"Calculation Confidence: {confidence:.0f}%")

            st.markdown("### 🔝 Top 10 Areas by FTE Requirement")
            topA = df_adv.sort_values('weekly_hours_total_adv', ascending=False).head(10)[[
                'location','package','sqm','area_type','effective_routine_freq_per_wk','weekly_hours_total_adv'
            ]].rename(columns={'location':'Location','package':'Package','sqm':'SQM','area_type':'Area Type','effective_routine_freq_per_wk':'Frequency/Week','weekly_hours_total_adv':'FTE Required (hrs/wk)'})
            st.dataframe(topA.round(2), use_container_width=True, hide_index=True)

        with adv_pkg:
            gA = df_adv.groupby('package').agg(
                Total_FTE_Adv=("weekly_hours_total_adv", lambda s: s.sum()/(weekly_hours_contract*efficiency_factor)),
                Total_SQM=("sqm","sum"), Areas=("location","count"), Weekly_Hours_Adv=("weekly_hours_total_adv","sum")
            ).sort_values('Total_FTE_Adv', ascending=False).round(2)
            st.dataframe(gA, use_container_width=True)

        with adv_bld:
            gB = df_adv.groupby('building').agg(
                Total_FTE_Adv=("weekly_hours_total_adv", lambda s: s.sum()/(weekly_hours_contract*efficiency_factor)),
                Total_SQM=("sqm","sum"), Areas=("location","count"), Weekly_Hours_Adv=("weekly_hours_total_adv","sum")
            ).sort_values('Total_FTE_Adv', ascending=False).round(2)
            st.dataframe(gB, use_container_width=True)

        with adv_area:
            gT = df_adv.groupby('area_type').agg(
                Total_FTE_Adv=("weekly_hours_total_adv", lambda s: s.sum()/(weekly_hours_contract*efficiency_factor)),
                Total_SQM=("sqm","sum"), Areas=("location","count"), Weekly_Hours_Adv=("weekly_hours_total_adv","sum")
            ).sort_values('Total_FTE_Adv', ascending=False).round(2)
            st.dataframe(gT, use_container_width=True)

        with adv_budget:
            annual_hours = 52 * 38
            base_salary = fte_adv * hourly_wage * annual_hours
            labor_cost = base_salary * (1.3 if include_benefits else 1.0)
            supplies_cost = labor_cost * 0.15
            robot_cost = num_robots * 35000
            smart_cost = 50000 if smart_facility else 0
            mgmt_cost = labor_cost * 0.10
            total_budget = labor_cost + supplies_cost + robot_cost + smart_cost + mgmt_cost
            st.metric("Total Annual Budget (Adv)", f"${total_budget:,.0f}")

        # (Keep legacy table at the end if needed)
        st.markdown("#### Top 10 Areas (Advanced)")
        topA = df_adv.sort_values('weekly_hours_total_adv', ascending=False).head(10)[[
            'location','package','sqm','area_type','weekly_hours_routine_adv','weekly_hours_detailed_adv','weekly_hours_total_adv'
        ]].rename(columns={
            'location':'Location','package':'Package','sqm':'SQM','area_type':'Area Type',
            'weekly_hours_routine_adv':'Routine Hrs/Wk (Adv)','weekly_hours_detailed_adv':'Detailed Hrs/Wk (Adv)','weekly_hours_total_adv':'Total Hrs/Wk (Adv)'
        })
        st.dataframe(topA.round(2), use_container_width=True, hide_index=True)

        st.markdown("#### Package Summary (Advanced)")
        gA = df_adv.groupby('package').agg(
            Total_FTE_Adv=("weekly_hours_total_adv", lambda s: s.sum()/(weekly_hours_contract*efficiency_factor)),
            Total_SQM=("sqm","sum"),
            Areas=("location","count"),
            Weekly_Hours_Adv=("weekly_hours_total_adv","sum")
        ).sort_values('Total_FTE_Adv', ascending=False).round(2)
        st.dataframe(gA, use_container_width=True)


with mode_tab3:
    st.markdown("### Risk (Monte Carlo)")
    st.caption("Turn your point estimate into a probability range. Choose a base scenario, set uncertainties, then simulate.")

    base_choice = st.radio("Base scenario", ["Standard", "Advanced"], index=0, horizontal=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        n_iter = st.slider("Simulations", 200, 3000, 1000, 100)
    with c2:
        prod_cv = st.slider("Productivity variability (±%)", 0, 30, 15) / 100.0
    with c3:
        dens_cv = st.slider("Fixture density variability (±%)", 0, 40, 0) / 100.0

    run_mc = st.button("Run Monte Carlo", type="primary")

    if run_mc:
        # Pull Advanced settings from session if available
        adv_service_map = st.session_state.get('adv_service_map', {})
        adv_complex_map = st.session_state.get('adv_complex_map', {})
        special_mult = st.session_state.get('adv_special_mult', 1.0)
        adv_dens = st.session_state.get('adv_dens', {"toilet":10, "gaming":7, "desks":4})

        ftes, budgets = run_monte_carlo_sydney(
            df=df,
            base_scenario=base_choice,
            peak_days_per_week=peak_days_per_week,
            prod_cv=prod_cv,
            deep_cv=prod_cv*1.33,  # deep a bit more variable
            dens_cv=dens_cv,
            productivity_modifier=productivity_modifier,
            weekly_hours_contract=weekly_hours_contract,
            efficiency_factor=efficiency_factor,
            default_service=default_service,
            default_complex=default_complex,
            n_iter=n_iter,
            hourly_wage=hourly_wage,
            include_benefits=include_benefits,
            num_robots=num_robots,
            smart_facility=smart_facility,
            adv_service_map=adv_service_map,
            adv_complex_map=adv_complex_map,
            special_mult=special_mult,
            dens_toilet=adv_dens.get('toilet', 10),
            dens_gaming=adv_dens.get('gaming', 7),
            dens_desks=adv_dens.get('desks', 4),
        )
        p10, p50, p80, p90 = np.percentile(ftes, [10, 50, 80, 90])
        b10, b50, b80, b90 = np.percentile(budgets, [10, 50, 80, 90])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("FTE P50", f"{p50:.1f}")
        m2.metric("FTE P10–P90", f"{p10:.1f} – {p90:.1f}")
        m3.metric("Budget P50", f"${b50:,.0f}")
        m4.metric("Budget P10–P90", f"${b10:,.0f} – ${b90:,.0f}")
        st.caption(f"Suggested planning level: **P80** → {p80:.1f} FTE, budget ≈ ${b80:,.0f}")
        fig_f = px.histogram(x=ftes, nbins=40, title="FTE Distribution")
        st.plotly_chart(fig_f, use_container_width=True)
        fig_b = px.histogram(x=budgets, nbins=40, title="Budget Distribution")
        st.plotly_chart(fig_b, use_container_width=True)

# =============================
# Tabs
# =============================

