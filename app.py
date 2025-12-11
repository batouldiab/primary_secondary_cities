#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import hashlib
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ==========================
# Page & Chrome Configuration
# ==========================
st.set_page_config(
    page_title="Job Posts & Skills Map",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Modern, compact CSS styling
st.markdown(
    """
    <style>
    /* Hide sidebar and streamlit branding */
    div[data-testid="stSidebar"] { display: none !important; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    /* Compact radio buttons */
    div[data-testid="stRadio"] > div {
        flex-direction: row !important;
        gap: 1rem;
    }
    div[data-testid="stRadio"] > div > label {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        font-weight: 500;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    div[data-testid="stRadio"] > div > label:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    div[data-testid="stRadio"] > div > label[data-checked="true"] {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);
    }
    /* Compact selectbox styling */
    div[data-testid="stSelectbox"] {
        max-width: 280px;
    }
    div[data-testid="stSelectbox"] > div > div {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        font-size: 0.9rem;
    }
    div[data-testid="stSelectbox"] label {
        font-weight: 600;
        color: #333;
        font-size: 0.85rem;
    }
    /* Modern data tables */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    div[data-testid="stDataFrame"] table {
        font-size: 0.85rem;
    }
    div[data-testid="stDataFrame"] th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600;
        padding: 0.75rem 1rem !important;
    }
    div[data-testid="stDataFrame"] td {
        padding: 0.6rem 1rem !important;
    }
    div[data-testid="stDataFrame"] tr:nth-child(even) {
        background: #f8f9fa;
    }
    div[data-testid="stDataFrame"] tr:hover {
        background: #e8f4f8 !important;
    }
    /* Skills panel headers */
    .skills-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        font-weight: 600;
        font-size: 1rem;
    }
    .skills-subheader {
        color: #555;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    /* Compact columns */
    div[data-testid="column"] {
        padding: 0 0.5rem;
    }
    /* Version info badge */
    .version-badge {
        display: inline-block;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-left: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================
# Config
# ==========================
UNKNOWN_SHEET_NAME = "Unknown_Country"

# Visual bubble defaults (meters) — for red circles
DEFAULT_COUNTRY_MIN_RADIUS = 400
DEFAULT_COUNTRY_MAX_RADIUS = 10000

# When Country = "All" (bigger bubbles)
DEFAULT_GLOBAL_MIN_RADIUS_ALL = 4000
DEFAULT_GLOBAL_MAX_RADIUS_ALL = 300000

# Skills aggregation fallback radius (km)
DEFAULT_FALLBACK_RADIUS_KM = 10

# Static always-label cities
ALWAYS_LABEL_CITIES_STATIC: Set[Tuple[str, str]] = set()

# ==========================
# Helpers
# ==========================
EXPECTED_SKILLS_COLS = ["Skill", "SkillType", "City", "Latitude", "Longitude", "count"]
COL_ALIASES = {
    "skill": "Skill",
    "skill type": "SkillType",
    "skilltype": "SkillType",
    "type": "SkillType",
    "city": "City",
    "latitude": "Latitude",
    "lat": "Latitude",
    "longitude": "Longitude",
    "lon": "Longitude",
    "long": "Longitude",
    "count": "count",
    "n": "count",
    "freq": "count",
    "frequency": "count",
}


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        key = re.sub(r"\s+", " ", str(c)).strip().lower()
        mapping[c] = COL_ALIASES.get(key, c)
    return df.rename(columns=mapping)


def _parse_int_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(s):
        return s
    if pd.api.types.is_float_dtype(s):
        return s.fillna(0).astype(int)
    s = s.astype(str).str.replace(r"[^\d]", "", regex=True)
    s = s.replace("", np.nan)
    return s.fillna(0).astype(int)


def _safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _strip_weird_whitespace(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def _clean_punctuation(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(r"^[\"':,\s]+", "", regex=True)
        .str.replace(r"[\"':,\s]+$", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _normcase(x: Optional[str]) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x).replace("_", " ")).strip().casefold()


def _scale_radius_by_global_sum(
    counts: pd.Series, global_sum: float, min_r: int, max_r: int
) -> np.ndarray:
    denom = global_sum if global_sum > 0 else 1.0
    proportions = counts.astype(float).clip(lower=0) / denom
    return (min_r + proportions * (max_r - min_r)).to_numpy()


def _haversine_km_vec(lat_arr, lon_arr, lat0, lon0) -> np.ndarray:
    R = 6371.0088
    lat1 = np.radians(lat_arr.astype(float))
    lon1 = np.radians(lon_arr.astype(float))
    lat2 = np.radians(float(lat0))
    lon2 = np.radians(float(lon0))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def _best_excel_engine() -> str:
    try:
        import calamine  # noqa: F401
        return "calamine"
    except Exception:
        return "openpyxl"


def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


# ==========================
# Paths
# ==========================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "mapping/data"

PRIMARY_SECONDARY_JSON_PATH = ROOT / "primary_secondary_cities.json"
SKILLS_XLSX_PATH = DATA_DIR / "cities_skills.xlsx"
AGGREGATED_XLSX_PATH = DATA_DIR / "all_OJAs_aggregated.xlsx"
SKILLS_AVAILABLE = SKILLS_XLSX_PATH.exists()

# ==========================
# Data Loading Functions
# ==========================
@st.cache_data(show_spinner=False)
def load_aggregated_all_sheets(path: Path, mtime: float) -> Dict[str, pd.DataFrame]:
    """Load all_OJAs_aggregated.xlsx with sheets per country."""
    if not path.exists():
        return {}
    engine = _best_excel_engine()
    sheets = pd.read_excel(path, sheet_name=None, engine=engine)
    out: Dict[str, pd.DataFrame] = {}
    for country, df in (sheets or {}).items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        rename_map = {}
        for col in list(df.columns):
            cl = str(col).strip().lower()
            if cl == "city":
                rename_map[col] = "City"
            elif cl == "latitude":
                rename_map[col] = "Latitude"
            elif cl == "longitude":
                rename_map[col] = "Longitude"
            elif cl == "count":
                rename_map[col] = "Count"
        df = df.rename(columns=rename_map)

        needed = ["City", "Latitude", "Longitude", "Count"]
        if any(k not in df.columns for k in needed):
            continue

        df["City"] = df["City"].astype(str).str.strip()
        df = df[df["City"].notna() & (df["City"] != "") & (df["City"] != "nan")].copy()

        if df.empty:
            continue

        cdf = pd.DataFrame(
            {
                "Sheet": country,
                "City": df["City"],
                "__City_norm__": df["City"].map(_normcase),
                "lat": pd.to_numeric(df["Latitude"], errors="coerce"),
                "lon": pd.to_numeric(df["Longitude"], errors="coerce"),
                "Count": pd.to_numeric(df["Count"], errors="coerce").fillna(0).astype(int),
            }
        ).dropna(subset=["lat", "lon"])

        cdf = cdf[cdf["__City_norm__"].isin(set(cdf["__City_norm__"]) - {"", "nan"})].copy()
        out[country] = cdf

    return out


@st.cache_data(show_spinner=False)
def load_primary_secondary_cities(path: Path) -> pd.DataFrame:
    """Read city locations and radii from primary_secondary_cities.json."""
    if not path.exists():
        st.error(f"Missing required cities file: {path.name}")
        st.stop()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Could not read {path.name}: {exc}")
        st.stop()

    records = raw.get("cities") if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        st.error(f"{path.name} must contain a list of city records.")
        st.stop()

    df = pd.DataFrame(records)
    rename_map = {
        "Country": "country",
        "country": "country",
        "City": "Center_City",
        "city": "Center_City",
        "Radius": "Radius_km_max",
        "radius": "Radius_km_max",
        "Latitude": "latitude",
        "Longitude": "longitude",
    }
    df = df.rename(columns=rename_map)

    required_cols = ["country", "Center_City", "Radius_km_max", "latitude", "longitude"]
    if any(col not in df.columns for col in required_cols):
        st.error(f"{path.name} is missing required columns: {required_cols}")
        st.stop()

    df["country"] = _strip_weird_whitespace(df["country"]).replace("nan", "")
    df["Center_City"] = _strip_weird_whitespace(df["Center_City"]).replace("nan", "")

    df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["Radius_km_max"] = pd.to_numeric(
        df["Radius_km_max"], errors="coerce"
    ).fillna(DEFAULT_FALLBACK_RADIUS_KM)

    df["__City_norm__"] = df["Center_City"].apply(_normcase)
    df["__Country_norm__"] = df["country"].apply(_normcase)
    df = df.dropna(subset=["country", "Center_City", "lat", "lon"]).copy()

    df["Sheet"] = df["country"]
    return df


@st.cache_data(show_spinner=False)
def build_major_centers_with_aggregated_posts(
    centers_df: pd.DataFrame, aggregated_by_country_norm: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """Build center data with post counts from aggregated data."""
    out: Dict[str, pd.DataFrame] = {}
    if centers_df is None or centers_df.empty:
        return out

    for country, group in centers_df.groupby("country"):
        norm_country = _normcase(country)
        agg = aggregated_by_country_norm.get(norm_country)
        records = []
        for _, row in group.iterrows():
            city_norm = row["__City_norm__"]
            city_name = row["Center_City"]
            radius_km = row["Radius_km_max"]
            if radius_km is None or pd.isna(radius_km):
                radius_km = DEFAULT_FALLBACK_RADIUS_KM

            lat0 = float(row["lat"])
            lon0 = float(row["lon"])
            own_posts = 0
            included_posts = 0
            included_count = 0

            if agg is None or agg.empty:
                continue

            major_city_data = agg[agg["__City_norm__"] == city_norm]
            if major_city_data.empty:
                continue

            own_posts = int(major_city_data["Count"].iloc[0])
            if own_posts <= 0:
                continue

            distances = _haversine_km_vec(
                agg["lat"].to_numpy(),
                agg["lon"].to_numpy(),
                lat0,
                lon0,
            )
            within_radius_mask = (distances <= float(radius_km)) & (
                agg["__City_norm__"] != city_norm
            )
            locations_within = agg[within_radius_mask]
            included_posts = int(locations_within["Count"].sum())
            included_count = len(locations_within)

            total_posts = own_posts + included_posts

            records.append(
                {
                    "Sheet": country,
                    "Center_City": city_name,
                    "__City_norm__": city_norm,
                    "lat": lat0,
                    "lon": lon0,
                    "Radius_km_max": radius_km,
                    "Own_posts": own_posts,
                    "Total_posts": total_posts,
                    "Included_locations": included_count,
                }
            )

        if records:
            out[country] = pd.DataFrame.from_records(records)

    return out


@st.cache_data(show_spinner=False)
def compute_global_total_posts(centers_by_country: Dict[str, pd.DataFrame]) -> int:
    pieces = [
        df["Total_posts"]
        for df in centers_by_country.values()
        if isinstance(df, pd.DataFrame) and not df.empty
    ]
    return int(pd.concat(pieces, ignore_index=True).sum()) if pieces else 0


@st.cache_data(show_spinner=False)
def compute_total_posts_by_country(
    centers_by_country: Dict[str, pd.DataFrame]
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for country, df in centers_by_country.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            out[country] = int(df["Total_posts"].sum())
    return out


@st.cache_data(show_spinner=False)
def compute_total_from_aggregated(
    aggregated_by_country: Dict[str, pd.DataFrame]
) -> int:
    """Compute total posts from all aggregated data."""
    pieces = [
        df["Count"]
        for df in aggregated_by_country.values()
        if isinstance(df, pd.DataFrame) and not df.empty
    ]
    return int(pd.concat(pieces, ignore_index=True).sum()) if pieces else 0


@st.cache_data(show_spinner=False)
def compute_country_total_from_aggregated(
    aggregated_by_country: Dict[str, pd.DataFrame]
) -> Dict[str, int]:
    """Compute total posts per country from aggregated data."""
    out: Dict[str, int] = {}
    for country, df in aggregated_by_country.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            out[country] = int(df["Count"].sum())
    return out


# ==========================
# Skills workbook functions
# ==========================
@st.cache_resource(show_spinner=False)
def _skills_excel_handle(path: Path, mtime: float):
    engine = _best_excel_engine()
    return pd.ExcelFile(path, engine=engine)


@st.cache_data(show_spinner=False)
def skills_sheet_names(path: Path, mtime: float) -> List[str]:
    if not SKILLS_AVAILABLE or not path.exists():
        return []
    xf = _skills_excel_handle(path, mtime)
    return list(xf.sheet_names or [])


@st.cache_data(show_spinner=False)
def load_country_skills(path: Path, mtime: float, country: str) -> pd.DataFrame:
    if not SKILLS_AVAILABLE or not path.exists():
        return pd.DataFrame(columns=EXPECTED_SKILLS_COLS + ["__City_norm__"])
    xf = _skills_excel_handle(path, mtime)
    if country not in xf.sheet_names:
        return pd.DataFrame(columns=EXPECTED_SKILLS_COLS + ["__City_norm__"])

    usecols = EXPECTED_SKILLS_COLS
    df = xf.parse(sheet_name=country, usecols=usecols)

    df = _canonicalize_columns(df)
    if any(c not in df.columns for c in EXPECTED_SKILLS_COLS):
        return pd.DataFrame(columns=EXPECTED_SKILLS_COLS + ["__City_norm__"])

    df["Skill"] = _clean_punctuation(_strip_weird_whitespace(df["Skill"]))
    df["SkillType"] = _clean_punctuation(
        _strip_weird_whitespace(df.get("SkillType", ""))
    ).fillna("")
    df["City"] = _clean_punctuation(_strip_weird_whitespace(df["City"])).fillna("")
    df["Latitude"] = _safe_to_numeric(df["Latitude"])
    df["Longitude"] = _safe_to_numeric(df["Longitude"])
    df["count"] = _parse_int_series(df["count"])

    df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    df["__City_norm__"] = df["City"].apply(_normcase)
    df = df[~df["__City_norm__"].isin({"", "nan"})].copy()
    return df


@st.cache_data(show_spinner=False)
def build_city_geo_for_country(path: Path, mtime: float, country: str) -> pd.DataFrame:
    if not SKILLS_AVAILABLE or not path.exists():
        return pd.DataFrame(columns=["Sheet", "City", "__City_norm__", "lat", "lon", "count"])
    df = load_country_skills(path, mtime, country)
    if df.empty:
        return pd.DataFrame(columns=["Sheet", "City", "__City_norm__", "lat", "lon", "count"])

    w = df["count"].astype(float).clip(lower=0)
    df["_wlat"] = df["Latitude"] * w
    df["_wlon"] = df["Longitude"] * w

    agg = (
        df.groupby("City", as_index=False)
        .agg(count=("count", "sum"), _wlat=("_wlat", "sum"), _wlon=("_wlon", "sum"))
    )
    agg["lat"] = (agg["_wlat"] / agg["count"].replace(0, np.nan)).astype(float)
    agg["lon"] = (agg["_wlon"] / agg["count"].replace(0, np.nan)).astype(float)

    if agg["lat"].isna().any() or agg["lon"].isna().any():
        means = (
            df.groupby("City", as_index=False)[["Latitude", "Longitude"]]
            .mean()
            .rename(columns={"Latitude": "lat_mean", "Longitude": "lon_mean"})
        )
        agg = agg.merge(means, on="City", how="left")
        agg["lat"] = agg["lat"].fillna(agg["lat_mean"])
        agg["lon"] = agg["lon"].fillna(agg["lon_mean"])
        agg = agg.drop(columns=["lat_mean", "lon_mean"])

    agg["Sheet"] = country
    agg["__City_norm__"] = agg["City"].apply(_normcase)
    agg = agg.drop(columns=["_wlat", "_wlon"]).dropna(subset=["lat", "lon"])
    return agg[["Sheet", "City", "__City_norm__", "lat", "lon", "count"]].copy()


# ==========================
# Load Data
# ==========================
primary_secondary_df = load_primary_secondary_cities(PRIMARY_SECONDARY_JSON_PATH)

MAJOR_CITIES_DF = primary_secondary_df[
    ["country", "Center_City", "Radius_km_max"]
].copy()
MAJOR_CITIES_DF["__City_norm__"] = MAJOR_CITIES_DF["Center_City"].apply(
    lambda x: re.sub(r"\s+", " ", str(x)).strip().casefold() if x else ""
)
MAJOR_CITIES_DF["__Country_norm__"] = MAJOR_CITIES_DF["country"].apply(
    lambda x: re.sub(r"\s+", " ", str(x)).strip().casefold() if x else ""
)

HARDCODED_RADIUS_LOOKUP = {
    (row["__Country_norm__"], row["__City_norm__"]): row["Radius_km_max"]
    for _, row in MAJOR_CITIES_DF.iterrows()
}

if not AGGREGATED_XLSX_PATH.exists():
    st.error(f"Missing aggregated Excel file: {AGGREGATED_XLSX_PATH}")
    st.stop()

aggregated_by_country = load_aggregated_all_sheets(
    AGGREGATED_XLSX_PATH, _file_mtime(AGGREGATED_XLSX_PATH)
)
FILTERED_AGGREGATED = {
    k: v for k, v in aggregated_by_country.items() if k != UNKNOWN_SHEET_NAME
}
if not FILTERED_AGGREGATED:
    st.error(f"No valid sheets found in {AGGREGATED_XLSX_PATH.name}")
    st.stop()

FILTERED_AGGREGATED_NORM = {_normcase(k): v for k, v in FILTERED_AGGREGATED.items()}
MAJOR_CENTERS = build_major_centers_with_aggregated_posts(
    primary_secondary_df, FILTERED_AGGREGATED_NORM
)

GLOBAL_TOTAL_POSTS = compute_global_total_posts(MAJOR_CENTERS)
TOTAL_POSTS_BY_COUNTRY = compute_total_posts_by_country(MAJOR_CENTERS)
GLOBAL_TOTAL_AGGREGATED = compute_total_from_aggregated(FILTERED_AGGREGATED)
COUNTRY_TOTAL_AGGREGATED = compute_country_total_from_aggregated(FILTERED_AGGREGATED)
SKILLS_MTIME = _file_mtime(SKILLS_XLSX_PATH) if SKILLS_AVAILABLE else 0.0


def get_top_center_norm_pairs(centers: Dict[str, pd.DataFrame]) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    for sheet, df in centers.items():
        if df is None or df.empty:
            continue
        row = df.sort_values("Total_posts", ascending=False).head(1)
        if row.empty:
            continue
        cn = _normcase(sheet)
        cty = _normcase(str(row["Center_City"].iloc[0]).strip())
        if cty not in {"", "nan"}:
            out.add((cn, cty))
    return out


ALWAYS_LABEL_CITIES = {
    (_normcase(cn), _normcase(cty)) for (cn, cty) in ALWAYS_LABEL_CITIES_STATIC
} | get_top_center_norm_pairs(MAJOR_CENTERS)


@st.cache_data(show_spinner=False)
def get_center_cities_per_country(
    centers: Dict[str, pd.DataFrame]
) -> Dict[str, List[str]]:
    m: Dict[str, List[str]] = {}
    for ctry, df in centers.items():
        if df is None or df.empty:
            m[ctry] = []
            continue
        m[ctry] = df.sort_values("Total_posts", ascending=False)["Center_City"].tolist()
    return m


CENTER_CITIES_BY_COUNTRY = get_center_cities_per_country(MAJOR_CENTERS)


# ==========================
# Helper Functions
# ==========================
def _total_posts_for_scope(country: str) -> int:
    if not country or country == "All":
        return GLOBAL_TOTAL_POSTS
    return TOTAL_POSTS_BY_COUNTRY.get(country, GLOBAL_TOTAL_POSTS)


def _aggregated_total_for_scope(country: str) -> int:
    if not country or country == "All":
        return GLOBAL_TOTAL_AGGREGATED
    return COUNTRY_TOTAL_AGGREGATED.get(country, GLOBAL_TOTAL_AGGREGATED)


def resolve_center_radius_and_centroid(
    country: str, center_city: str
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    country_norm = _normcase(country)
    city_norm = _normcase(center_city)

    radius = HARDCODED_RADIUS_LOOKUP.get((country_norm, city_norm))
    if radius is None or pd.isna(radius):
        radius = DEFAULT_FALLBACK_RADIUS_KM

    df = MAJOR_CENTERS.get(country)
    if df is None or df.empty:
        return None, None, None
    r = df[df["__City_norm__"] == city_norm]
    if r.empty:
        return None, None, None
    return float(radius), float(r["lat"].iloc[0]), float(r["lon"].iloc[0])


def _label_layer_v1(df: pd.DataFrame, layer_id: str = "city-labels") -> Optional[pdk.Layer]:
    """Text layer for Version 1 - shows ALWAYS_LABEL_CITIES only."""
    if df is None or df.empty:
        return None
    label_df = df.copy()
    label_df["__country_norm__"] = label_df["Sheet"].apply(_normcase)
    label_df["__city_norm__"] = label_df["City"].apply(_normcase)
    label_df["__should_label__"] = label_df.apply(
        lambda row: (row["__country_norm__"], row["__city_norm__"]) in ALWAYS_LABEL_CITIES,
        axis=1,
    )
    subset = label_df[label_df["__should_label__"]].copy()
    if subset.empty:
        return None

    return pdk.Layer(
        "TextLayer",
        data=subset,
        id=layer_id,
        get_position="[lon, lat]",
        get_text="City",
        get_size=14,
        get_color=[0, 0, 0, 255],
        get_text_anchor="'middle'",
        get_alignment_baseline="'bottom'",
        background=False,
        pickable=True,
    )


def _label_layer_v2(df: pd.DataFrame, layer_id: str = "city-labels") -> Optional[pdk.Layer]:
    """Text layer for Version 2 - shows all primary/secondary city labels."""
    if df is None or df.empty:
        return None

    return pdk.Layer(
        "TextLayer",
        data=df,
        id=layer_id,
        get_position="[lon, lat]",
        get_text="Center_City",
        get_size=14,
        get_color=[0, 0, 0, 255],
        get_text_anchor="'middle'",
        get_alignment_baseline="'bottom'",
        background=False,
        pickable=False,
    )


# ==========================
# Skills Panel (Modern)
# ==========================
def render_skills_panel(
    country: str,
    city: str,
    include_cities_norm: Optional[Set[str]] = None,
    radius_km: Optional[float] = None,
) -> None:
    if not country or not city:
        return
    if not SKILLS_AVAILABLE:
        st.info("Skills workbook not found - unable to show skill breakdowns.")
        return

    df_country = load_country_skills(SKILLS_XLSX_PATH, SKILLS_MTIME, country)
    if df_country is None or df_country.empty:
        st.info(f"No skills data found for **{country}**.")
        return

    if include_cities_norm and len(include_cities_norm) > 0:
        city_filter = include_cities_norm
        scope_label = f"within {int(radius_km)} km" if radius_km else "nearby"
        scope_note = f"({len(city_filter)} locations)"
    else:
        city_filter = {_normcase(city)}
        scope_label = "city only"
        scope_note = ""

    df_city_scope = df_country[df_country["__City_norm__"].isin(city_filter)].copy()
    if df_city_scope.empty:
        st.info(f"No skills found for **{city}**, {country}.")
        return

    # Modern header
    st.markdown(
        f"""<div class="skills-header">
            Skills for {city}, {country}
            <span style="font-weight: normal; font-size: 0.85rem;">({scope_label} {scope_note})</span>
        </div>""",
        unsafe_allow_html=True,
    )

    def _render_skills_table(df_in: pd.DataFrame, title: str) -> None:
        top50 = (
            df_in.groupby("Skill", as_index=False)["count"]
            .sum()
            .sort_values("count", ascending=False)
            .head(50)
        )
        top50.columns = ["Skill", "Count"]
        st.markdown(f'<div class="skills-subheader">{title}</div>', unsafe_allow_html=True)
        st.dataframe(
            top50,
            use_container_width=True,
            hide_index=True,
            height=400,
        )
        # Allow users to download the visible table
        csv_buf = io.StringIO()
        top50.to_csv(csv_buf, index=False)
        safe_title = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_") or "skills"
        safe_country = re.sub(r"[^A-Za-z0-9]+", "_", country).strip("_") or "country"
        safe_city = re.sub(r"[^A-Za-z0-9]+", "_", city).strip("_") or "city"
        filename = f"skills_{safe_country}_{safe_city}_{safe_title}.csv"
        st.download_button(
            "Download CSV",
            data=csv_buf.getvalue(),
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
        )

    if "SkillType" not in df_city_scope.columns or df_city_scope["SkillType"].eq("").all():
        _render_skills_table(df_city_scope, "Top 50 Skills")
        return

    c1_, c2_ = st.columns(2)
    with c1_:
        hard_skills = df_city_scope[df_city_scope["SkillType"] == "ST1"]
        _render_skills_table(hard_skills, "Hard Skills (Top 50)")
    with c2_:
        soft_skills = df_city_scope[df_city_scope["SkillType"] == "ST2"]
        _render_skills_table(soft_skills, "Soft Skills (Top 50)")


# ==========================
# Click Handling
# ==========================
def _extract_clicked_center(event_obj: Any) -> Tuple[Optional[str], Optional[str]]:
    if event_obj is None:
        return None, None
    sel = getattr(event_obj, "selection", None)
    if isinstance(sel, dict):
        objs = sel.get("objects") or {}
        picked = objs.get("city-points") or objs.get("coverage-circles") or []
        if picked:
            obj = picked[0]
            city = (
                str(obj.get("Center_City", "")).strip()
                if isinstance(obj, dict)
                else None
            )
            country = (
                str(obj.get("Sheet", "")).strip() if isinstance(obj, dict) else None
            )
            return (city or None), (country or None)
    return None, None


def _set_dropdown_selection(country: str, city: str) -> None:
    st.session_state["__pending_country"] = country
    st.session_state["__pending_city"] = city


def _apply_click_selection(event: Any) -> bool:
    clicked_city, clicked_country = _extract_clicked_center(event)
    if clicked_city and clicked_country:
        _set_dropdown_selection(clicked_country, clicked_city)
        st.rerun()
        return True
    return False


def _ingest_pending_selection() -> None:
    pc = st.session_state.pop("__pending_country", None)
    pcity = st.session_state.pop("__pending_city", None)
    if pc:
        st.session_state["v2_country"] = pc
        st.session_state["__pending_city_after_country__"] = pcity


# ==========================
# Map Style
# ==========================
MAP_STYLE = "https://basemaps.cartocdn.com/gl/voyager-nolabels-gl-style/style.json"


# ==========================
# VERSION 1: Global Map
# ==========================
def render_version_1():
    """
    Global Map: Shows ALL locations from aggregated data as red circles
    scaled by Total_posts. Only country dropdown. No city selection.
    """
    available_countries = sorted(FILTERED_AGGREGATED.keys())
    country_options = ["All"] + available_countries

    st.session_state.setdefault("v1_country", "All")

    col1, col2 = st.columns([1, 3])
    with col1:
        sel_country = st.selectbox(
            "Country",
            options=country_options,
            key="v1_country",
        )

    # Get aggregated data for selected scope
    if sel_country == "All":
        # Combine all countries
        all_data = []
        for country, df in FILTERED_AGGREGATED.items():
            if df is not None and not df.empty:
                temp = df.copy()
                temp["Sheet"] = country
                all_data.append(temp)
        if all_data:
            map_df = pd.concat(all_data, ignore_index=True)
        else:
            st.info("No data available.")
            return
        rmin = DEFAULT_GLOBAL_MIN_RADIUS_ALL
        rmax = DEFAULT_GLOBAL_MAX_RADIUS_ALL
        total_for_scaling = GLOBAL_TOTAL_AGGREGATED
        zoom = 3
    else:
        df = FILTERED_AGGREGATED.get(sel_country)
        if df is None or df.empty:
            st.info(f"No data available for {sel_country}.")
            return
        map_df = df.copy()
        map_df["Sheet"] = sel_country
        rmin = DEFAULT_COUNTRY_MIN_RADIUS
        rmax = DEFAULT_COUNTRY_MAX_RADIUS
        total_for_scaling = COUNTRY_TOTAL_AGGREGATED.get(sel_country, 1)
        zoom = 5

    # Scale red circles by Count (Total_posts equivalent)
    map_df["radius"] = _scale_radius_by_global_sum(
        map_df["Count"], float(total_for_scaling), rmin, rmax
    )

    # Red circle colors
    map_df["red_r"] = 220
    map_df["red_g"] = 50
    map_df["red_b"] = 50
    map_df["red_a"] = 180

    # Mark which cities should be hoverable (version 1 only)
    map_df["__country_norm__"] = map_df["Sheet"].apply(_normcase)
    map_df["__city_norm__"] = map_df["City"].apply(_normcase)
    map_df["__is_hoverable__"] = map_df.apply(
        lambda row: (row["__country_norm__"], row["__city_norm__"]) in ALWAYS_LABEL_CITIES,
        axis=1,
    )
    hoverable_df = map_df[map_df["__is_hoverable__"]].copy()
    non_hoverable_df = map_df[~map_df["__is_hoverable__"]].copy()

    vlat = float(map_df["lat"].mean())
    vlon = float(map_df["lon"].mean())

    # Red circles layers (hover enabled only for ALWAYS_LABEL_CITIES)
    layers: List[pdk.Layer] = []
    if not non_hoverable_df.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=non_hoverable_df,
                id="red-circles-static",
                get_position="[lon, lat]",
                get_radius="radius",
                get_fill_color="[red_r, red_g, red_b, red_a]",
                pickable=False,
                opacity=0.7,
                stroked=False,
            )
        )
    if not hoverable_df.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=hoverable_df,
                id="red-circles-hoverable",
                get_position="[lon, lat]",
                get_radius="radius",
                get_fill_color="[red_r, red_g, red_b, red_a]",
                pickable=True,
                opacity=0.7,
                stroked=False,
            )
        )

    # Labels for ALWAYS_LABEL_CITIES
    label_layer = _label_layer_v1(map_df)
    if label_layer:
        layers.append(label_layer)

    tooltip = {
        "html": "<b>{City}</b><br/>Posts: {Count}",
        "style": {"backgroundColor": "#c0392b", "color": "white", "borderRadius": "8px"},
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlon, zoom=zoom),
            tooltip=tooltip,
            height=550,
            map_style=MAP_STYLE,
        ),
        use_container_width=True,
        key="v1_map",
    )

    st.markdown("---")
    st.caption(
        "**Red circles** scaled by job post count (aggregated). "
        "Hover over circles to see city details."
    )


# ==========================
# VERSION 2: Coverage of Main Locations
# ==========================
def render_version_2():
    """
    Coverage Map: Shows only primary/secondary cities as green circles
    representing coverage area (Radius_km_max). Cities are clickable.
    """
    _ingest_pending_selection()

    available_countries = sorted(MAJOR_CENTERS.keys())
    country_options = ["All"] + available_countries

    st.session_state.setdefault("v2_country", "All")
    st.session_state.setdefault("v2_city", "All")

    col1, col2 = st.columns([1, 1])

    with col1:
        current_country = st.session_state.get("v2_country", "All")
        if current_country not in country_options:
            current_country = "All"
            st.session_state["v2_country"] = "All"
        sel_country = st.selectbox(
            "Country",
            options=country_options,
            key="v2_country",
        )

    # Build city choices
    if sel_country != "All":
        city_choices = ["All"] + CENTER_CITIES_BY_COUNTRY.get(sel_country, [])
    else:
        # All cities from all countries
        all_cities = []
        for cities in CENTER_CITIES_BY_COUNTRY.values():
            all_cities.extend(cities)
        city_choices = ["All"] + sorted(set(all_cities))

    _pending_city = st.session_state.pop("__pending_city_after_country__", None)
    if _pending_city:
        st.session_state["v2_city"] = _pending_city if _pending_city in city_choices else "All"

    with col2:
        current_city = st.session_state.get("v2_city", "All")
        if current_city not in city_choices:
            current_city = "All"
            st.session_state["v2_city"] = "All"
        sel_city = st.selectbox(
            "City",
            options=city_choices,
            key="v2_city",
        )

    # Build centers dataframe
    if sel_country == "All":
        pieces = []
        for ctry, df in MAJOR_CENTERS.items():
            if df is not None and not df.empty:
                pieces.append(df.copy())
        if pieces:
            centers_df = pd.concat(pieces, ignore_index=True)
        else:
            st.info("No coverage data available.")
            return
        zoom = 3
    else:
        df = MAJOR_CENTERS.get(sel_country)
        if df is None or df.empty:
            st.info(f"No coverage data for {sel_country}.")
            return
        centers_df = df.copy()
        zoom = 5

    # Filter by city if selected
    selected_city_norm = None
    if sel_city != "All":
        selected_city_norm = _normcase(sel_city)
        # Find the actual country for this city
        for ctry, df in MAJOR_CENTERS.items():
            if df is not None and not df.empty:
                match = df[df["__City_norm__"] == selected_city_norm]
                if not match.empty:
                    sel_country = ctry
                    break

    # Green coverage circles (Radius_km_max in meters)
    centers_df["coverage_radius_m"] = centers_df["Radius_km_max"] * 1000

    # Color coding - highlight selected city
    if selected_city_norm:
        centers_df["green_r"] = np.where(
            centers_df["__City_norm__"] == selected_city_norm, 0, 100
        )
        centers_df["green_g"] = np.where(
            centers_df["__City_norm__"] == selected_city_norm, 180, 200
        )
        centers_df["green_b"] = np.where(
            centers_df["__City_norm__"] == selected_city_norm, 0, 100
        )
        centers_df["green_a"] = np.where(
            centers_df["__City_norm__"] == selected_city_norm, 180, 80
        )
    else:
        centers_df["green_r"] = 34
        centers_df["green_g"] = 139
        centers_df["green_b"] = 34
        centers_df["green_a"] = 120

    # Center map on selected city or average
    if selected_city_norm:
        sel_row = centers_df[centers_df["__City_norm__"] == selected_city_norm]
        if not sel_row.empty:
            vlat = float(sel_row["lat"].iloc[0])
            vlon = float(sel_row["lon"].iloc[0])
            R_km = float(sel_row["Radius_km_max"].iloc[0])
            # Adjust zoom based on radius
            if R_km <= 10:
                zoom = 10
            elif R_km <= 25:
                zoom = 9
            elif R_km <= 50:
                zoom = 8
            elif R_km <= 100:
                zoom = 7
            else:
                zoom = 6
        else:
            vlat = float(centers_df["lat"].mean())
            vlon = float(centers_df["lon"].mean())
    else:
        vlat = float(centers_df["lat"].mean())
        vlon = float(centers_df["lon"].mean())

    # Green coverage circles layer
    green_layer = pdk.Layer(
        "ScatterplotLayer",
        data=centers_df,
        id="coverage-circles",
        get_position="[lon, lat]",
        get_radius="coverage_radius_m",
        get_fill_color="[green_r, green_g, green_b, green_a]",
        pickable=True,
        opacity=0.6,
        stroked=True,
        get_line_color=[0, 100, 0, 200],
        line_width_min_pixels=1,
    )

    # Labels
    label_df = centers_df.copy()
    # Normalize with fallbacks in case cached data lacks precomputed columns
    if "__Country_norm__" in label_df.columns:
        label_df["__country_norm__"] = label_df["__Country_norm__"]
    elif "country" in label_df.columns:
        label_df["__country_norm__"] = label_df["country"].apply(_normcase)
    elif "Sheet" in label_df.columns:
        label_df["__country_norm__"] = label_df["Sheet"].apply(_normcase)
    else:
        label_df["__country_norm__"] = ""

    if "__City_norm__" in label_df.columns:
        label_df["__city_norm__"] = label_df["__City_norm__"]
    elif "Center_City" in label_df.columns:
        label_df["__city_norm__"] = label_df["Center_City"].apply(_normcase)
    elif "City" in label_df.columns:
        label_df["__city_norm__"] = label_df["City"].apply(_normcase)
    else:
        label_df["__city_norm__"] = ""
    if sel_country == "All":
        label_df = label_df[
            label_df.apply(
                lambda row: (row["__country_norm__"], row["__city_norm__"]) in ALWAYS_LABEL_CITIES,
                axis=1,
            )
        ]
    label_layer = _label_layer_v2(label_df)
    layers = [green_layer]
    if label_layer:
        layers.append(label_layer)

    tooltip = {
        "html": "<b>{Center_City}</b><br/>Coverage: {Radius_km_max} km<br/>Total Posts: {Total_posts}",
        "style": {"backgroundColor": "#27ae60", "color": "white", "borderRadius": "8px"},
    }

    event = st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlon, zoom=zoom),
            tooltip=tooltip,
            height=550,
            map_style=MAP_STYLE,
        ),
        use_container_width=True,
        key="v2_map",
        on_select="rerun",
        selection_mode="single-object",
    )

    _apply_click_selection(event)

    st.markdown("---")

    # Show skills panel if a city is selected
    if sel_city != "All" and sel_country != "All":
        R_km, lat0, lon0 = resolve_center_radius_and_centroid(sel_country, sel_city)
        if R_km and lat0 and lon0:
            df_country_geo = build_city_geo_for_country(
                SKILLS_XLSX_PATH, SKILLS_MTIME, sel_country
            )
            include_norms: Set[str] = set()
            if not df_country_geo.empty:
                d = _haversine_km_vec(
                    df_country_geo["lat"].to_numpy(),
                    df_country_geo["lon"].to_numpy(),
                    lat0,
                    lon0,
                )
                df_near = df_country_geo.assign(__dist_km__=d)
                df_near = df_near[df_near["__dist_km__"] <= float(R_km)].copy()
                include_norms = {
                    _normcase(c)
                    for c in df_near["City"].dropna().astype(str).tolist()
                    if _normcase(c) not in {"", "nan"}
                }
            if include_norms:
                render_skills_panel(
                    sel_country,
                    sel_city,
                    include_cities_norm=include_norms,
                    radius_km=float(R_km),
                )
            else:
                render_skills_panel(sel_country, sel_city)
        else:
            render_skills_panel(sel_country, sel_city)
    else:
        st.caption(
            "**Green circles** show coverage areas (Radius_km_max). "
            "Click a city or select from dropdown to view skills."
        )


# ==========================
# Main App
# ==========================
st.markdown("### Job Posts & Skills Map")

# Radio button for version selection
version = st.radio(
    "Select View",
    options=["Global Map", "Coverage of Main Locations"],
    key="map_version",
    horizontal=True,
)

st.markdown("---")

if version == "Global Map":
    render_version_1()
else:
    render_version_2()
