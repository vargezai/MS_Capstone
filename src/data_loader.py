"""
U.S. Energy Transition Data Pipeline
All sources are public government data (no Kaggle):
  EIA, EPA eGRID, NOAA, BEA GDP, DSIRE RPS

Ported from Final_data_loading.ipynb — logic unchanged, paths updated
from Google Drive to local data/raw/ and data/processed/.
"""

import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILES = {
    "generation" : DATA_DIR / "generation_monthly.xlsx",
    "consumption": DATA_DIR / "consumption_monthly.xlsx",
    "temperature": DATA_DIR / "average_monthly_temperature_by_state_1950-2022.csv",
    "gdp"        : DATA_DIR / "GDP_Table.xlsx",
    "rps"        : DATA_DIR / "RPS_list.xlsx",
}

EGRID_FILES = {
    2004: DATA_DIR / "eGRID2004_aggregation.xls",
    2005: DATA_DIR / "eGRID2005_aggregation.xls",
    2007: DATA_DIR / "eGRID2007_aggregation.xls",
    2009: DATA_DIR / "eGRID2009_data.xls",
    2010: DATA_DIR / "eGRID2010_Data.xls",
    2012: DATA_DIR / "eGRID2012_Data.xlsx",
    2014: DATA_DIR / "eGRID2014_Data_v2.xlsx",
    2016: DATA_DIR / "egrid2016_data.xlsx",
    2018: DATA_DIR / "egrid2018_data_v2.xlsx",
    2019: DATA_DIR / "egrid2019_data.xlsx",
    2020: DATA_DIR / "eGRID2020_Data_v2.xlsx",
    2021: DATA_DIR / "eGRID2021_data.xlsx",
    2022: DATA_DIR / "egrid2022_data.xlsx",
    2023: DATA_DIR / "egrid2023_data_rev2.xlsx",
}

EGRID_SPECS = {
    2004: {"sheet": "EGRDST04", "header_row": 4, "divisor": 2000.0},
    2005: {"sheet": "ST05",     "header_row": 4, "divisor": 2000.0},
    2007: {"sheet": "ST07",     "header_row": 4, "divisor": 2000.0},
    2009: {"sheet": "ST09",     "header_row": 4, "divisor": 2000.0},
    2010: {"sheet": "ST10",     "header_row": 4, "divisor": 2000.0},
    2012: {"sheet": "ST12",     "header_row": 4, "divisor": 2000.0},
    2014: {"sheet": "ST14",     "header_row": 1, "divisor": 2000.0},
    2016: {"sheet": "ST16",     "header_row": 1, "divisor": 2000.0},
    2018: {"sheet": "ST18",     "header_row": 1, "divisor": 2000.0},
    2019: {"sheet": "ST19",     "header_row": 1, "divisor": 2000.0},
    2020: {"sheet": "ST20",     "header_row": 1, "divisor": 2000.0},
    2021: {"sheet": "ST21",     "header_row": 1, "divisor": 2000.0},
    2022: {"sheet": "ST22",     "header_row": 1, "divisor": 2000.0},
    2023: {"sheet": "ST23",     "header_row": 1, "divisor": 2000.0},
}


# ============================================================================
# FUNCTION 1: VERIFY INPUT FILES
# ============================================================================

def verify_input_files():
    print("\n🔍 VERIFYING INPUT FILES:")
    print("=" * 80)
    all_exist = True
    for name, path in INPUT_FILES.items():
        status = "✅" if path.exists() else "❌"
        print(f"{status} {name:15s}: {path}")
        if not path.exists():
            all_exist = False

    print("\n📋 eGRID FILES:")
    found = 0
    for year in sorted(EGRID_FILES):
        path = EGRID_FILES[year]
        status = "✅" if path.exists() else "❌"
        print(f"   {status} {year}: {path.name}")
        if path.exists():
            found += 1
    print(f"\n   Found {found}/{len(EGRID_FILES)} eGRID files")
    print("=" * 80)
    return all_exist


# ============================================================================
# FUNCTION 2: LOAD GENERATION DATA
# ============================================================================

def load_multi_sheet_generation(file_path):
    print("\n📊 LOADING GENERATION DATA...")
    xl_file     = pd.ExcelFile(file_path)
    data_sheets = [s for s in xl_file.sheet_names if s not in ["README", "Notes"]]
    print(f"   Found {len(data_sheets)} data sheets")
    all_data = []

    for sheet_name in data_sheets:
        for skiprows in [0, 4, 5, 3, 6, 7, 8]:
            try:
                df_sheet = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skiprows)
                df_sheet.columns = (df_sheet.columns
                                    .str.replace("\n", " ", regex=False)
                                    .str.replace("  ", " ", regex=True)
                                    .str.strip())
                required = ["STATE", "YEAR", "MONTH", "ENERGY SOURCE"]
                if all(c in df_sheet.columns for c in required):
                    producer_cols = [c for c in df_sheet.columns if "PRODUCER" in c.upper()]
                    if producer_cols:
                        before = len(df_sheet)
                        df_sheet = df_sheet[
                            df_sheet[producer_cols[0]] == "Total Electric Power Industry"
                        ].copy()
                        after = len(df_sheet)
                        y0 = df_sheet["YEAR"].min() if after else "N/A"
                        y1 = df_sheet["YEAR"].max() if after else "N/A"
                        print(f"   ✅ Loaded: {sheet_name:30s} ({after:>7,} rows, skip={skiprows}, "
                              f"years {y0}-{y1}, filtered from {before:,})")
                    else:
                        y0, y1 = df_sheet["YEAR"].min(), df_sheet["YEAR"].max()
                        print(f"   ✅ Loaded: {sheet_name:30s} ({len(df_sheet):>7,} rows, "
                              f"skip={skiprows}, years {y0}-{y1})")
                    all_data.append(df_sheet)
                    break
            except Exception:
                continue

    df = pd.concat(all_data, ignore_index=True)
    print(f"\n   ✅ Combined {len(data_sheets)} sheets → {len(df):,} records")
    print(f"   📅 {df['YEAR'].min()}–{df['YEAR'].max()} | "
          f"🗺️ {df['STATE'].nunique()} states | "
          f"⚡ {df['ENERGY SOURCE'].nunique()} sources")
    return df


# ============================================================================
# FUNCTION 3: TRANSFORM GENERATION TO WIDE FORMAT
# ============================================================================

def transform_generation_to_wide(df):
    print("\n🔄 TRANSFORMING GENERATION DATA TO WIDE FORMAT...")
    gen_cols = [c for c in df.columns if "GENERATION" in c.upper() and "MEGAWATT" in c.upper()]
    if not gen_cols:
        raise ValueError("Could not find generation column")
    gen_col = gen_cols[0]
    print(f"   ✅ Using generation column: '{gen_col}'")
    df[gen_col] = pd.to_numeric(df[gen_col], errors="coerce")

    df_wide = df.pivot_table(
        index=["STATE", "YEAR", "MONTH"],
        columns="ENERGY SOURCE",
        values=gen_col,
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    df_wide.columns = [c.strip() if isinstance(c, str) else c for c in df_wide.columns]
    print(f"   ✅ Pivoted: {df_wide.shape} | {df_wide['YEAR'].min()}–{df_wide['YEAR'].max()}")

    energy_sources = [c for c in df_wide.columns if c not in ["STATE", "YEAR", "MONTH"]]
    gen_sources    = [c for c in energy_sources if c not in ["Total", "Pumped Storage"]]
    print(f"\n   🔧 Recalculating total from: {gen_sources}")
    df_wide["Total_Generation_MWh"] = df_wide[gen_sources].sum(axis=1)

    if "Total" in df_wide.columns:
        sample = df_wide[df_wide["YEAR"] >= 2015][
            ["STATE", "YEAR", "MONTH", "Total", "Total_Generation_MWh"]].head(5)
        print(f"      Sample comparison:\n{sample.to_string(index=False)}")

    renewable_sources = [
        "Wind", "Solar Thermal and Photovoltaic", "Geothermal",
        "Hydroelectric Conventional", "Wood and Wood Derived Fuels", "Other Biomass",
    ]
    fossil_sources = ["Coal", "Natural Gas", "Petroleum"]

    df_wide["Renewable_Generation_MWh"] = sum(
        df_wide[s].fillna(0) for s in renewable_sources if s in df_wide.columns)
    df_wide["Fossil_Generation_MWh"] = sum(
        df_wide[s].fillna(0) for s in fossil_sources if s in df_wide.columns)
    df_wide["Nuclear_Generation_MWh"] = (
        df_wide["Nuclear"].fillna(0) if "Nuclear" in df_wide.columns else 0)

    for name, num_col in [
        ("Renewable_Share_Pct", "Renewable_Generation_MWh"),
        ("Fossil_Share_Pct",    "Fossil_Generation_MWh"),
        ("Nuclear_Share_Pct",   "Nuclear_Generation_MWh"),
    ]:
        df_wide[name] = np.where(
            df_wide["Total_Generation_MWh"] > 0,
            df_wide[num_col] / df_wide["Total_Generation_MWh"] * 100, 0)

    print(f"\n   📊 Mean renewable {df_wide['Renewable_Share_Pct'].mean():.2f}% | "
          f"fossil {df_wide['Fossil_Share_Pct'].mean():.2f}% | "
          f"nuclear {df_wide['Nuclear_Share_Pct'].mean():.2f}%")
    return df_wide


# ============================================================================
# FUNCTION 4: LOAD CONSUMPTION DATA
# ============================================================================

def load_consumption_data(file_path):
    print("\n📊 LOADING CONSUMPTION DATA...")
    xl_file     = pd.ExcelFile(file_path)
    data_sheets = [s for s in xl_file.sheet_names if s not in ["README", "Notes"]]
    print(f"   Found {len(data_sheets)} data sheets")
    all_data = []

    for sheet_name in data_sheets:
        for skiprows in [0, 4, 5, 3, 6]:
            try:
                df_sheet = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skiprows)
                cons_cols = [c for c in df_sheet.columns if "CONSUMPTION" in c.upper()]
                if cons_cols and "STATE" in df_sheet.columns:
                    df_sheet = df_sheet[["STATE", "YEAR", "MONTH", cons_cols[0]]].copy()
                    df_sheet.columns = ["STATE", "YEAR", "MONTH", "Total_Consumption_MWh"]
                    df_sheet["Total_Consumption_MWh"] = pd.to_numeric(
                        df_sheet["Total_Consumption_MWh"], errors="coerce")
                    df_sheet = df_sheet.groupby(
                        ["STATE", "YEAR", "MONTH"], as_index=False
                    )["Total_Consumption_MWh"].sum()
                    print(f"      Loaded: {sheet_name:30s} ({len(df_sheet):>6,} unique state-months)")
                    all_data.append(df_sheet)
                    break
            except Exception:
                continue

    df = pd.concat(all_data, ignore_index=True)
    df = df.groupby(["STATE", "YEAR", "MONTH"], as_index=False)["Total_Consumption_MWh"].sum()
    print(f"\n   ✅ {len(df):,} unique state-months | "
          f"{df['YEAR'].min()}–{df['YEAR'].max()} | {df['STATE'].nunique()} states")
    return df


# ============================================================================
# FUNCTION 5: LOAD ALL eGRID FILES
# ============================================================================

def load_all_egrid_files():
    """
    Load CO2 emission rates from yearly eGRID Excel files (EPA public data).
    Handles all formats 2004-2023 and unit conversions automatically.
    Returns monthly state-level DataFrame.
    """
    print("\n📊 LOADING eGRID DATA FROM YEARLY EXCEL FILES...")
    print("=" * 80)

    all_annual = []

    for year in sorted(EGRID_FILES.keys()):
        fpath = EGRID_FILES[year]
        if not fpath.exists():
            print(f"   ⚠️  {year}: file not found — {fpath.name}")
            continue

        spec       = EGRID_SPECS[year]
        sheet      = spec["sheet"]
        header_row = spec["header_row"]
        divisor    = spec["divisor"]

        try:
            engine = "xlrd" if str(fpath).endswith(".xls") else "openpyxl"
            df = pd.read_excel(fpath, sheet_name=sheet, header=header_row, engine=engine)
        except Exception as e:
            print(f"   ❌ {year}: read error — {e}")
            continue

        if "PSTATABB" not in df.columns or "STCO2RTA" not in df.columns:
            print(f"   ❌ {year}: required columns (PSTATABB, STCO2RTA) not found. "
                  f"Available: {df.columns.tolist()[:6]}...")
            continue

        data = df[["PSTATABB", "STCO2RTA"]].copy()
        data.columns = ["STATE", "CO2_raw"]
        data = data[data["STATE"].astype(str).str.match(r"^[A-Z]{2}$")]
        data["CO2_raw"] = pd.to_numeric(data["CO2_raw"], errors="coerce")
        data = data.dropna(subset=["CO2_raw"])
        data["CO2_Intensity_Tons_per_MWh"] = data["CO2_raw"] / divisor
        data["YEAR"] = year

        n    = len(data)
        mean = data["CO2_Intensity_Tons_per_MWh"].mean()
        unit = "kg/MWh" if divisor == 907.185 else "lb/MWh"
        print(f"   ✅ {year} ({fpath.name}): {n} states | "
              f"unit={unit} ÷{divisor:.0f} | mean={mean:.4f} tons/MWh")

        all_annual.append(data[["STATE", "YEAR", "CO2_Intensity_Tons_per_MWh"]])

    if not all_annual:
        print("   ❌ No eGRID files loaded!")
        return pd.DataFrame(columns=["STATE", "YEAR", "MONTH", "CO2_Intensity_Tons_per_MWh"])

    annual = pd.concat(all_annual, ignore_index=True)

    rows = []
    for _, row in annual.iterrows():
        for month in range(1, 13):
            rows.append({
                "STATE": row["STATE"],
                "YEAR" : int(row["YEAR"]),
                "MONTH": month,
                "CO2_Intensity_Tons_per_MWh": row["CO2_Intensity_Tons_per_MWh"],
            })
    monthly = pd.DataFrame(rows).sort_values(["YEAR", "STATE", "MONTH"]).reset_index(drop=True)

    print(f"\n{'=' * 80}")
    print(f"✅ COMBINED eGRID: {len(monthly):,} state-month records")
    print(f"   Years: {sorted(monthly['YEAR'].unique())}")
    print("\n   📈 Summary by year:")
    for yr in sorted(monthly["YEAR"].unique()):
        d = monthly[monthly["YEAR"] == yr]
        print(f"      {yr}: {d['STATE'].nunique():>2} states | "
              f"mean={d['CO2_Intensity_Tons_per_MWh'].mean():.4f} | "
              f"min={d['CO2_Intensity_Tons_per_MWh'].min():.4f} | "
              f"max={d['CO2_Intensity_Tons_per_MWh'].max():.4f}")
    print("=" * 80)
    return monthly


# ============================================================================
# FUNCTION 6: LOAD TEMPERATURE DATA
# ============================================================================

def load_temperature_data(file_path):
    print("\n📊 LOADING TEMPERATURE DATA...")
    df_temp = pd.read_csv(file_path, encoding="latin-1")
    print(f"   ✅ Loaded {len(df_temp):,} records | "
          f"{df_temp['year'].min()}–{df_temp['year'].max()}")

    df_temp.columns = df_temp.columns.str.strip()
    df_temp = df_temp.rename(columns={
        "state": "STATE", "year": "YEAR", "month": "MONTH", "average_temp": "Avg_Temp_F"})

    sample = str(df_temp["MONTH"].iloc[0]).strip()
    if not sample.isdigit():
        print("   🔄 Converting month names to numbers...")
        month_map = {
            "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
            "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
            "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
            "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
        }
        df_temp["MONTH"] = df_temp["MONTH"].apply(
            lambda x: month_map.get(re.sub(r"[^a-z]", "", str(x).lower().strip()), np.nan)
        )
        print(f"      Converted: {df_temp['MONTH'].notna().sum():,}/{len(df_temp):,}")
    else:
        df_temp["MONTH"] = pd.to_numeric(df_temp["MONTH"], errors="coerce")

    if df_temp["Avg_Temp_F"].dtype == object:
        print("   🔄 Cleaning temperature values...")
        df_temp["Avg_Temp_F"] = df_temp["Avg_Temp_F"].apply(
            lambda x: float(re.sub(r"[¡°]F?", "", str(x)).strip())
            if pd.notna(x) else np.nan
        )
    else:
        df_temp["Avg_Temp_F"] = pd.to_numeric(df_temp["Avg_Temp_F"], errors="coerce")

    state_abbrev_map = {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
        "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
        "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
        "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
        "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
        "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
        "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
        "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
        "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
        "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
        "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
        "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
        "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
    }
    print("   🔄 Converting state names to abbreviations...")
    df_temp["STATE"] = df_temp["STATE"].str.strip().apply(
        lambda x: re.sub(r"[^a-zA-Z\s]", "", str(x)).strip() if pd.notna(x) else x)
    df_temp["STATE"] = df_temp["STATE"].map(state_abbrev_map)
    df_temp = df_temp.dropna(subset=["STATE"])

    df_temp["YEAR"]  = pd.to_numeric(df_temp["YEAR"],  errors="coerce").astype("Int64")
    df_temp["MONTH"] = pd.to_numeric(df_temp["MONTH"], errors="coerce").astype("Int64")
    df_temp = df_temp.dropna(subset=["YEAR", "MONTH", "Avg_Temp_F"])
    df_temp["YEAR"]  = df_temp["YEAR"].astype(int)
    df_temp["MONTH"] = df_temp["MONTH"].astype(int)

    # Average across climate divisions — more accurate than keeping first row
    before = len(df_temp)
    df_temp = df_temp.groupby(["STATE", "YEAR", "MONTH"], as_index=False)["Avg_Temp_F"].mean()
    after   = len(df_temp)
    if before > after:
        print(f"   🔧 Averaged {before - after:,} duplicate climate-division rows "
              f"({before:,} → {after:,})")

    print(f"\n   ✅ {len(df_temp):,} records | "
          f"{df_temp['YEAR'].min()}–{df_temp['YEAR'].max()} | "
          f"{df_temp['STATE'].nunique()} states | "
          f"{df_temp['Avg_Temp_F'].min():.1f}°F–{df_temp['Avg_Temp_F'].max():.1f}°F")
    return df_temp


# ============================================================================
# FUNCTION 7: LOAD GDP DATA
# ============================================================================

def process_gdp_data_annual(file_path):
    print("\n📊 LOADING GDP DATA...")

    state_abbrev_map = {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
        "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
        "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
        "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
        "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
        "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
        "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
        "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
        "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
        "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
        "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
        "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
        "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
        "Puerto Rico": "PR", "Guam": "GU",
    }

    # Support both CSV (original SAGDP long format) and xlsx (SQGDP1 quarterly table)
    if str(file_path).endswith(".xlsx") or str(file_path).endswith(".xls"):
        # GDP_Table.xlsx: SQGDP1 quarterly percent-change table
        # Row 5 (0-indexed) is the actual header; skip metadata rows 0-4
        raw = pd.read_excel(file_path, sheet_name=0, header=None)
        header_row = 5
        df_gdp = pd.read_excel(file_path, sheet_name=0, header=header_row)
        # Columns: GeoFIPS, GeoName, then quarterly labels like "2005:Q2"
        quarter_cols = [c for c in df_gdp.columns if isinstance(c, str) and ":" in c]
        print(f"   ✅ Loaded quarterly GDP table: {df_gdp.shape[0]} rows, {len(quarter_cols)} quarters")
        print(f"   Quarters: {quarter_cols[0]} → {quarter_cols[-1]}")

        # Reconstruct GDP index levels from annualized quarterly percent-change
        # Starting index = 100 for each state; apply (1 + rate/100)^(1/4) each quarter
        gdp_long = []
        for _, row in df_gdp.iterrows():
            state = state_abbrev_map.get(str(row.get("GeoName", "")).strip())
            if state is None:
                continue
            level = 100.0
            quarterly = []
            for qc in quarter_cols:
                v = row[qc]
                try:
                    pct = float(v)
                    level = level * ((1 + pct / 100) ** 0.25)
                except (ValueError, TypeError):
                    pass  # keep prior level for missing quarter
                year_str, q = qc.split(":")
                quarter_num = int(q[1])
                quarterly.append({"STATE": state, "YEAR": int(year_str),
                                   "QUARTER": quarter_num, "GDP_Index": level})

            # Aggregate to annual by averaging quarterly index within each year
            qdf = pd.DataFrame(quarterly)
            annual = qdf.groupby(["STATE", "YEAR"])["GDP_Index"].mean().reset_index()
            annual.rename(columns={"GDP_Index": "Real_GDP_Millions"}, inplace=True)
            gdp_long.append(annual)

        df_long = pd.concat(gdp_long, ignore_index=True) if gdp_long else pd.DataFrame()
        print(f"   ✅ Reconstructed {len(df_long):,} state-year index levels from quarterly growth rates")

    else:
        # Original SAGDP1 CSV long format (GeoName + Description + year columns)
        df_gdp = pd.read_csv(file_path, encoding="latin-1")
        print(f"   ✅ Loaded {df_gdp.shape}")

        gdp_filters = [
            "Real GDP (millions of chained 2017 dollars)",
            "Real GDP (millions of chained 2012 dollars)",
            "Real GDP", "All industry total",
        ]
        df_filtered = None
        for f in gdp_filters:
            df_filtered = df_gdp[df_gdp["Description"].str.contains(f, case=False, na=False)]
            if len(df_filtered):
                print(f"   ✅ Found {len(df_filtered)} rows using filter: '{f}'")
                break
        if df_filtered is None or len(df_filtered) == 0:
            print("   ❌ GDP data not found")
            return pd.DataFrame(
                columns=["STATE", "YEAR", "MONTH", "Real_GDP_Millions", "GDP_Growth_Rate_Annual"])

        df_gdp    = df_filtered.copy()
        year_cols = [c for c in df_gdp.columns if str(c).isdigit()]
        print(f"   Found {len(year_cols)} year columns: {year_cols[0]}–{year_cols[-1]}")

        gdp_long = []
        for _, row in df_gdp.iterrows():
            state = state_abbrev_map.get(row["GeoName"])
            if state is None:
                continue
            for yc in year_cols:
                v = row[yc]
                if pd.notna(v) and v not in ["(D)", "(L)"]:
                    try:
                        gdp_long.append({
                            "STATE": state,
                            "YEAR":  int(yc),
                            "Real_GDP_Millions": float(str(v).replace(",", "")),
                        })
                    except (ValueError, TypeError):
                        continue

        df_long = pd.DataFrame(gdp_long)
        print(f"   ✅ Processed {len(df_long):,} state-year records")

    # Remove duplicates before expanding to monthly
    dup_check = df_long.groupby(["STATE", "YEAR"]).size()
    dups = dup_check[dup_check > 1]
    if len(dups):
        print(f"   ⚠️  Found {len(dups):,} duplicate STATE-YEAR — removing...")
        df_long = df_long.drop_duplicates(subset=["STATE", "YEAR"], keep="first")
        print(f"   ✅ After dedup: {len(df_long):,} state-year records")

    df_long = df_long.sort_values(["STATE", "YEAR"])
    df_long["GDP_Growth_Rate_Annual"] = (
        df_long.groupby("STATE")["Real_GDP_Millions"].pct_change() * 100)

    monthly = []
    for _, row in df_long.iterrows():
        for m in range(1, 13):
            monthly.append({
                "STATE": row["STATE"], "YEAR": row["YEAR"], "MONTH": m,
                "Real_GDP_Millions": row["Real_GDP_Millions"],
                "GDP_Growth_Rate_Annual": row["GDP_Growth_Rate_Annual"],
            })
    df_monthly = pd.DataFrame(monthly)
    print(f"   ✅ Expanded to {len(df_monthly):,} state-month records")
    return df_monthly


# ============================================================================
# FUNCTION 8: CREATE RPS PANEL
# ============================================================================

def create_rps_panel(file_path, years_range):
    print("\n📊 CREATING RPS PANEL DATA...")
    df_rps = pd.read_excel(file_path)
    print(f"   ✅ Loaded {len(df_rps)} RPS records")

    df_rps.columns = df_rps.columns.str.strip()
    for col in ["State", "STATE", "State/Teritory", "State/Territory"]:
        if col in df_rps.columns:
            df_rps = df_rps.rename(columns={col: "State"})
            print(f"   ✅ Using state column: '{col}'")
            break

    if "Policy/Incentive type" in df_rps.columns:
        df_rps = df_rps[df_rps["Policy/Incentive type"].str.contains(
            "RPS|Renewable|Portfolio|Standard", case=False, na=False)]
        print(f"   Filtered to {len(df_rps)} RPS policies")

    rps_impl = {
        "CA": {"year": 2002, "target": 50},   "CO": {"year": 2004, "target": 30},
        "CT": {"year": 1998, "target": 40},   "DE": {"year": 2005, "target": 25},
        "IL": {"year": 2007, "target": 25},   "MA": {"year": 2002, "target": 35},
        "MD": {"year": 2004, "target": 50},   "ME": {"year": 1999, "target": 40},
        "MI": {"year": 2008, "target": 15},   "MN": {"year": 2007, "target": 25},
        "MO": {"year": 2007, "target": 15},   "MT": {"year": 2005, "target": 15},
        "NC": {"year": 2007, "target": 12.5}, "NH": {"year": 2007, "target": 25},
        "NJ": {"year": 1999, "target": 50},   "NM": {"year": 2002, "target": 50},
        "NV": {"year": 1997, "target": 50},   "NY": {"year": 2004, "target": 70},
        "OH": {"year": 2008, "target": 12.5}, "OR": {"year": 2007, "target": 50},
        "PA": {"year": 2004, "target": 18},   "RI": {"year": 2004, "target": 38.5},
        "TX": {"year": 1999, "target": 10000},"VT": {"year": 2005, "target": 75},
        "WA": {"year": 2006, "target": 15},   "WI": {"year": 1999, "target": 10},
        "DC": {"year": 2005, "target": 100},
    }
    all_states = [
        "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID",
        "IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS",
        "MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK",
        "OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV",
        "WI","WY","DC",
    ]

    records = []
    for state in all_states:
        info = rps_impl.get(state)
        for year in range(years_range[0], years_range[1] + 1):
            for month in range(1, 13):
                has_rps, ysr, impl_yr, tgt = 0, None, None, None
                if info:
                    impl_yr = info["year"]
                    tgt     = info["target"]
                    if year >= impl_yr:
                        has_rps = 1
                        ysr     = year - impl_yr
                records.append({
                    "STATE": state, "YEAR": year, "MONTH": month,
                    "Has_RPS": has_rps, "RPS_Target_Pct": tgt if has_rps else None,
                    "RPS_Implementation_Year": impl_yr, "Years_Since_RPS": ysr,
                })

    df_panel = pd.DataFrame(records)
    print(f"   ✅ {len(df_panel):,} state-month records | "
          f"{df_panel['YEAR'].min()}–{df_panel['YEAR'].max()} | "
          f"{df_panel[df_panel['Has_RPS']==1]['STATE'].nunique()} RPS states")
    return df_panel


# ============================================================================
# FUNCTION 9: DIAGNOSE MERGE KEYS
# ============================================================================

def diagnose_merge_issues(df_gen, df_cons, df_temp, df_gdp, df_rps, df_egrid):
    print("\n🔬 DIAGNOSING MERGE KEYS...")
    print("=" * 80)
    for name, df in [
        ("GENERATION", df_gen), ("CONSUMPTION", df_cons), ("TEMPERATURE", df_temp),
        ("GDP", df_gdp), ("eGRID", df_egrid),
    ]:
        n_unique = df.groupby(["STATE", "YEAR", "MONTH"]).size().shape[0]
        dups     = len(df) - n_unique
        yr_range = f"{df['YEAR'].min()}–{df['YEAR'].max()}"
        print(f"   {name:12s}: {len(df):>7,} rows | {n_unique:>7,} unique | "
              f"{dups:>4} dups | {yr_range}")

    test = df_gen.merge(df_cons, on=["STATE", "YEAR", "MONTH"], how="left", indicator=True)
    print(f"\n   Gen+Cons merge: {len(test):,} rows | "
          f"both={test['_merge'].eq('both').sum():,} | "
          f"left_only={test['_merge'].eq('left_only').sum():,}")
    print("=" * 80)


# ============================================================================
# FUNCTION 10: MERGE ALL DATASETS
# ============================================================================

def merge_all_datasets(df_gen, df_cons, df_temp, df_gdp, df_rps, df_egrid):
    print("\n🔗 MERGING ALL DATASETS...")
    print("=" * 80)

    df = df_gen.copy()
    print(f"1. Base (Generation):    {len(df):>7,} rows")

    for label, right in [
        ("Consumption", df_cons),
        ("Temperature", df_temp),
        ("GDP",         df_gdp),
        ("RPS",         df_rps),
        ("eGRID",       df_egrid),
    ]:
        prev = len(df)
        df   = df.merge(right, on=["STATE", "YEAR", "MONTH"], how="left")
        diff = len(df) - prev
        flag = f"⚠️  GREW by {diff:,}" if diff != 0 else "✓"
        print(f"   + {label:12s}: {len(df):>7,} rows  {flag}")

    print("=" * 80)

    dups = df.groupby(["STATE", "YEAR", "MONTH"]).size()
    dups = dups[dups > 1]
    if len(dups):
        print(f"\n⚠️  {len(dups):,} duplicate state-months detected — removing...")
        df = df.drop_duplicates(subset=["STATE", "YEAR", "MONTH"], keep="first")
        print(f"   ✅ After dedup: {len(df):,} rows")
    else:
        print(f"\n✅ No duplicates — {len(df):,} unique state-months")

    years   = df["YEAR"].max() - df["YEAR"].min() + 1
    max_pos = years * 53 * 12
    print(f"\n📊 Coverage: {len(df):,} / {max_pos:,} "
          f"({len(df)/max_pos*100:.1f}%)  [{years} yrs × 53 states × 12 months]")
    return df


# ============================================================================
# FUNCTION 11: CREATE CO2 PROXY
# ============================================================================

def create_co2_intensity_proxy(df):
    print("\n🔄 CREATING CO2 INTENSITY PROXY...")
    COAL_F, GAS_F, OIL_F = 1.115, 0.455, 1.065
    df["CO2_Emissions_Tons_Proxy"] = (
        df["Coal"].fillna(0) * COAL_F +
        df["Natural Gas"].fillna(0) * GAS_F +
        df["Petroleum"].fillna(0) * OIL_F
    )
    df["CO2_Intensity_Proxy"] = np.where(
        df["Total_Generation_MWh"] > 0,
        df["CO2_Emissions_Tons_Proxy"] / df["Total_Generation_MWh"], np.nan)

    comparison = df[
        df["CO2_Intensity_Tons_per_MWh"].notna() & df["CO2_Intensity_Proxy"].notna()]
    if len(comparison):
        r = comparison["CO2_Intensity_Tons_per_MWh"].corr(comparison["CO2_Intensity_Proxy"])
        print(f"   📊 Proxy validation vs actual eGRID ({len(comparison):,} records):")
        print(f"      Correlation  : {r:.4f}")
        print(f"      Mean actual  : {comparison['CO2_Intensity_Tons_per_MWh'].mean():.4f} tons/MWh")
        print(f"      Mean proxy   : {comparison['CO2_Intensity_Proxy'].mean():.4f} tons/MWh")

    df["CO2_Intensity_Combined"] = df["CO2_Intensity_Tons_per_MWh"].fillna(
        df["CO2_Intensity_Proxy"])

    actual = df["CO2_Intensity_Tons_per_MWh"].notna().sum()
    proxy  = (df["CO2_Intensity_Combined"].notna() &
              df["CO2_Intensity_Tons_per_MWh"].isna()).sum()
    total  = df["CO2_Intensity_Combined"].notna().sum()
    n      = len(df)
    print(f"\n   📊 CO2 intensity coverage:")
    print(f"      Actual eGRID : {actual:>6,} ({actual/n*100:.1f}%)")
    print(f"      Proxy        : {proxy:>6,} ({proxy/n*100:.1f}%)")
    print(f"      Total        : {total:>6,} ({total/n*100:.1f}%)")
    print(f"      Missing      : {n-total:>6,} ({(n-total)/n*100:.1f}%)")
    return df


# ============================================================================
# FUNCTION 12: ADD DERIVED VARIABLES
# ============================================================================

def add_derived_variables(df):
    print("\n➕ ADDING DERIVED VARIABLES...")

    df["date"] = pd.to_datetime(df[["YEAR", "MONTH"]].assign(DAY=1))
    df["High_Demand_Month"] = df["MONTH"].isin([1, 2, 6, 7, 8, 12]).astype(int)

    if df["Avg_Temp_F"].notna().sum() > 0:
        q25 = df["Avg_Temp_F"].quantile(0.25)
        q75 = df["Avg_Temp_F"].quantile(0.75)
        df["Temp_Extreme"] = ((df["Avg_Temp_F"] < q25) | (df["Avg_Temp_F"] > q75)).astype(int)
        print(f"   ✅ Temp_Extreme: <{q25:.1f}°F or >{q75:.1f}°F")

    df["Fossil_Intensity"] = np.where(
        df["Total_Generation_MWh"] > 0,
        df["Fossil_Generation_MWh"] / df["Total_Generation_MWh"] * 100, np.nan)
    valid = df["Fossil_Intensity"].notna().sum()
    print(f"   ✅ Fossil_Intensity: {valid:,}/{len(df):,} ({valid/len(df)*100:.1f}%) valid")

    df["Has_RPS"]         = df["Has_RPS"].fillna(0).astype(int)
    df["RPS_Target_Pct"]  = df["RPS_Target_Pct"].fillna(0)
    df["Years_Since_RPS"] = df["Years_Since_RPS"].fillna(0)

    df = create_co2_intensity_proxy(df)

    if "Renewable_Share_Pct" in df.columns:
        df.loc[df["Renewable_Share_Pct"] > 100, "Renewable_Share_Pct"] = 100
    if "Fossil_Share_Pct" in df.columns:
        df.loc[df["Fossil_Share_Pct"] < 0, "Fossil_Share_Pct"] = 0

    # BH4 target variable: High_Fossil_Backup (1 if Fossil_Intensity > 75th percentile)
    p75 = df["Fossil_Intensity"].quantile(0.75)
    df["High_Fossil_Backup"] = (df["Fossil_Intensity"] > p75).astype(int)

    print("   ✅ Added: date, High_Demand_Month, Temp_Extreme, Fossil_Intensity, "
          "CO2_Intensity_Combined, High_Fossil_Backup")
    return df


# ============================================================================
# FUNCTION 13: FINAL VALIDATION
# ============================================================================

def final_validation(df):
    print("\n" + "=" * 80)
    print("  🔍 FINAL DATASET VALIDATION")
    print("=" * 80)

    dups = df.groupby(["STATE", "YEAR", "MONTH"]).size()
    dups = dups[dups > 1]
    if len(dups):
        print(f"❌ CRITICAL: {len(dups):,} duplicate state-months!")
        raise ValueError("Duplicates found — fix before proceeding")
    print(f"✅ No duplicates: all {len(df):,} observations are unique")

    years   = df["YEAR"].max() - df["YEAR"].min() + 1
    max_pos = years * 53 * 12
    print(f"\n📊 Full dataset:  {len(df):,} / {max_pos:,}  ({len(df)/max_pos*100:.1f}%)")

    df_core = df[(df["YEAR"] >= 2005) & (df["YEAR"] <= 2022)]
    exp     = 18 * 53 * 12
    print(f"📊 Core 2005-2022: {len(df_core):,} / {exp:,}  ({len(df_core)/exp*100:.1f}%)")

    co2_ok = df_core["CO2_Intensity_Combined"].notna().sum()
    print(f"📊 CO2 coverage (core): {co2_ok:,}/{len(df_core):,}  ({co2_ok/len(df_core)*100:.1f}%)")
    print(f"\n✅ DATASET IS VALID AND READY FOR ANALYSIS!")
    print("=" * 80)


# ============================================================================
# FUNCTION 14: SAVE OUTPUTS
# ============================================================================

def save_outputs(df, output_dir=None):
    if output_dir is None:
        output_dir = OUTPUT_DIR
    print("\n💾 SAVING OUTPUTS...")
    print("=" * 80)

    out_full = output_dir / "FINAL_MASTER_DATASET_2001_2026.csv"
    df.to_csv(out_full, index=False)
    print(f"✅ Full dataset:    {out_full.name}  ({len(df):,} rows, {len(df.columns)} cols)")

    compact_cols = [
        "STATE", "YEAR", "MONTH", "date",
        "Total_Generation_MWh", "Renewable_Generation_MWh",
        "Fossil_Generation_MWh", "Nuclear_Generation_MWh",
        "Renewable_Share_Pct", "Fossil_Share_Pct", "Nuclear_Share_Pct",
        "Total_Consumption_MWh", "Avg_Temp_F", "Temp_Extreme",
        "Real_GDP_Millions", "GDP_Growth_Rate_Annual",
        "Has_RPS", "RPS_Target_Pct", "Years_Since_RPS",
        "CO2_Intensity_Tons_per_MWh", "CO2_Intensity_Proxy", "CO2_Intensity_Combined",
        "High_Demand_Month", "Fossil_Intensity", "High_Fossil_Backup",
    ]
    avail = [c for c in compact_cols if c in df.columns]
    out_compact = output_dir / "FINAL_COMPACT_DATASET_2001_2026.csv"
    df[avail].to_csv(out_compact, index=False)
    print(f"✅ Compact dataset: {out_compact.name}  ({len(df):,} rows, {len(avail)} cols)")

    out_stats = output_dir / "SUMMARY_STATISTICS.csv"
    df.describe().to_csv(out_stats)
    print(f"✅ Summary stats:   {out_stats.name}")
    print("=" * 80)
    return out_full, out_compact


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("  U.S. ENERGY TRANSITION DATA PIPELINE")
    print("  Generation + Consumption + eGRID (2004-2023) + Temperature + GDP + RPS")
    print("  All sources: public government data (EIA, EPA, NOAA, BEA, DSIRE) — no Kaggle")
    print("=" * 80)
    t0 = time.time()

    if not verify_input_files():
        print("\n⚠️  Core input files missing — aborting")
        return None

    df_gen_raw = load_multi_sheet_generation(INPUT_FILES["generation"])
    df_gen     = transform_generation_to_wide(df_gen_raw)
    df_cons    = load_consumption_data(INPUT_FILES["consumption"])
    df_egrid   = load_all_egrid_files()
    df_temp    = load_temperature_data(INPUT_FILES["temperature"])
    df_gdp     = process_gdp_data_annual(INPUT_FILES["gdp"])
    df_rps     = create_rps_panel(INPUT_FILES["rps"], (2001, 2026))

    diagnose_merge_issues(df_gen, df_cons, df_temp, df_gdp, df_rps, df_egrid)

    df_master = merge_all_datasets(df_gen, df_cons, df_temp, df_gdp, df_rps, df_egrid)
    df_master = add_derived_variables(df_master)

    try:
        final_validation(df_master)
    except ValueError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        return None

    save_outputs(df_master)

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"  ✅ PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"{'=' * 80}")
    print(f"\n📊 FINAL DATASET:  {len(df_master):,} obs | "
          f"{df_master['STATE'].nunique()} states | "
          f"{df_master['YEAR'].min()}–{df_master['YEAR'].max()} | "
          f"{len(df_master.columns)} variables")

    print(f"\n❓ MISSING DATA SUMMARY:")
    for col in df_master.columns:
        miss = df_master[col].isna().sum()
        if miss > 0:
            print(f"   {col:35s}: {miss:>6,} ({miss/len(df_master)*100:>5.1f}%)")

    print(f"\n{'=' * 80}")
    return df_master


if __name__ == "__main__":
    df_final = main()
