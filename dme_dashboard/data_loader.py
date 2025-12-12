from __future__ import annotations

from pathlib import Path
import pandas as pd


# -------------------------
# Path to Excel file
# -------------------------
APP_DIR = Path(__file__).resolve().parent
EXCEL_FILE = APP_DIR / "Operations Final X.xlsx"


def normalize_name(s: str) -> str:
    """Helper to match 'Wheelchair' vs 'Wheelchairs' etc."""
    return str(s).strip().lower().replace(" ", "").rstrip("s")


def _require_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Excel file not found: {path}\n"
            f"Put 'Operations Final X.xlsx' in the SAME folder as data_loader.py "
            f"(currently: {APP_DIR})."
        )


def _safe_rename_by_index(df: pd.DataFrame, mapping: dict[int, str]) -> pd.DataFrame:
    cols = list(df.columns)
    rename = {}
    for idx, new_name in mapping.items():
        if idx >= len(cols):
            raise ValueError(f"Expected at least {idx+1} columns but got {len(cols)}. Columns={cols}")
        rename[cols[idx]] = new_name
    return df.rename(columns=rename)


def load_data(excel_file: str | Path = EXCEL_FILE) -> dict:
    """
    Returns a dict with keys that match what model.py expects:
    Z, K, R, W, T,
    supply, demand,
    traffic_rate, gamma,
    lane_cap_ZR, lane_cap_WZ,
    C_ZR, C_RW, C_WZ
    """
    excel_file = Path(excel_file)
    _require_file_exists(excel_file)

    # -------------------------
    # 1) Supply & Demand
    # -------------------------
    supply_df = pd.read_excel(excel_file, sheet_name="SupplyClean")
    demand_df = pd.read_excel(excel_file, sheet_name="DemandClean")

    supply_df = supply_df.rename(columns={supply_df.columns[0]: "Zone"})
    demand_df = demand_df.rename(columns={demand_df.columns[0]: "Zone"})

    Z = [str(z).strip() for z in supply_df["Zone"].tolist()]
    K = [str(c).strip() for c in supply_df.columns[1:].tolist()]

    R = ["R1", "R2"]
    W = ["W1"]
    T = [1]

    supply = {(row["Zone"], k): float(row[k]) for _, row in supply_df.iterrows() for k in K}
    demand = {(row["Zone"], k): float(row[k]) for _, row in demand_df.iterrows() for k in K}

    # -------------------------
    # 2) Distances + Levels
    # -------------------------
    # DistClean_ZR: Zone, R1_Dist, R2_Dist, R1_Level, R2_Level
    distZR_df = pd.read_excel(excel_file, sheet_name="DistClean_ZR")
    distZR_df = _safe_rename_by_index(
        distZR_df,
        {0: "Zone", 1: "R1_Dist", 2: "R2_Dist", 3: "R1_Level", 4: "R2_Level"},
    )

    dist_ZR = {}
    level_ZR = {}
    for _, row in distZR_df.iterrows():
        z = str(row["Zone"]).strip()
        dist_ZR[(z, "R1")] = float(row["R1_Dist"])
        dist_ZR[(z, "R2")] = float(row["R2_Dist"])
        level_ZR[(z, "R1")] = str(row["R1_Level"]).strip()
        level_ZR[(z, "R2")] = str(row["R2_Level"]).strip()

    # DistClean_RW: Refurb, Distance, Level  (LEVEL is text like Low/Med/High)
    distRW_df = pd.read_excel(excel_file, sheet_name="DistClean_RW")
    distRW_df = _safe_rename_by_index(distRW_df, {0: "Refurb", 1: "Distance", 2: "Level"})

    dist_RW = {}
    level_RW = {}
    for _, row in distRW_df.iterrows():
        r = str(row["Refurb"]).strip()  # should be R1/R2
        dist_RW[r] = float(row["Distance"])
        level_RW[r] = str(row["Level"]).strip()

    # DistClean_WZ: Zone, Distance, Level
    distWZ_df = pd.read_excel(excel_file, sheet_name="DistClean_WZ")
    distWZ_df = _safe_rename_by_index(distWZ_df, {0: "Zone", 1: "Distance", 2: "Level"})

    dist_WZ = {}
    level_WZ = {}
    for _, row in distWZ_df.iterrows():
        z = str(row["Zone"]).strip()
        dist_WZ[z] = float(row["Distance"])
        level_WZ[z] = str(row["Level"]).strip()

    # -------------------------
    # 3) Traffic rates + gamma
    # -------------------------
    traffic_df = pd.read_excel(excel_file, sheet_name="TrafficRatesClean")
    traffic_df = _safe_rename_by_index(traffic_df, {0: "Level", 1: "Rate"})

    traffic_rate = {str(row["Level"]).strip(): float(row["Rate"]) for _, row in traffic_df.iterrows()}

    gamma_df = pd.read_excel(excel_file, sheet_name="GammaClean")
    gamma_df = _safe_rename_by_index(gamma_df, {0: "Type", 1: "Gamma"})

    gamma_by_norm = {normalize_name(row["Type"]): float(row["Gamma"]) for _, row in gamma_df.iterrows()}
    gamma = {}
    for k in K:
        nk = normalize_name(k)
        if nk not in gamma_by_norm:
            raise KeyError(f"Gamma not found for equipment type '{k}'")
        gamma[k] = gamma_by_norm[nk]

    # -------------------------
    # 4) Lane capacities
    # -------------------------
    laneZR_df = pd.read_excel(excel_file, sheet_name="LaneCap_ZR")
    laneZR_df = _safe_rename_by_index(laneZR_df, {0: "Zone", 1: "R1_Cap", 2: "R2_Cap"})

    lane_cap_ZR = {}
    for _, row in laneZR_df.iterrows():
        z = str(row["Zone"]).strip()
        lane_cap_ZR[(z, "R1")] = float(row["R1_Cap"])
        lane_cap_ZR[(z, "R2")] = float(row["R2_Cap"])

    laneWZ_df = pd.read_excel(excel_file, sheet_name="LaneCap_Wz")
    laneWZ_df = _safe_rename_by_index(laneWZ_df, {0: "Zone", 1: "Cap_WZ"})

    lane_cap_WZ = {str(row["Zone"]).strip(): float(row["Cap_WZ"]) for _, row in laneWZ_df.iterrows()}

    # -------------------------
    # 5) Cost coefficients = distance * traffic_rate(level)
    # -------------------------
    C_ZR = {}
    for z in Z:
        for r in R:
            lvl = level_ZR[(z, r)]
            if lvl not in traffic_rate:
                raise KeyError(f"Traffic level '{lvl}' not found in TrafficRatesClean")
            C_ZR[(z, r)] = dist_ZR[(z, r)] * traffic_rate[lvl]

    C_RW = {}
    for r in R:
        # dist_RW keys should be 'R1','R2' — handle if sheet uses something else
        key = r
        if key not in dist_RW:
            # try case-insensitive match
            matches = [k for k in dist_RW.keys() if str(k).strip().lower() == r.lower()]
            if matches:
                key = matches[0]
            else:
                raise KeyError(f"DistClean_RW missing refurb row for '{r}'. Found keys: {list(dist_RW.keys())}")

        lvl = level_RW[key]
        if lvl not in traffic_rate:
            raise KeyError(f"Traffic level '{lvl}' not found in TrafficRatesClean")
        C_RW[r] = dist_RW[key] * traffic_rate[lvl]

    C_WZ = {}
    for z in Z:
        lvl = level_WZ[z]
        if lvl not in traffic_rate:
            raise KeyError(f"Traffic level '{lvl}' not found in TrafficRatesClean")
        C_WZ[z] = dist_WZ[z] * traffic_rate[lvl]

    return {
        "Z": Z,
        "K": K,
        "R": R,
        "W": W,
        "T": T,
        "supply": supply,
        "demand": demand,
        "traffic_rate": traffic_rate,
        "gamma": gamma,
        "lane_cap_ZR": lane_cap_ZR,
        "lane_cap_WZ": lane_cap_WZ,
        "C_ZR": C_ZR,
        "C_RW": C_RW,
        "C_WZ": C_WZ,
    }


if __name__ == "__main__":
    d = load_data()
    print("Loaded OK:", len(d["Z"]), "zones,", len(d["K"]), "types")
