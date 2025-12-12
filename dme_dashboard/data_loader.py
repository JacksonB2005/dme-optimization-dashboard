import pandas as pd

EXCEL_FILE = "Operations Final X.xlsx"


def normalize_name(s: str) -> str:
    """Helper to match 'Wheelchair' vs 'Wheelchairs' etc."""
    return s.strip().lower().replace(" ", "").rstrip("s")


def load_data(excel_file: str = EXCEL_FILE):
    """
    Read all cleaned Excel sheets and build sets + parameter dictionaries.
    Returns a dict with keys:
      Z, K, R, W, T,
      supply, demand,
      dist_ZR, dist_RW, dist_WZ,
      level_ZR, level_RW, level_WZ,
      traffic_rate, gamma,
      lane_cap_ZR, lane_cap_WZ,
      C_ZR, C_RW, C_WZ
    """

    # -------------------------
    # 1) Supply & Demand
    # -------------------------
    supply_df = pd.read_excel(excel_file, sheet_name="SupplyClean")
    demand_df = pd.read_excel(excel_file, sheet_name="DemandClean")

    # Make sure first column is called "Zone"
    supply_df = supply_df.rename(columns={supply_df.columns[0]: "Zone"})
    demand_df = demand_df.rename(columns={demand_df.columns[0]: "Zone"})

    # Sets
    Z = [z.strip() for z in supply_df["Zone"].tolist()]        # zones
    K = [c.strip() for c in supply_df.columns[1:].tolist()]    # equipment types
    R = ["R1", "R2"]                                           # refurb centers
    W = ["W1"]                                                 # single warehouse
    T = [1]                                                    # single time period (for now)

    # Supply[z,k]
    supply = {
        (row["Zone"], k): float(row[k])
        for _, row in supply_df.iterrows()
        for k in K
    }

    # Demand[z,k]
    demand = {
        (row["Zone"], k): float(row[k])
        for _, row in demand_df.iterrows()
        for k in K
    }

    # -------------------------
    # 2) Distances & Levels
    # -------------------------

    # 2a) Zone -> Refurb (DistClean_ZR)
    distZR_df = pd.read_excel(excel_file, sheet_name="DistClean_ZR")
    # Force clean column names: Zone, R1_Dist, R2_Dist, R1_Level, R2_Level
    distZR_df = distZR_df.rename(
        columns={
            distZR_df.columns[0]: "Zone",
            distZR_df.columns[1]: "R1_Dist",
            distZR_df.columns[2]: "R2_Dist",
            distZR_df.columns[3]: "R1_Level",
            distZR_df.columns[4]: "R2_Level",
        }
    )

    dist_ZR = {}
    level_ZR = {}
    for _, row in distZR_df.iterrows():
        z = row["Zone"]
        dist_ZR[(z, "R1")] = float(row["R1_Dist"])
        dist_ZR[(z, "R2")] = float(row["R2_Dist"])
        level_ZR[(z, "R1")] = str(row["R1_Level"])
        level_ZR[(z, "R2")] = str(row["R2_Level"])

    # 2b) Refurb -> Warehouse (DistClean_RW)
    distRW_df = pd.read_excel(excel_file, sheet_name="DistClean_RW")
    distRW_df = distRW_df.rename(
        columns={
            distRW_df.columns[0]: "Refurb",
            distRW_df.columns[1]: "Distance",
            distRW_df.columns[2]: "Level",
        }
    )

    dist_RW = {}
    level_RW = {}
    for _, row in distRW_df.iterrows():
        r = row["Refurb"]
        dist_RW[r] = float(row["Distance"])
        level_RW[r] = str(row["Level"])

    # 2c) Warehouse -> Zones (DistClean_WZ)
    distWZ_df = pd.read_excel(excel_file, sheet_name="DistClean_WZ")
    distWZ_df = distWZ_df.rename(
        columns={
            distWZ_df.columns[0]: "Zone",
            distWZ_df.columns[1]: "Distance",
            distWZ_df.columns[2]: "Level",
        }
    )

    dist_WZ = {}
    level_WZ = {}
    for _, row in distWZ_df.iterrows():
        z = row["Zone"]
        dist_WZ[z] = float(row["Distance"])
        level_WZ[z] = str(row["Level"])

    # -------------------------
    # 3) Traffic rates & gamma
    # -------------------------

    # TrafficRatesClean: Level, Rate
    traffic_df = pd.read_excel(excel_file, sheet_name="TrafficRatesClean")
    traffic_df = traffic_df.rename(
        columns={
            traffic_df.columns[0]: "Level",
            traffic_df.columns[1]: "Rate",
        }
    )

    traffic_rate = {
        str(row["Level"]): float(row["Rate"])
        for _, row in traffic_df.iterrows()
    }

    # GammaClean: Type, Gamma (but names may be singular vs plural)
    gamma_df = pd.read_excel(excel_file, sheet_name="GammaClean")
    gamma_df = gamma_df.rename(
        columns={
            gamma_df.columns[0]: "Type",
            gamma_df.columns[1]: "Gamma",
        }
    )

    # Build map by normalized name
    gamma_by_norm = {
        normalize_name(row["Type"]): float(row["Gamma"])
        for _, row in gamma_df.iterrows()
    }

    gamma = {}
    for k in K:
        nk = normalize_name(k)
        if nk not in gamma_by_norm:
            raise KeyError(f"Gamma not found for equipment type '{k}'")
        gamma[k] = gamma_by_norm[nk]

    # -------------------------
    # 4) Lane capacities
    # -------------------------

    # LaneCap_ZR: Zone, R1_Cap, R2_Cap
    laneZR_df = pd.read_excel(excel_file, sheet_name="LaneCap_ZR")
    laneZR_df = laneZR_df.rename(
        columns={
            laneZR_df.columns[0]: "Zone",
            laneZR_df.columns[1]: "R1_Cap",
            laneZR_df.columns[2]: "R2_Cap",
        }
    )

    lane_cap_ZR = {}
    for _, row in laneZR_df.iterrows():
        z = row["Zone"]
        lane_cap_ZR[(z, "R1")] = float(row["R1_Cap"])
        lane_cap_ZR[(z, "R2")] = float(row["R2_Cap"])

    # LaneCap_WZ: Zone, Cap_WZ
    laneWZ_df = pd.read_excel(excel_file, sheet_name="LaneCap_Wz")
    laneWZ_df = laneWZ_df.rename(
        columns={
            laneWZ_df.columns[0]: "Zone",
            laneWZ_df.columns[1]: "Cap_WZ",
        }
    )

    lane_cap_WZ = {
        row["Zone"]: float(row["Cap_WZ"])
        for _, row in laneWZ_df.iterrows()
    }

    # -------------------------
    # 5) Travel cost coefficients
    #    C_ZR[z,r], C_RW[r], C_WZ[z]
    #    = distance * traffic_rate(level)
    # -------------------------

    C_ZR = {}
    for z in Z:
        for r in R:
            lvl = level_ZR[(z, r)]
            C_ZR[(z, r)] = dist_ZR[(z, r)] * traffic_rate[lvl]

    C_RW = {}
    for r in R:
        lvl = level_RW[r]
        C_RW[r] = dist_RW[r] * traffic_rate[lvl]

    C_WZ = {}
    for z in Z:
        lvl = level_WZ[z]
        C_WZ[z] = dist_WZ[z] * traffic_rate[lvl]

    # -------------------------
    # Pack everything
    # -------------------------
    data = {
        "Z": Z,
        "K": K,
        "R": R,
        "W": W,
        "T": T,
        "supply": supply,
        "demand": demand,
        "dist_ZR": dist_ZR,
        "dist_RW": dist_RW,
        "dist_WZ": dist_WZ,
        "level_ZR": level_ZR,
        "level_RW": level_RW,
        "level_WZ": level_WZ,
        "traffic_rate": traffic_rate,
        "gamma": gamma,
        "lane_cap_ZR": lane_cap_ZR,
        "lane_cap_WZ": lane_cap_WZ,
        "C_ZR": C_ZR,
        "C_RW": C_RW,
        "C_WZ": C_WZ,
    }

    return data


if __name__ == "__main__":
    # Quick self-test
    d = load_data()
    print("Zones (Z):", d["Z"])
    print("Types (K):", d["K"])
    print("Refurb centers (R):", d["R"])
    print("Warehouse (W):", d["W"])
    print("Time periods (T):", d["T"])

    print("\nSample supply entries:", list(d["supply"].items())[:5])
    print("Sample demand entries:", list(d["demand"].items())[:5])
    print("Sample C_ZR entries:", list(d["C_ZR"].items())[:4])
    print("C_RW:", d["C_RW"])
    print("C_WZ:", d["C_WZ"])
