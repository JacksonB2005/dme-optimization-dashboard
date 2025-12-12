import pulp
import pandas as pd
from data_loader import load_data


# -----------------------------
# Helpers
# -----------------------------
def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("-", "_")


def _col(df: pd.DataFrame, candidates):
    """
    Find a column in df whose normalized name matches any candidate (also normalized).
    """
    cols = list(df.columns)
    norm_map = {_norm(c): c for c in cols}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def _to_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _traffic_multiplier_map(traffic_df: pd.DataFrame) -> dict:
    """
    Build Level -> multiplier map from TrafficRatesClean if possible.
    If not possible, return defaults.
    """
    if traffic_df is None or traffic_df.empty:
        return {"low": 1.0, "med": 1.0, "medium": 1.0, "high": 1.0}

    lvl_col = _col(traffic_df, ["level", "service_level", "tier"])
    val_col = _col(traffic_df, ["multiplier", "rate", "traffic_rate", "factor"])

    if not lvl_col or not val_col:
        return {"low": 1.0, "med": 1.0, "medium": 1.0, "high": 1.0}

    m = {}
    for _, r in traffic_df[[lvl_col, val_col]].dropna().iterrows():
        lvl = _norm(r[lvl_col])
        val = _to_float(r[val_col])
        if val is not None:
            m[lvl] = val

    # sensible fallbacks
    if "med" not in m and "medium" in m:
        m["med"] = m["medium"]
    for k in ["low", "med", "medium", "high"]:
        m.setdefault(k, 1.0)

    return m


def _gamma_map(gamma_df: pd.DataFrame, K: list) -> dict:
    """
    Build type -> gamma map. If sheet missing or columns unclear, default gamma=1 for all K.
    """
    if gamma_df is None or gamma_df.empty:
        return {k: 1.0 for k in K}

    type_col = _col(gamma_df, ["type", "category", "k"])
    gam_col = _col(gamma_df, ["gamma", "weight", "multiplier", "factor"])

    if not type_col or not gam_col:
        return {k: 1.0 for k in K}

    g = {}
    for _, r in gamma_df[[type_col, gam_col]].dropna().iterrows():
        t = str(r[type_col]).strip()
        v = _to_float(r[gam_col])
        if v is not None:
            g[t] = v

    return {k: float(g.get(k, 1.0)) for k in K}


# -----------------------------
# Main optimization model
# -----------------------------
def run_model(
    cost_mult_ZR=1.0,
    cost_mult_RW=1.0,
    cost_mult_WZ=1.0,
    cap_mult=1.0,
):
    """
    Solve DME network LP.
    Uses these sheets (exactly like your workbook tabs):
      - SupplyClean, DemandClean
      - DistClean_ZR, DistClean_RW, DistClean_WZ
      - LaneCap_ZR, LaneCap_Wz
      - TrafficRatesClean, GammaClean
    """

    data = load_data()

    supply_df = data["supply"].copy()
    demand_df = data["demand"].copy()
    dist_zr_df = data["dist_ZR"].copy()
    dist_rw_df = data["dist_RW"].copy()
    dist_wz_df = data["dist_WZ"].copy()
    cap_zr_df = data["cap_ZR"].copy()
    cap_wz_df = data["cap_WZ"].copy()
    traffic_df = data.get("traffic", pd.DataFrame()).copy()
    gamma_df = data.get("gamma", pd.DataFrame()).copy()

    # ---- Build sets Z, K ----
    zone_col_s = _col(supply_df, ["zone"])
    zone_col_d = _col(demand_df, ["zone"])
    if not zone_col_s or not zone_col_d:
        raise ValueError(f"Supply/Demand must have a 'Zone' column. "
                         f"Supply cols={list(supply_df.columns)} Demand cols={list(demand_df.columns)}")

    supply_df[zone_col_s] = supply_df[zone_col_s].astype(str).str.strip()
    demand_df[zone_col_d] = demand_df[zone_col_d].astype(str).str.strip()

    Z = sorted(set(supply_df[zone_col_s]).intersection(set(demand_df[zone_col_d])))

    # Assume "types" are all columns except Zone
    K = [c for c in supply_df.columns if c != zone_col_s]
    if len(K) == 0:
        # fallback: single commodity
        K = ["All"]

    # ---- Refurb set R ----
    # From DistClean_RW or DistClean_ZR columns
    r_col_rw = _col(dist_rw_df, ["refurb", "refurbcenter", "r"])
    if r_col_rw:
        R = sorted(dist_rw_df[r_col_rw].dropna().astype(str).str.strip().unique().tolist())
    else:
        # fallback: parse R? columns like R1_Dist, R2_Dist
        R = []
        for c in dist_zr_df.columns:
            n = _norm(c)
            if n.endswith("_dist") and n.startswith("r"):
                R.append(c.split("_")[0].strip())
        R = sorted(set(R))

    if not R:
        raise ValueError(f"Could not infer refurb centers R from DistClean_RW or DistClean_ZR. "
                         f"DistClean_RW cols={list(dist_rw_df.columns)}, DistClean_ZR cols={list(dist_zr_df.columns)}")

    # ---- Time periods ----
    T = ["T1"]  # single-period model (matches your original approach)

    # ---- Build parameter maps ----
    traffic_mult = _traffic_multiplier_map(traffic_df)

    # Supply / Demand dicts
    supply = {}
    demand = {}
    for _, row in supply_df.iterrows():
        z = str(row[zone_col_s]).strip()
        if z not in Z:
            continue
        for k in K:
            supply[(z, k)] = float(row[k]) if _to_float(row[k]) is not None else 0.0

    for _, row in demand_df.iterrows():
        z = str(row[zone_col_d]).strip()
        if z not in Z:
            continue
        for k in K:
            # demand sheet might have slightly different columns; default to 0 if missing
            if k in demand_df.columns:
                demand[(z, k)] = float(row[k]) if _to_float(row[k]) is not None else 0.0
            else:
                demand[(z, k)] = 0.0

    gamma = _gamma_map(gamma_df, K)

    # ---- Costs: Z -> R from wide sheet like:
    # Zone | R1_Dist | R2_Dist | R1_Level | R2_Level
    zone_col_zr = _col(dist_zr_df, ["zone"])
    if not zone_col_zr:
        raise ValueError(f"DistClean_ZR must have Zone column. Columns={list(dist_zr_df.columns)}")
    dist_zr_df[zone_col_zr] = dist_zr_df[zone_col_zr].astype(str).str.strip()

    C_ZR = {}
    for _, row in dist_zr_df.iterrows():
        z = str(row[zone_col_zr]).strip()
        if z not in Z:
            continue
        for r in R:
            dist_col = _col(dist_zr_df, [f"{r}_Dist", f"{r}Dist", f"{r}_distance"])
            lvl_col = _col(dist_zr_df, [f"{r}_Level", f"{r}Level", f"{r}_tier"])
            if not dist_col:
                continue
            dist_val = _to_float(row[dist_col])
            if dist_val is None:
                continue
            lvl_val = _norm(row[lvl_col]) if lvl_col and not pd.isna(row[lvl_col]) else "med"
            m = traffic_mult.get(lvl_val, traffic_mult.get("med", 1.0))
            C_ZR[(z, r)] = dist_val * float(m)

    # ---- Costs: R -> W1 from 3-col sheet like:
    # Refurb | Dist_to_W1 | Level
    a = r_col_rw or _col(dist_rw_df, ["refurb", "r"])
    dist_col_rw = _col(dist_rw_df, ["dist_to_w1", "distance_to_w1", "dist", "distance"])
    lvl_col_rw = _col(dist_rw_df, ["level", "tier"])
    if not a or not dist_col_rw:
        raise ValueError(f"DistClean_RW must have refurb + distance columns. Columns={list(dist_rw_df.columns)}")

    C_RW = {}
    for _, row in dist_rw_df.iterrows():
        rname = str(row[a]).strip()
        if rname not in R:
            continue
        dist_val = _to_float(row[dist_col_rw])
        if dist_val is None:
            continue
        lvl_val = _norm(row[lvl_col_rw]) if lvl_col_rw and not pd.isna(row[lvl_col_rw]) else "med"
        m = traffic_mult.get(lvl_val, traffic_mult.get("med", 1.0))
        C_RW[rname] = dist_val * float(m)

    # ---- Costs: W1 -> Z (try similar formats)
    zone_col_wz = _col(dist_wz_df, ["zone"])
    if not zone_col_wz:
        raise ValueError(f"DistClean_WZ must have Zone column. Columns={list(dist_wz_df.columns)}")
    dist_wz_df[zone_col_wz] = dist_wz_df[zone_col_wz].astype(str).str.strip()

    dist_col_wz = _col(dist_wz_df, ["dist_from_w1", "dist_to_zone", "dist", "distance", "w1_dist"])
    lvl_col_wz = _col(dist_wz_df, ["level", "tier"])
    if not dist_col_wz:
        # fallback: maybe there is only one non-zone column that is numeric
        for c in dist_wz_df.columns:
            if c == zone_col_wz:
                continue
            # pick first column that has any numeric values
            if pd.to_numeric(dist_wz_df[c], errors="coerce").notna().any():
                dist_col_wz = c
                break
    if not dist_col_wz:
        raise ValueError(f"DistClean_WZ needs a numeric distance column. Columns={list(dist_wz_df.columns)}")

    C_WZ = {}
    for _, row in dist_wz_df.iterrows():
        z = str(row[zone_col_wz]).strip()
        if z not in Z:
            continue
        dist_val = _to_float(row[dist_col_wz])
        if dist_val is None:
            continue
        lvl_val = _norm(row[lvl_col_wz]) if lvl_col_wz and not pd.isna(row[lvl_col_wz]) else "med"
        m = traffic_mult.get(lvl_val, traffic_mult.get("med", 1.0))
        C_WZ[z] = dist_val * float(m)

    # ---- Lane capacities ----
    # LaneCap_ZR likely wide: Zone | R1 | R2 ... (or R1_Cap etc)
    zone_col_capzr = _col(cap_zr_df, ["zone"])
    if not zone_col_capzr:
        raise ValueError(f"LaneCap_ZR must have Zone column. Columns={list(cap_zr_df.columns)}")
    cap_zr_df[zone_col_capzr] = cap_zr_df[zone_col_capzr].astype(str).str.strip()

    lane_cap_ZR = {}
    for _, row in cap_zr_df.iterrows():
        z = str(row[zone_col_capzr]).strip()
        if z not in Z:
            continue
        for r in R:
            ccap = _col(cap_zr_df, [f"{r}_Cap", f"{r}Cap", f"{r}", f"{r}_Capacity"])
            if not ccap:
                continue
            v = _to_float(row[ccap])
            if v is not None:
                lane_cap_ZR[(z, r)] = float(v)

    # LaneCap_Wz likely: Zone | Cap (or similar)
    zone_col_capwz = _col(cap_wz_df, ["zone"])
    if not zone_col_capwz:
        raise ValueError(f"LaneCap_Wz must have Zone column. Columns={list(cap_wz_df.columns)}")
    cap_wz_df[zone_col_capwz] = cap_wz_df[zone_col_capwz].astype(str).str.strip()

    cap_col_wz = _col(cap_wz_df, ["cap", "capacity", "lane_cap", "max"])
    if not cap_col_wz:
        # fallback: first numeric col besides zone
        for c in cap_wz_df.columns:
            if c == zone_col_capwz:
                continue
            if pd.to_numeric(cap_wz_df[c], errors="coerce").notna().any():
                cap_col_wz = c
                break
    if not cap_col_wz:
        raise ValueError(f"LaneCap_Wz needs a numeric capacity column. Columns={list(cap_wz_df.columns)}")

    lane_cap_WZ = {}
    for _, row in cap_wz_df.iterrows():
        z = str(row[zone_col_capwz]).strip()
        if z not in Z:
            continue
        v = _to_float(row[cap_col_wz])
        if v is not None:
            lane_cap_WZ[z] = float(v)

    # -----------------------------
    # Build LP
    # -----------------------------
    model = pulp.LpProblem("DME_Recovery_Network", pulp.LpMinimize)

    x = {(z, r, k, t): pulp.LpVariable(f"x_{z}_{r}_{k}_{t}", lowBound=0)
         for z in Z for r in R for k in K for t in T}
    y = {(r, k, t): pulp.LpVariable(f"y_{r}_{k}_{t}", lowBound=0)
         for r in R for k in K for t in T}
    w = {(z, k, t): pulp.LpVariable(f"w_{z}_{k}_{t}", lowBound=0)
         for z in Z for k in K for t in T}
    zDump = {(k, t): pulp.LpVariable(f"zDump_{k}_{t}", lowBound=0)
             for k in K for t in T}

    # Objective
    model += (
        pulp.lpSum(
            cost_mult_ZR * C_ZR.get((z, r), 0.0) * gamma[k] * x[(z, r, k, t)]
            for z in Z for r in R for k in K for t in T
        )
        + pulp.lpSum(
            cost_mult_RW * C_RW.get(r, 0.0) * gamma[k] * y[(r, k, t)]
            for r in R for k in K for t in T
        )
        + pulp.lpSum(
            cost_mult_WZ * C_WZ.get(z, 0.0) * gamma[k] * w[(z, k, t)]
            for z in Z for k in K for t in T
        ),
        "Total_Weekly_Transportation_Cost",
    )

    # Constraints
    for z in Z:
        for k in K:
            for t in T:
                model += (
                    pulp.lpSum(x[(z, r, k, t)] for r in R) <= supply.get((z, k), 0.0),
                    f"SupplyLimit_{z}_{k}_{t}",
                )

    for r in R:
        for k in K:
            for t in T:
                model += (
                    pulp.lpSum(x[(z, r, k, t)] for z in Z) == y[(r, k, t)],
                    f"RefurbBalance_{r}_{k}_{t}",
                )

    for k in K:
        for t in T:
            model += (
                pulp.lpSum(y[(r, k, t)] for r in R)
                == pulp.lpSum(w[(z, k, t)] for z in Z) + zDump[(k, t)],
                f"WarehouseBalance_{k}_{t}",
            )

    for z in Z:
        for k in K:
            for t in T:
                model += (
                    w[(z, k, t)] >= demand.get((z, k), 0.0),
                    f"Demand_{z}_{k}_{t}",
                )

    for z in Z:
        for r in R:
            for t in T:
                model += (
                    pulp.lpSum(gamma[k] * x[(z, r, k, t)] for k in K)
                    <= cap_mult * lane_cap_ZR.get((z, r), 1e18),
                    f"LaneCap_ZR_{z}_{r}_{t}",
                )

    for z in Z:
        for t in T:
            model += (
                pulp.lpSum(gamma[k] * w[(z, k, t)] for k in K)
                <= cap_mult * lane_cap_WZ.get(z, 1e18),
                f"LaneCap_WZ_{z}_{t}",
            )

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    result_status = model.solve(solver)

    status_str = pulp.LpStatus[result_status]
    obj_value = pulp.value(model.objective)

    # Output tables (single time period)
    t0 = T[0]

    flows_ZR = pd.DataFrame([
        {"Zone": z, "Refurb": r, "Type": k, "Flow_to_Refurb": pulp.value(x[(z, r, k, t0)])}
        for z in Z for r in R for k in K
        if pulp.value(x[(z, r, k, t0)]) and pulp.value(x[(z, r, k, t0)]) > 1e-6
    ])

    flows_RW = pd.DataFrame([
        {"Refurb": r, "Type": k, "Flow_to_Warehouse": pulp.value(y[(r, k, t0)])}
        for r in R for k in K
        if pulp.value(y[(r, k, t0)]) and pulp.value(y[(r, k, t0)]) > 1e-6
    ])

    flows_WZ = pd.DataFrame([
        {"Zone": z, "Type": k, "Flow_to_Zone": pulp.value(w[(z, k, t0)])}
        for z in Z for k in K
        if pulp.value(w[(z, k, t0)]) and pulp.value(w[(z, k, t0)]) > 1e-6
    ])

    # ---- Cost breakdown ----
    cost_ZR = sum(
        cost_mult_ZR * C_ZR.get((z, r), 0.0) * gamma[k] * pulp.value(x[(z, r, k, t)])
        for z in Z for r in R for k in K for t in T
        if pulp.value(x[(z, r, k, t)]) is not None
    )

    cost_RW = sum(
        cost_mult_RW * C_RW.get(r, 0.0) * gamma[k] * pulp.value(y[(r, k, t)])
        for r in R for k in K for t in T
        if pulp.value(y[(r, k, t)]) is not None
    )

    cost_WZ = sum(
        cost_mult_WZ * C_WZ.get(z, 0.0) * gamma[k] * pulp.value(w[(z, k, t)])
        for z in Z for k in K for t in T
        if pulp.value(w[(z, k, t)]) is not None
    )

    cost_breakdown = {
        "Zone_to_Refurb": float(cost_ZR),
        "Refurb_to_Warehouse": float(cost_RW),
        "Warehouse_to_Zone": float(cost_WZ),
        "Total": float(cost_ZR + cost_RW + cost_WZ),
    }




       return {
        "status": status_str,
        "objective": obj_value,
        "cost_breakdown": cost_breakdown,
        "flows_ZR": flows_ZR,
        "flows_RW": flows_RW,
        "flows_WZ": flows_WZ,
    }
