import pulp
import pandas as pd
from data_loader import load_data


def _col(df: pd.DataFrame, *candidates: str) -> str:
    """Return the first matching column name (case-insensitive) from candidates."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise KeyError(f"None of these columns found: {candidates}. Columns = {list(df.columns)}")


def _as_long_cost(df: pd.DataFrame, a: str, b: str, v: str) -> dict:
    """Convert long-form (a,b,value) df -> dict[(a,b)] = value."""
    return {(row[a], row[b]): float(row[v]) for _, row in df.iterrows()}


def _as_long_cap(df: pd.DataFrame, a: str, b: str, v: str) -> dict:
    return {(row[a], row[b]): float(row[v]) for _, row in df.iterrows()}


def run_model(
    cost_mult_ZR=1.0,
    cost_mult_RW=1.0,
    cost_mult_WZ=1.0,
    cap_mult=1.0,
):
    """
    Solves a 3-stage DME flow model:
      Zone -> Refurb -> Warehouse -> Zone
    using the sheets you showed:
      SupplyClean, DemandClean,
      DistClean_ZR, DistClean_RW, DistClean_WZ,
      LaneCap_ZR, LaneCap_Wz,
      TrafficRatesClean, GammaClean
    """

    data = load_data()

    supply_df = data["supply"]
    demand_df = data["demand"]
    dist_zr_df = data["dist_ZR"]
    dist_rw_df = data["dist_RW"]
    dist_wz_df = data["dist_WZ"]
    cap_zr_df = data["cap_ZR"]
    cap_wz_df = data["cap_WZ"]
    traffic_df = data["traffic"]
    gamma_df = data["gamma"]

    # ---------------------------
    # Infer core sets Z, R, K, T
    # ---------------------------
    z_col_s = _col(supply_df, "Zone", "zone")
    z_col_d = _col(demand_df, "Zone", "zone")

    # Types: try Type column, else assume 2nd column is type-like
    try:
        k_col_s = _col(supply_df, "Type", "type", "Item", "item", "Class", "class")
    except KeyError:
        k_col_s = supply_df.columns[1]
    try:
        k_col_d = _col(demand_df, "Type", "type", "Item", "item", "Class", "class")
    except KeyError:
        k_col_d = demand_df.columns[1]

    Z = sorted(pd.unique(pd.concat([supply_df[z_col_s], demand_df[z_col_d]]).dropna()))
    K = sorted(pd.unique(pd.concat([supply_df[k_col_s], demand_df[k_col_d]]).dropna()))

    # Refurb set from DistClean_ZR
    r_col = _col(dist_zr_df, "Refurb", "refurb", "R", "r", "Center", "center", "Facility", "facility")
    z_col_zr = _col(dist_zr_df, "Zone", "zone", z_col_s)
    R = sorted(pd.unique(dist_zr_df[r_col].dropna()))

    # Single period (your app/UI is single run anyway)
    T = [0]

    # ---------------------------
    # Gamma (weight per type)
    # ---------------------------
    # Expect columns like: Type, Gamma
    try:
        gk = _col(gamma_df, "Type", "type", k_col_s)
        gv = _col(gamma_df, "Gamma", "gamma", "Weight", "weight")
        gamma = {row[gk]: float(row[gv]) for _, row in gamma_df.iterrows()}
    except Exception:
        # fallback: all 1.0
        gamma = {k: 1.0 for k in K}

    # ---------------------------
    # Supply / Demand dicts
    # ---------------------------
    # Expect columns like: Zone, Type, Supply / Demand
    s_val = _col(supply_df, "Supply", "supply", "Qty", "qty", "Quantity", "quantity", "Amount", "amount")
    d_val = _col(demand_df, "Demand", "demand", "Qty", "qty", "Quantity", "quantity", "Amount", "amount")

    supply = {(row[z_col_s], row[k_col_s]): float(row[s_val]) for _, row in supply_df.iterrows()}
    demand = {(row[z_col_d], row[k_col_d]): float(row[d_val]) for _, row in demand_df.iterrows()}

    # ---------------------------
    # Costs: use distances as "cost proxies"
    # ---------------------------
    # DistClean_ZR: Zone, Refurb, Distance (or Cost)
    zr_val = None
    for cand in ["Cost", "cost", "Distance", "distance", "Miles", "miles"]:
        if cand in dist_zr_df.columns or cand.lower() in [c.lower() for c in dist_zr_df.columns]:
            zr_val = _col(dist_zr_df, cand)
            break
    if zr_val is None:
        zr_val = dist_zr_df.columns[-1]  # last column fallback

    C_ZR = _as_long_cost(dist_zr_df, z_col_zr, r_col, zr_val)

    # DistClean_RW: Refurb, Distance/Cost to Warehouse (single warehouse)
    rw_r = _col(dist_rw_df, "Refurb", "refurb", r_col)
    rw_val = None
    for cand in ["Cost", "cost", "Distance", "distance", "Miles", "miles"]:
        if cand.lower() in [c.lower() for c in dist_rw_df.columns]:
            rw_val = _col(dist_rw_df, cand)
            break
    if rw_val is None:
        rw_val = dist_rw_df.columns[-1]
    C_RW = {row[rw_r]: float(row[rw_val]) for _, row in dist_rw_df.iterrows()}

    # DistClean_WZ: Zone, Distance/Cost Warehouse->Zone
    wz_z = _col(dist_wz_df, "Zone", "zone", z_col_s)
    wz_val = None
    for cand in ["Cost", "cost", "Distance", "distance", "Miles", "miles"]:
        if cand.lower() in [c.lower() for c in dist_wz_df.columns]:
            wz_val = _col(dist_wz_df, cand)
            break
    if wz_val is None:
        wz_val = dist_wz_df.columns[-1]
    C_WZ = {row[wz_z]: float(row[wz_val]) for _, row in dist_wz_df.iterrows()}

    # ---------------------------
    # Lane capacities
    # ---------------------------
    # LaneCap_ZR: Zone, Refurb, Cap
    cap_z = _col(cap_zr_df, "Zone", "zone", z_col_s)
    cap_r = _col(cap_zr_df, "Refurb", "refurb", r_col)
    cap_v = _col(cap_zr_df, "Cap", "cap", "Capacity", "capacity", "LaneCap", "lanecap", "Limit", "limit")
    lane_cap_ZR = _as_long_cap(cap_zr_df, cap_z, cap_r, cap_v)

    # LaneCap_Wz: Zone, Cap
    capw_z = _col(cap_wz_df, "Zone", "zone", z_col_s)
    capw_v = _col(cap_wz_df, "Cap", "cap", "Capacity", "capacity", "LaneCap", "lanecap", "Limit", "limit")
    lane_cap_WZ = {row[capw_z]: float(row[capw_v]) for _, row in cap_wz_df.iterrows()}

    # ---------------------------
    # Build model
    # ---------------------------
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
            cost_mult_ZR * C_ZR[(z, r)] * gamma.get(k, 1.0) * x[(z, r, k, t)]
            for z in Z for r in R for k in K for t in T
            if (z, r) in C_ZR
        )
        + pulp.lpSum(
            cost_mult_RW * C_RW[r] * gamma.get(k, 1.0) * y[(r, k, t)]
            for r in R for k in K for t in T
            if r in C_RW
        )
        + pulp.lpSum(
            cost_mult_WZ * C_WZ[z] * gamma.get(k, 1.0) * w[(z, k, t)]
            for z in Z for k in K for t in T
            if z in C_WZ
        )
    )

    # Supply
    for z in Z:
        for k in K:
            s = supply.get((z, k), 0.0)
            for t in T:
                model += pulp.lpSum(x[(z, r, k, t)] for r in R) <= s

    # Refurb balance
    for r in R:
        for k in K:
            for t in T:
                model += pulp.lpSum(x[(z, r, k, t)] for z in Z) == y[(r, k, t)]

    # Warehouse balance
    for k in K:
        for t in T:
            model += pulp.lpSum(y[(r, k, t)] for r in R) == pulp.lpSum(w[(z, k, t)] for z in Z) + zDump[(k, t)]

    # Demand (>=)
    for z in Z:
        for k in K:
            d = demand.get((z, k), 0.0)
            for t in T:
                model += w[(z, k, t)] >= d

    # LaneCap ZR
    for z in Z:
        for r in R:
            cap = lane_cap_ZR.get((z, r), None)
            if cap is None:
                continue
            for t in T:
                model += pulp.lpSum(gamma.get(k, 1.0) * x[(z, r, k, t)] for k in K) <= cap_mult * cap

    # LaneCap WZ
    for z in Z:
        cap = lane_cap_WZ.get(z, None)
        if cap is None:
            continue
        for t in T:
            model += pulp.lpSum(gamma.get(k, 1.0) * w[(z, k, t)] for k in K) <= cap_mult * cap

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = model.solve(solver)

    status_str = pulp.LpStatus[status]
    obj_value = pulp.value(model.objective)

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

    return {
        "status": status_str,
        "objective": obj_value,
        "flows_ZR": flows_ZR,
        "flows_RW": flows_RW,
        "flows_WZ": flows_WZ,
    }
