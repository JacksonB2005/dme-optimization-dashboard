import pulp
import pandas as pd
from data_loader import load_data


def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in str(s).strip() if ch.isalnum())


def _pick_col(df: pd.DataFrame, candidates):
    """
    Try to find a column by name (flexible). If not found, return None.
    """
    cols = list(df.columns)
    norm_map = {_norm(c): c for c in cols}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def _as_3col(df: pd.DataFrame):
    """
    Return (colA, colB, colV) using best-effort detection,
    else fallback to first/second/third column.
    """
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns, got {df.shape[1]} in {list(df.columns)}")

    # try common patterns
    a = _pick_col(df, ["zone", "from", "origin", "i", "start"])
    b = _pick_col(df, ["refurb", "warehouse", "to", "dest", "destination", "j", "end"])
    v = _pick_col(df, ["dist", "distance", "miles", "cost", "value", "rate", "cap", "capacity"])

    cols = list(df.columns)

    if a is None:
        a = cols[0]
    if b is None:
        b = cols[1]
    if v is None:
        v = cols[2] if len(cols) >= 3 else None

    return a, b, v


def _supply_demand_to_long(df: pd.DataFrame, value_name: str):
    """
    Accepts either:
    - long format: Zone, Type, Supply/Demand
    - wide format: Zone then types as columns
    Returns a dataframe with columns: Zone, Type, Value
    """
    zone_col = _pick_col(df, ["zone", "z"])
    type_col = _pick_col(df, ["type", "k", "item", "category"])
    val_col  = _pick_col(df, [value_name, "value", "qty", "quantity", "amount"])

    if zone_col and type_col and val_col:
        out = df[[zone_col, type_col, val_col]].copy()
        out.columns = ["Zone", "Type", "Value"]
        return out

    # wide fallback: assume first col is Zone, remaining are types
    cols = list(df.columns)
    zone_col = cols[0]
    melted = df.melt(id_vars=[zone_col], var_name="Type", value_name="Value")
    melted.rename(columns={zone_col: "Zone"}, inplace=True)
    return melted


def _gamma_to_dict(df: pd.DataFrame, K):
    # Try (Type, Gamma) long format
    type_col = _pick_col(df, ["type", "k", "item"])
    gam_col  = _pick_col(df, ["gamma", "weight", "multiplier", "factor", "value"])
    if type_col and gam_col:
        d = {}
        for _, r in df[[type_col, gam_col]].dropna().iterrows():
            d[str(r[type_col])] = float(r[gam_col])
        # fill missing with 1.0
        return {k: float(d.get(k, 1.0)) for k in K}
    # fallback: all 1
    return {k: 1.0 for k in K}


def _traffic_to_dict(df: pd.DataFrame):
    """
    If traffic sheet has a time column, use it.
    Otherwise return one period with multiplier 1.0 (or the single value if present).
    """
    t_col = _pick_col(df, ["t", "time", "period", "week"])
    v_col = _pick_col(df, ["traffic", "rate", "multiplier", "factor", "value"])

    if t_col and v_col:
        out = {}
        for _, r in df[[t_col, v_col]].dropna().iterrows():
            out[str(r[t_col])] = float(r[v_col])
        return out

    # if single numeric value exists anywhere, use it
    for c in df.columns:
        try:
            val = float(df[c].dropna().iloc[0])
            return {"T1": val}
        except Exception:
            continue

    return {"T1": 1.0}


def run_model(cost_mult_ZR=1.0, cost_mult_RW=1.0, cost_mult_WZ=1.0, cap_mult=1.0):
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

    supply_long = _supply_demand_to_long(supply_df, "supply")
    demand_long = _supply_demand_to_long(demand_df, "demand")

    # sets
    Z = sorted(set(supply_long["Zone"].astype(str)) | set(demand_long["Zone"].astype(str)))
    K = sorted(set(supply_long["Type"].astype(str)) | set(demand_long["Type"].astype(str)))
    traffic = _traffic_to_dict(traffic_df)
    T = sorted(traffic.keys())

    gamma = _gamma_to_dict(gamma_df, K)

    # supply/demand dicts
    supply = {(z, k): 0.0 for z in Z for k in K}
    for _, r in supply_long.dropna().iterrows():
        z = str(r["Zone"]); k = str(r["Type"])
        try:
            supply[(z, k)] = float(r["Value"])
        except Exception:
            pass

    demand = {(z, k): 0.0 for z in Z for k in K}
    for _, r in demand_long.dropna().iterrows():
        z = str(r["Zone"]); k = str(r["Type"])
        try:
            demand[(z, k)] = float(r["Value"])
        except Exception:
            pass

    # costs / distances
    a, b, v = _as_3col(dist_zr_df)
    if v is None:
        raise ValueError(f"DistClean_ZR needs a value column. Columns={list(dist_zr_df.columns)}")
    C_ZR = {(str(r[a]), str(r[b])): float(r[v]) for _, r in dist_zr_df[[a, b, v]].dropna().iterrows()}

    a, b, v = _as_3col(dist_rw_df)
    if v is None:
        raise ValueError(f"DistClean_RW needs a value column. Columns={list(dist_rw_df.columns)}")
    # RW might be keyed by refurb only; handle both 2-col and 3-col cases
    if dist_rw_df.shape[1] >= 3:
        C_RW = {str(r[a]): float(r[v]) for _, r in dist_rw_df[[a, v]].dropna().iterrows()}
    else:
        # fallback: first col refurb, second col value
        cols = list(dist_rw_df.columns)
        C_RW = {str(r[cols[0]]): float(r[cols[1]]) for _, r in dist_rw_df[[cols[0], cols[1]]].dropna().iterrows()}

    a, b, v = _as_3col(dist_wz_df)
    if v is None:
        raise ValueError(f"DistClean_WZ needs a value column. Columns={list(dist_wz_df.columns)}")
    # WZ might be keyed by zone only
    if dist_wz_df.shape[1] >= 3:
        C_WZ = {str(r[b]): float(r[v]) for _, r in dist_wz_df[[b, v]].dropna().iterrows()}
    else:
        cols = list(dist_wz_df.columns)
        C_WZ = {str(r[cols[0]]): float(r[cols[1]]) for _, r in dist_wz_df[[cols[0], cols[1]]].dropna().iterrows()}

    # lane capacities
    a, b, v = _as_3col(cap_zr_df)
    if v is None:
        # if only two cols, treat as (a,b)=(zone,refurb) and v=second?? but cap must exist
        cols = list(cap_zr_df.columns)
        if len(cols) >= 3:
            v = cols[2]
        else:
            raise ValueError(f"LaneCap_ZR needs a capacity column. Columns={list(cap_zr_df.columns)}")
    lane_cap_ZR = {(str(r[a]), str(r[b])): float(r[v]) for _, r in cap_zr_df[[a, b, v]].dropna().iterrows()}

    # WZ cap: assume (zone, cap) even if sheet has extra cols
    cols = list(cap_wz_df.columns)
    zc = _pick_col(cap_wz_df, ["zone", "z"]) or cols[0]
    vc = _pick_col(cap_wz_df, ["cap", "capacity", "value"]) or (cols[1] if len(cols) > 1 else cols[0])
    lane_cap_WZ = {str(r[zc]): float(r[vc]) for _, r in cap_wz_df[[zc, vc]].dropna().iterrows()}

    # refurb set R from ZR costs/caps
    R = sorted({r for (_, r) in lane_cap_ZR.keys()} | {r for (_, r) in C_ZR.keys()})

    # ----------------- MODEL -----------------
    m = pulp.LpProblem("DME_Recovery_Network", pulp.LpMinimize)

    x = {(z, r, k, t): pulp.LpVariable(f"x_{z}_{r}_{k}_{t}", lowBound=0)
         for z in Z for r in R for k in K for t in T}
    y = {(r, k, t): pulp.LpVariable(f"y_{r}_{k}_{t}", lowBound=0)
         for r in R for k in K for t in T}
    w = {(z, k, t): pulp.LpVariable(f"w_{z}_{k}_{t}", lowBound=0)
         for z in Z for k in K for t in T}
    zDump = {(k, t): pulp.LpVariable(f"zDump_{k}_{t}", lowBound=0)
             for k in K for t in T}

    # objective (use traffic multiplier per period)
    m += (
        pulp.lpSum(
            cost_mult_ZR * traffic[t] * C_ZR.get((z, r), 0.0) * gamma[k] * x[(z, r, k, t)]
            for z in Z for r in R for k in K for t in T
        )
        + pulp.lpSum(
            cost_mult_RW * traffic[t] * C_RW.get(r, 0.0) * gamma[k] * y[(r, k, t)]
            for r in R for k in K for t in T
        )
        + pulp.lpSum(
            cost_mult_WZ * traffic[t] * C_WZ.get(z, 0.0) * gamma[k] * w[(z, k, t)]
            for z in Z for k in K for t in T
        )
    )

    # supply
    for z in Z:
        for k in K:
            for t in T:
                m += pulp.lpSum(x[(z, r, k, t)] for r in R) <= supply[(z, k)]

    # refurb balance
    for r in R:
        for k in K:
            for t in T:
                m += pulp.lpSum(x[(z, r, k, t)] for z in Z) == y[(r, k, t)]

    # warehouse balance
    for k in K:
        for t in T:
            m += pulp.lpSum(y[(r, k, t)] for r in R) == pulp.lpSum(w[(z, k, t)] for z in Z) + zDump[(k, t)]

    # demand
    for z in Z:
        for k in K:
            for t in T:
                m += w[(z, k, t)] >= demand[(z, k)]

    # lane cap ZR
    for z in Z:
        for r in R:
            cap = lane_cap_ZR.get((z, r), None)
            if cap is None:
                continue
            for t in T:
                m += pulp.lpSum(gamma[k] * x[(z, r, k, t)] for k in K) <= cap_mult * cap

    # lane cap WZ
    for z in Z:
        cap = lane_cap_WZ.get(z, None)
        if cap is None:
            continue
        for t in T:
            m += pulp.lpSum(gamma[k] * w[(z, k, t)] for k in K) <= cap_mult * cap

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = m.solve(solver)

    status_str = pulp.LpStatus[status]
    obj_value = float(pulp.value(m.objective)) if pulp.value(m.objective) is not None else None

    # basic outputs (period 1)
    t0 = T[0]

    flows_ZR = pd.DataFrame([
        {"Zone": z, "Refurb": r, "Type": k, "Flow_to_Refurb": float(pulp.value(x[(z, r, k, t0)]))}
        for z in Z for r in R for k in K
        if pulp.value(x[(z, r, k, t0)]) and pulp.value(x[(z, r, k, t0)]) > 1e-6
    ])

    flows_RW = pd.DataFrame([
        {"Refurb": r, "Type": k, "Flow_to_Warehouse": float(pulp.value(y[(r, k, t0)]))}
        for r in R for k in K
        if pulp.value(y[(r, k, t0)]) and pulp.value(y[(r, k, t0)]) > 1e-6
    ])

    flows_WZ = pd.DataFrame([
        {"Zone": z, "Type": k, "Flow_to_Zone": float(pulp.value(w[(z, k, t0)]))}
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
