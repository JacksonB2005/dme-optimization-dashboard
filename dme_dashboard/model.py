import pulp
import pandas as pd
from data_loader import load_data


def run_model(
    cost_mult_ZR=1.0,
    cost_mult_RW=1.0,
    cost_mult_WZ=1.0,
    cap_mult=1.0,
):
    """
    Build and solve the DME network LP with scenario multipliers.
    Returns a dict: status, objective, cost_breakdown, flows tables.
    """

    # ---- 1) Load data ----
    data = load_data()
    Z = data["Z"]
    K = data["K"]
    R = data["R"]
    T = data["T"]

    supply = data["supply"]
    demand = data["demand"]
    gamma = data["gamma"]

    C_ZR = data["C_ZR"]
    C_RW = data["C_RW"]
    C_WZ = data["C_WZ"]

    lane_cap_ZR = data["lane_cap_ZR"]
    lane_cap_WZ = data["lane_cap_WZ"]

    # ---- 2) Model ----
    model = pulp.LpProblem("DME_Recovery_Network", pulp.LpMinimize)

    # ---- 3) Variables ----
    x = {(z, r, k, t): pulp.LpVariable(f"x_{z}_{r}_{k}_{t}", lowBound=0)
         for z in Z for r in R for k in K for t in T}

    y = {(r, k, t): pulp.LpVariable(f"y_{r}_{k}_{t}", lowBound=0)
         for r in R for k in K for t in T}

    w = {(z, k, t): pulp.LpVariable(f"w_{z}_{k}_{t}", lowBound=0)
         for z in Z for k in K for t in T}

    zDump = {(k, t): pulp.LpVariable(f"zDump_{k}_{t}", lowBound=0)
             for k in K for t in T}

    # ---- 4) Objective ----
    model += (
        pulp.lpSum(
            cost_mult_ZR * C_ZR[(z, r)] * gamma[k] * x[(z, r, k, t)]
            for z in Z for r in R for k in K for t in T
        )
        + pulp.lpSum(
            cost_mult_RW * C_RW[r] * gamma[k] * y[(r, k, t)]
            for r in R for k in K for t in T
        )
        + pulp.lpSum(
            cost_mult_WZ * C_WZ[z] * gamma[k] * w[(z, k, t)]
            for z in Z for k in K for t in T
        ),
        "Total_Weekly_Transportation_Cost"
    )

    # ---- 5) Constraints ----

    # (1) Supply limit at each zone/type
    for z in Z:
        for k in K:
            for t in T:
                model += (
                    pulp.lpSum(x[(z, r, k, t)] for r in R) <= supply[(z, k)],
                    f"SupplyLimit_{z}_{k}_{t}",
                )

    # (2) Refurb flow balance: inflow = outflow
    for r in R:
        for k in K:
            for t in T:
                model += (
                    pulp.lpSum(x[(z, r, k, t)] for z in Z) == y[(r, k, t)],
                    f"RefurbBalance_{r}_{k}_{t}",
                )

    # (3) Warehouse balance: inflow = outflow + disposal
    for k in K:
        for t in T:
            model += (
                pulp.lpSum(y[(r, k, t)] for r in R)
                == pulp.lpSum(w[(z, k, t)] for z in Z) + zDump[(k, t)],
                f"WarehouseBalance_{k}_{t}",
            )

    # (4) Demand satisfaction
    for z in Z:
        for k in K:
            for t in T:
                model += (
                    w[(z, k, t)] >= demand[(z, k)],
                    f"Demand_{z}_{k}_{t}",
                )

    # (6) Lane capacity: Zone -> Refurb (weighted by gamma)
    for z in Z:
        for r in R:
            for t in T:
                model += (
                    pulp.lpSum(gamma[k] * x[(z, r, k, t)] for k in K)
                    <= cap_mult * lane_cap_ZR[(z, r)],
                    f"LaneCap_ZR_{z}_{r}_{t}",
                )

    # (7) Lane capacity: Warehouse -> Zone (weighted by gamma)
    for z in Z:
        for t in T:
            model += (
                pulp.lpSum(gamma[k] * w[(z, k, t)] for k in K)
                <= cap_mult * lane_cap_WZ[z],
                f"LaneCap_WZ_{z}_{t}",
            )

    # ---- 6) Solve ----
    solver = pulp.PULP_CBC_CMD(msg=False)
    result_status = model.solve(solver)

    status_str = pulp.LpStatus[result_status]
    obj_value = pulp.value(model.objective)

    # ---- 7) Cost breakdown ----
    cost_ZR = sum(
        cost_mult_ZR * C_ZR[(z, r)] * gamma[k] * pulp.value(x[(z, r, k, t)])
        for z in Z for r in R for k in K for t in T
    )
    cost_RW = sum(
        cost_mult_RW * C_RW[r] * gamma[k] * pulp.value(y[(r, k, t)])
        for r in R for k in K for t in T
    )
    cost_WZ = sum(
        cost_mult_WZ * C_WZ[z] * gamma[k] * pulp.value(w[(z, k, t)])
        for z in Z for k in K for t in T
    )

    cost_breakdown = {
        "Zone_to_Refurb": cost_ZR,
        "Refurb_to_Warehouse": cost_RW,
        "Warehouse_to_Zone": cost_WZ,
        "Total": cost_ZR + cost_RW + cost_WZ,
    }

    # ---- 8) Flows ----
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
        "cost_breakdown": cost_breakdown,
        "flows_ZR": flows_ZR,
        "flows_RW": flows_RW,
        "flows_WZ": flows_WZ,
    }
