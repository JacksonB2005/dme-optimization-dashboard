from pathlib import Path
import pandas as pd

# -------------------------------------------------
# Resolve path to the Excel file (robust for cloud)
# -------------------------------------------------
HERE = Path(__file__).resolve().parent
EXCEL_FILE = HERE / "Operations Final X.xlsx"


def load_data(
    excel_file=EXCEL_FILE,
    stage_levels: dict | None = None,
    cap_multiplier: float = 1.0
):
    """
    Load cleaned Excel sheets and return all sets + parameters
    required by the optimization model.

    Works both locally and on Streamlit Cloud.
    """

    excel_file = Path(excel_file)

    # ---- Safety check ----
    if not excel_file.exists():
        available = [p.name for p in HERE.glob("*.xlsx")]
        raise FileNotFoundError(
            f"\nExcel file not found.\n"
            f"Expected: {excel_file}\n"
            f"Directory: {HERE}\n"
            f"Excel files found: {available}\n"
        )

    # -------------------------------------------------
    # Load sheets
    # -------------------------------------------------
    supply_df = pd.read_excel(excel_file, sheet_name="SupplyClean")
    demand_df = pd.read_excel(excel_file, sheet_name="DemandClean")
    cost_df = pd.read_excel(excel_file, sheet_name="CostsClean")
    capacity_df = pd.read_excel(excel_file, sheet_name="CapacitiesClean")

    # -------------------------------------------------
    # Normalize column names
    # -------------------------------------------------
    for df in [supply_df, demand_df, cost_df, capacity_df]:
        df.columns = [c.strip() for c in df.columns]

    # -------------------------------------------------
    # Build sets
    # -------------------------------------------------
    Z = supply_df["Zone"].unique().tolist()
    R = cost_df["Refurb"].unique().tolist()
    W = cost_df["Warehouse"].unique().tolist()
    K = capacity_df["Lane"].unique().tolist()

    # -------------------------------------------------
    # Parameters
    # -------------------------------------------------
    supply = dict(zip(supply_df["Zone"], supply_df["Supply"]))
    demand = dict(zip(demand_df["Zone"], demand_df["Demand"]))

    cost_ZR = {
        (row["Zone"], row["Refurb"]): row["Cost"]
        for _, row in cost_df[cost_df["Type"] == "ZR"].iterrows()
    }

    cost_RW = {
        (row["Refurb"], row["Warehouse"]): row["Cost"]
        for _, row in cost_df[cost_df["Type"] == "RW"].iterrows()
    }

    cost_WZ = {
        (row["Warehouse"], row["Zone"]): row["Cost"]
        for _, row in cost_df[cost_df["Type"] == "WZ"].iterrows()
    }

    capacity = {
        row["Lane"]: row["Capacity"] * cap_multiplier
        for _, row in capacity_df.iterrows()
    }

    # -------------------------------------------------
    # Return everything as a dict
    # -------------------------------------------------
    return {
        "Z": Z,
        "R": R,
        "W": W,
        "K": K,
        "supply": supply,
        "demand": demand,
        "cost_ZR": cost_ZR,
        "cost_RW": cost_RW,
        "cost_WZ": cost_WZ,
        "capacity": capacity,
    }
