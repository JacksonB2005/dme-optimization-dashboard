from __future__ import annotations

from pathlib import Path
import pandas as pd


def _find_excel_file(filename: str = "Operations Final X.xlsx") -> Path:
    """
    Look for the Excel file in common deploy locations:
    - same folder as this file (dme_dashboard/)
    - repo root
    - current working directory
    """
    app_dir = Path(__file__).resolve().parent          # .../dme_dashboard
    repo_dir = app_dir.parent                         # repo root
    candidates = [
        app_dir / filename,
        repo_dir / filename,
        Path.cwd() / filename,
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Excel file not found. Tried:\n" + "\n".join(f" - {c}" for c in candidates)
    )


def _sheet_map(excel_path: Path) -> dict[str, str]:
    """
    Build a case-insensitive map of sheet names -> actual sheet name.
    """
    xl = pd.ExcelFile(excel_path, engine="openpyxl")
    return {name.strip().lower(): name for name in xl.sheet_names}


def _read_sheet(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Read a sheet by exact name, with fallback to case-insensitive match.
    """
    smap = _sheet_map(excel_path)
    key = sheet_name.strip().lower()
    actual = smap.get(key)

    if actual is None:
        available = sorted(smap.values())
        raise ValueError(
            f"Worksheet named '{sheet_name}' not found.\n"
            f"Available sheets: {available}"
        )

    return pd.read_excel(excel_path, sheet_name=actual, engine="openpyxl")


def load_data(excel_file: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """
    Load cleaned Excel sheets exactly as they exist in the workbook.
    """
    excel_path = Path(excel_file) if excel_file is not None else _find_excel_file()

    supply_df = _read_sheet(excel_path, "SupplyClean")
    demand_df = _read_sheet(excel_path, "DemandClean")

    # Distance matrices
    dist_zr = _read_sheet(excel_path, "DistClean_ZR")
    dist_rw = _read_sheet(excel_path, "DistClean_RW")
    dist_wz = _read_sheet(excel_path, "DistClean_WZ")

    # Lane capacities (NOTE: your workbook tab shows LaneCap_Wz)
    cap_zr = _read_sheet(excel_path, "LaneCap_ZR")
    cap_wz = _read_sheet(excel_path, "LaneCap_Wz")

    # Other parameters
    traffic_df = _read_sheet(excel_path, "TrafficRatesClean")
    gamma_df = _read_sheet(excel_path, "GammaClean")

    return {
        "supply": supply_df,
        "demand": demand_df,
        "dist_ZR": dist_zr,
        "dist_RW": dist_rw,
        "dist_WZ": dist_wz,
        "cap_ZR": cap_zr,
        "cap_WZ": cap_wz,
        "traffic": traffic_df,
        "gamma": gamma_df,
    }
