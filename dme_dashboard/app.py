import streamlit as st
from model import run_model

# ----------------------------
# Page setup (MUST be first Streamlit call)
# ----------------------------
st.set_page_config(page_title="DME Optimization Dashboard", layout="wide")

st.title("DME Optimization Dashboard")
st.caption(
    "Standalone decision-support tool for the DME reverse logistics network "
    "(no code output shown)."
)

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Scenario inputs")

zr_mult = st.sidebar.slider(
    "Zone → Refurb cost multiplier",
    min_value=0.5, max_value=3.0, value=1.0, step=0.1
)

rw_mult = st.sidebar.slider(
    "Refurb → Warehouse cost multiplier",
    min_value=0.5, max_value=3.0, value=1.0, step=0.1
)

wz_mult = st.sidebar.slider(
    "Warehouse → Zone cost multiplier",
    min_value=0.5, max_value=3.0, value=1.0, step=0.1
)

cap_mult = st.sidebar.slider(
    "Lane capacity multiplier",
    min_value=0.5, max_value=2.0, value=1.0, step=0.1
)

run_button = st.sidebar.button("Run optimization", use_container_width=True)

# ----------------------------
# Session state
# ----------------------------
if "results" not in st.session_state:
    st.session_state["results"] = None
if "last_error" not in st.session_state:
    st.session_state["last_error"] = None

# ----------------------------
# Helpers
# ----------------------------
def _as_money(x):
    try:
        if x is None:
            return "—"
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)

def _get_first(res: dict, keys: list, default=None):
    for k in keys:
        if isinstance(res, dict) and k in res:
            return res[k]
    return default

def _safe_dataframe(res: dict, keys: list):
    df = _get_first(res, keys, default=None)
    if df is None:
        return None
    return df

# ----------------------------
# Run model
# ----------------------------
if run_button:
    st.session_state["last_error"] = None
    with st.spinner("Running optimization model..."):
        try:
            st.session_state["results"] = run_model(
                cost_mult_ZR=zr_mult,
                cost_mult_RW=rw_mult,
                cost_mult_WZ=wz_mult,
                cap_mult=cap_mult,
            )
            st.success("Optimization complete ✅")
        except Exception as e:
            st.session_state["results"] = None
            st.session_state["last_error"] = e

# ----------------------------
# Display results / errors
# ----------------------------
if st.session_state["last_error"] is not None:
    st.error("Model run failed.")
    st.exception(st.session_state["last_error"])
    st.stop()

if st.session_state["results"] is None:
    st.info("Adjust parameters in the sidebar and click **Run optimization**.")
    st.stop()

res = st.session_state["results"]

# If model accidentally returns something not a dict, show it and stop cleanly
if not isinstance(res, dict):
    st.warning("Model returned a non-dict result. Showing raw output:")
    st.write(res)
    st.stop()

# ---- pull common outputs with fallbacks ----
status = _get_first(res, ["status", "solver_status", "lp_status"], default=None)
objective = _get_first(res, ["objective", "obj", "objective_value", "total_cost"], default=None)

# cost breakdown might be named differently, or not provided
cb = _get_first(res, ["cost_breakdown", "costs", "breakdown"], default=None)

st.subheader("Key results")

cA, cB, cC = st.columns(3)
cA.metric("Status", status if status is not None else "—")
cB.metric("Optimal weekly cost", _as_money(objective))
cC.metric("ZR / RW / WZ multipliers", f"{zr_mult:.1f} / {rw_mult:.1f} / {wz_mult:.1f}")

st.divider()

# ---- Cost breakdown ----
st.subheader("Cost breakdown")

if isinstance(cb, dict):
    zrc = cb.get("Zone_to_Refurb", cb.get("ZR", cb.get("zone_refurb")))
    rwc = cb.get("Refurb_to_Warehouse", cb.get("RW", cb.get("refurb_warehouse")))
    wzc = cb.get("Warehouse_to_Zone", cb.get("WZ", cb.get("warehouse_zone")))
    tot = cb.get("Total", cb.get("total", None))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Zone → Refurb", _as_money(zrc))
    c2.metric("Refurb → Warehouse", _as_money(rwc))
    c3.metric("Warehouse → Zone", _as_money(wzc))
    c4.metric("Total", _as_money(tot if tot is not None else objective))
else:
    st.warning("No cost_breakdown dict returned by model. (App will still show flows.)")
    with st.expander("Debug: keys returned by model"):
        st.write(sorted(list(res.keys())))

st.divider()

# ---- Flows ----
flows_ZR = _safe_dataframe(res, ["flows_ZR", "flow_ZR", "zr_flows", "flows_zone_refurb"])
flows_RW = _safe_dataframe(res, ["flows_RW", "flow_RW", "rw_flows", "flows_refurb_warehouse"])
flows_WZ = _safe_dataframe(res, ["flows_WZ", "flow_WZ", "wz_flows", "flows_warehouse_zone"])

st.subheader("Flows")

tab1, tab2, tab3 = st.tabs(["Zone → Refurb", "Refurb → Warehouse", "Warehouse → Zones"])

with tab1:
    if flows_ZR is None:
        st.info("No Zone→Refurb flow table returned.")
    else:
        st.dataframe(flows_ZR, use_container_width=True)

with tab2:
    if flows_RW is None:
        st.info("No Refurb→Warehouse flow table returned.")
    else:
        st.dataframe(flows_RW, use_container_width=True)

with tab3:
    if flows_WZ is None:
        st.info("No Warehouse→Zone flow table returned.")
    else:
        st.dataframe(flows_WZ, use_container_width=True)

with st.expander("Debug: raw model output (dict)"):
    st.write(res)
