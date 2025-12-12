import streamlit as st
from model import run_model

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="DME Optimization Dashboard", layout="wide")

st.title("DME Optimization Dashboard")
st.caption(
    "Standalone decision-support tool for the DME reverse logistics network "
    "(no code output shown)."
)

# ----------------------------
# Sidebar controls (USER INPUTS)
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

run_button = st.sidebar.button("Run optimization")

# ----------------------------
# Session state
# ----------------------------
if "results" not in st.session_state:
    st.session_state["results"] = None

# ----------------------------
# Run model
# ----------------------------
if run_button:
    with st.spinner("Running optimization model..."):
        st.session_state["results"] = run_model(
            cost_mult_ZR=zr_mult,
            cost_mult_RW=rw_mult,
            cost_mult_WZ=wz_mult,
            cap_mult=cap_mult,
        )
    st.success("Optimization complete ✅")

# ----------------------------
# Display results
# ----------------------------
if st.session_state["results"] is None:
    st.info("Adjust parameters in the sidebar and click **Run optimization**.")
else:
    res = st.session_state["results"]
    cb = res["cost_breakdown"]

    st.subheader("Key results")

    st.metric("Optimal weekly cost", f"${res['objective']:.2f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Zone → Refurb", f"${cb['Zone_to_Refurb']:.2f}")
    c2.metric("Refurb → Warehouse", f"${cb['Refurb_to_Warehouse']:.2f}")
    c3.metric("Warehouse → Zone", f"${cb['Warehouse_to_Zone']:.2f}")
    c4.metric("Total", f"${cb['Total']:.2f}")

    st.divider()

    st.subheader("Flows: Zone → Refurb")
    st.dataframe(res["flows_ZR"], use_container_width=True)

    st.subheader("Flows: Refurb → Warehouse")
    st.dataframe(res["flows_RW"], use_container_width=True)

    st.subheader("Flows: Warehouse → Zones")
    st.dataframe(res["flows_WZ"], use_container_width=True)
