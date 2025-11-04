import streamlit as st
import numpy as np
import pandas as pd

# --------------------------
# ✅ Initialize Session State
# --------------------------
st.session_state.setdefault("reset", False)
st.session_state.setdefault("results", None)
st.session_state.setdefault("params", {})

# --------------------------
# ✅ App Title
# --------------------------
st.title("📡 CL-NetSim Pro — Cross-Layer Wireless Network Simulator")

st.markdown("""
Interactive simulator for **CDMA MAI, adaptive PHY/MAC, mobility & MANET routing**  
🚫 No datasets | 🚫 No ML models | ✅ Live simulation only
""")

# --------------------------
# ✅ Sidebar Controls
# --------------------------
with st.sidebar:
    st.header("Simulation Controls")

    num_users = st.slider("Number of CDMA Users", 1, 200, 20)
    snr_db = st.slider("SNR (dB)", -5, 30, 10)
    mobility_model = st.selectbox("Mobility Model", 
                                  ["None", "Random Waypoint", "Gauss-Markov"])
    routing = st.selectbox("MANET Routing Protocol", 
                           ["AODV", "DSR", "OLSR"])
    sim_time = st.number_input("Simulation Time (s)", 0.1, 100.0, 5.0)

    run_btn = st.button("▶️ Run Simulation")
    reset_btn = st.button("🔄 Reset")

# --------------------------
# ✅ Handle Reset Button
# --------------------------
if reset_btn:
    st.session_state["reset"] = True

if st.session_state["reset"]:
    st.session_state["results"] = None
    st.session_state["params"] = {}
    st.session_state["reset"] = False
    st.success("Simulation reset ✅")

# --------------------------
# ✅ Simulation Function
# --------------------------
def simulate_cdma(num_users, snr_db):
    # -------- Placeholder physics layer -------
    snr_linear = 10**(snr_db/10)
    mai = np.random.normal(0, 1/np.sqrt(num_users), num_users)  # MAI noise
    ber = 0.5 * np.exp(-snr_linear / (1 + np.var(mai)))  # Approx BER
    throughput = (1 - ber) * 1e6  # 1 Mbps baseline link
    
    return ber, throughput

# --------------------------
# ✅ Run Simulation
# --------------------------
if run_btn:
    ber, thr = simulate_cdma(num_users, snr_db)
    
    results = pd.DataFrame({
        "Metric": ["BER", "Throughput (bps)"],
        "Value": [ber, thr]
    })

    st.session_state["results"] = results
    st.session_state["params"] = {
        "Users": num_users,
        "SNR dB": snr_db,
        "Mobility": mobility_model,
        "Routing": routing,
        "Time": sim_time
    }

# --------------------------
# ✅ Display Output
# --------------------------
if st.session_state["results"] is not None:
    st.subheader("📊 Simulation Output")
    st.table(st.session_state["results"])
    
    with st.expander("📁 Simulation Parameters"):
        st.json(st.session_state["params"])
else:
    st.info("Click **Run Simulation** to begin ⏳")
