# app.py
"""
CL-NetSim Pro — Full-featured single-file Streamlit simulator
Features:
 - CDMA MAI (multi-access interference)
 - Adaptive modulation + BER estimates
 - Simple 802.11-like MAC contention model
 - Random Waypoint mobility
 - AODV-like routing (energy-aware shortest-path)
 - Live charts, CSV export
 - Safe Streamlit session_state initialization & reset
No datasets, no ML models. Self-contained.
"""
import streamlit as st
import numpy as np
import pandas as pd
import math
import time
import random
import io
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

# -------------------------------
# Session state safe initialization
# -------------------------------
st.set_page_config(layout="wide", page_title="CL-NetSim Pro")
st.session_state.setdefault("initialized", False)
st.session_state.setdefault("nodes", [])
st.session_state.setdefault("history", {"time": [], "avg_throughput": [], "avg_delay": [], "avg_battery": [], "packet_drops": []})
st.session_state.setdefault("time_step", 0)
st.session_state.setdefault("running", False)
st.session_state.setdefault("params", {})

# -------------------------------
# Small utility functions
# -------------------------------
def modulation_bits(mod):
    return {"BPSK":1,"QPSK":2,"16-QAM":4,"64-QAM":6}.get(mod,1)

def select_modulation(snr_db):
    if snr_db < 6:
        return "BPSK"
    elif snr_db < 12:
        return "QPSK"
    elif snr_db < 18:
        return "16-QAM"
    else:
        return "64-QAM"

def ber_formula(mod, snr_db):
    # Use approximate closed-form style expressions (not exact but illustrative)
    snr = 10**(snr_db/10)
    if mod == "BPSK":
        # BER approx for BPSK over AWGN
        return 0.5*math.erfc(math.sqrt(snr))/1.0
    elif mod == "QPSK":
        return 0.5*math.erfc(math.sqrt(snr/1.0))/1.0
    elif mod == "16-QAM":
        # Approx for square M-QAM
        return 3/8 * math.erfc(math.sqrt(snr/10)) 
    else: # 64-QAM
        return 7/24 * math.erfc(math.sqrt(snr/21))

def path_loss(dist, freq_factor=1.0):
    # Simplified distance-based loss (normalized)
    if dist <= 0:
        return 1.0
    # Free-space style with soft floor
    return 1.0 / (1.0 + (dist/50.0)**2 * freq_factor)

def compute_cdma_mai(idx, nodes, activity_prob, cross_corr):
    # Sum of interfering received power (from other simultaneously active users)
    target = nodes[idx]
    total = 0.0
    for j, other in enumerate(nodes):
        if j == idx: continue
        if other['active'] and random.random() < activity_prob:
            d = math.dist(target['pos'], other['pos'])
            p_rx = other['tx_power'] * path_loss(d)
            total += p_rx * cross_corr
    return total

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

# -------------------------------
# App UI: Controls
# -------------------------------
st.title("📡 CL-NetSim Pro — Cross-Layer Wireless Network Simulator (Full)")
st.markdown("Interactive demonstration of cross-layer design: CDMA MAI, adaptive PHY, MAC contention, mobility and AODV-like routing. No datasets or ML models.")

with st.sidebar:
    st.header("Simulation Parameters")
    NUM_NODES = st.slider("Number of nodes", 3, 20, 8)
    AREA_SIZE = st.slider("Area size (square units)", 100, 500, 300)
    COMM_RANGE = st.slider("Communication range (units)", 30, 200, 120)
    SIM_STEP_MS = st.slider("Step interval (ms)", 50, 2000, 300)
    MAX_STEPS = st.number_input("Max steps in Run", min_value=10, max_value=2000, value=400)
    NODE_SPEED = st.slider("Max node speed (units/step)", 0.0, 6.0, 1.5)
    ACTIVITY_PROB = st.slider("User activity probability", 0.0, 1.0, 0.65)
    CROSS_CORR = st.slider("CDMA cross-correlation factor", 0.0, 0.5, 0.12)
    DROP_POLICY = st.selectbox("Queue drop policy", ["Tail Drop", "Early Drop (prob)", "Priority Drop"])
    mobility_model = st.selectbox("Mobility model", ["None", "Random Waypoint", "Gauss-Markov"])
    routing_algo = st.selectbox("Routing metric (AODV-like)", ["Hop-count", "Energy-aware"])
    enable_recharge = st.checkbox("Ambient recharge (very slow)", value=False)

    st.markdown("---")
    st.write("MAC Parameters (simplified 802.11-like)")
    CW_MIN = st.number_input("CWmin (contention window min)", min_value=2, max_value=256, value=8)
    PAYLOAD_BITS = st.selectbox("Payload size (abstract units)", [1,2,4,8], index=1)

    st.markdown("---")
    run_mode = st.radio("Execution mode", ["Step", "Run"], index=0)
    if st.button("Reset Simulation"):
        st.session_state['initialized'] = False
        st.session_state['nodes'] = []
        st.session_state['history'] = {"time": [], "avg_throughput": [], "avg_delay": [], "avg_battery": [], "packet_drops": []}
        st.session_state['time_step'] = 0
        st.session_state['running'] = False
        st.success("Simulation reset — reinitialize to start fresh.")

# -------------------------------
# Initialize nodes (only once unless reset)
# -------------------------------
if not st.session_state['initialized']:
    st.session_state['nodes'] = []
    st.session_state['history'] = {"time": [], "avg_throughput": [], "avg_delay": [], "avg_battery": [], "packet_drops": []}
    st.session_state['time_step'] = 0

    for i in range(NUM_NODES):
        node = {
            "id": i,
            "pos": [random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE)],
            "vel": [random.uniform(-NODE_SPEED, NODE_SPEED), random.uniform(-NODE_SPEED, NODE_SPEED)],
            "battery": random.uniform(0.5, 1.0),
            "tx_power": random.uniform(0.4, 1.0),
            "queue": deque(),
            "stats": {"sent":0, "dropped":0, "delay_sum":0, "throughput":0.0},
            "active": False,   # transmitting this step?
            "priority": random.choice([0,1]), # 1 high priority
            "backoff": 0,
            "cw": CW_MIN
        }
        st.session_state['nodes'].append(node)
    st.session_state['initialized'] = True

nodes = st.session_state['nodes']

# -------------------------------
# Mobility models
# -------------------------------
def random_waypoint_update(node):
    # If velocity nearly zero, pick new waypoint
    if math.hypot(node['vel'][0], node['vel'][1]) < 0.01 or random.random() < 0.02:
        dst = [random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE)]
        dx = dst[0] - node['pos'][0]; dy = dst[1] - node['pos'][1]
        dist = math.hypot(dx, dy) + 1e-9
        speed = random.uniform(0.2, NODE_SPEED) if NODE_SPEED>0 else 0.0
        node['vel'][0] = (dx/dist) * speed
        node['vel'][1] = (dy/dist) * speed
    node['pos'][0] = (node['pos'][0] + node['vel'][0]) % AREA_SIZE
    node['pos'][1] = (node['pos'][1] + node['vel'][1]) % AREA_SIZE

def gauss_markov_update(node, alpha=0.85):
    # simple gauss-markov drift
    mu = 0
    sigma = NODE_SPEED/2 if NODE_SPEED>0 else 0.1
    node['vel'][0] = alpha*node['vel'][0] + (1-alpha)*random.gauss(mu, sigma)
    node['vel'][1] = alpha*node['vel'][1] + (1-alpha)*random.gauss(mu, sigma)
    node['vel'][0] = max(-NODE_SPEED, min(NODE_SPEED, node['vel'][0]))
    node['vel'][1] = max(-NODE_SPEED, min(NODE_SPEED, node['vel'][1]))
    node['pos'][0] = (node['pos'][0] + node['vel'][0]) % AREA_SIZE
    node['pos'][1] = (node['pos'][1] + node['vel'][1]) % AREA_SIZE

# -------------------------------
# Simplified MAC behavior (CSMA/CA-like)
# -------------------------------
def mac_access_step(nodes):
    # Nodes with non-empty queue attempt to access channel based on backoff
    contenders = []
    for n in nodes:
        if len(n['queue']) > 0:
            if n['backoff'] <= 0:
                contenders.append(n)
            else:
                n['backoff'] -= 1
    # If multiple contenders, collisions can occur
    if len(contenders) == 0:
        return []
    if len(contenders) == 1:
        node = contenders[0]
        node['active'] = True
        # reset cw after success
        node['cw'] = CW_MIN
        return [node]
    # multiple contenders -> collision with some probability based on CW
    # We'll simulate each contender picking a random slot in [0, cw) and winner is min slot.
    slots = {}
    for n in contenders:
        slot = random.randint(0, max(1, n['cw']-1))
        slots.setdefault(slot, []).append(n)
    min_slot = min(slots.keys())
    winners = slots[min_slot]
    if len(winners) == 1:
        winner = winners[0]
        winner['active'] = True
        # all others increase CW (binary exponential backoff)
        for n in contenders:
            if n is not winner:
                n['cw'] = min(512, n['cw'] * 2)
                n['backoff'] = random.randint(0, n['cw'])
        # success for winner
        winner['cw'] = CW_MIN
        return [winner]
    else:
        # collision between those in winners -> collision causes drops or retries
        for n in winners:
            # collision -> increase cw and set backoff
            n['cw'] = min(512, n['cw'] * 2)
            n['backoff'] = random.randint(0, n['cw'])
            # count as collision (drop or retransmit strategy)
            # We'll simulate that one packet experiences error and will be retransmitted (count as dropped)
            if n['queue']:
                n['queue'].popleft()
                n['stats']['dropped'] += 1
        # non-winner contenders also pick backoff
        for n in contenders:
            if n not in winners:
                n['backoff'] = random.randint(0, n['cw'])
        return []

# -------------------------------
# Routing: Build graph & compute AODV-like route
# -------------------------------
def build_connectivity_graph(nodes):
    G = nx.Graph()
    for n in nodes:
        G.add_node(n['id'], battery=n['battery'])
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            d = math.dist(nodes[i]['pos'], nodes[j]['pos'])
            if d <= COMM_RANGE:
                # weight for hop = 1; energy-aware weight = energy factor
                G.add_edge(i, j, weight=1.0)
    return G

def choose_route(src, dst, nodes, metric="Hop-count"):
    G = build_connectivity_graph(nodes)
    try:
        if metric == "Hop-count":
            path = nx.shortest_path(G, source=src, target=dst, weight='weight')
        else:
            # Energy-aware: weight edges by inverse avg battery of nodes -> prefer high energy nodes
            H = G.copy()
            for u,v,data in list(H.edges(data=True)):
                batt_avg = (nodes[u]['battery'] + nodes[v]['battery']) / 2.0
                # lower weight when batt high
                H[u][v]['weight'] = 1.0 / (0.01 + batt_avg)
            path = nx.shortest_path(H, source=src, target=dst, weight='weight')
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        path = None
    return path

# -------------------------------
# Main simulation step
# -------------------------------
def simulation_step():
    t = st.session_state['time_step'] + 1
    total_throughput = 0.0
    total_delay = 0.0
    total_battery = 0.0
    total_drops = 0

    # 1) New arrivals (bursty)
    for n in nodes:
        if random.random() < ACTIVITY_PROB:
            burst = np.random.poisson(1 + 2*random.random())
            for _ in range(burst):
                pkt = {'arrival': t, 'size': PAYLOAD_BITS, 'priority': n['priority']}
                # Admission control: simple queue limit proportional to battery
                qlimit = int(20 * (0.5 + n['battery']))  # nodes with more battery handle larger queues
                if len(n['queue']) < qlimit:
                    n['queue'].append(pkt)
                else:
                    # drop policy
                    if DROP_POLICY == "Tail Drop":
                        n['stats']['dropped'] += 1
                        total_drops += 1
                    elif DROP_POLICY == "Early Drop (prob)":
                        if random.random() < 0.5:
                            n['stats']['dropped'] += 1
                            total_drops += 1
                        else:
                            if n['queue']:
                                n['queue'].popleft()
                                n['queue'].append(pkt)
                    else:  # priority drop
                        removed = False
                        for idx,q in enumerate(n['queue']):
                            if q['priority'] == 0:
                                del n['queue'][idx]
                                n['queue'].append(pkt)
                                removed = True
                                break
                        if not removed:
                            n['stats']['dropped'] += 1
                            total_drops += 1

    # 2) Mobility update
    for n in nodes:
        if mobility_model == "Random Waypoint":
            random_waypoint_update(n)
        elif mobility_model == "Gauss-Markov":
            gauss_markov_update(n)
        else:
            # small random drift if None
            n['pos'][0] = (n['pos'][0] + n['vel'][0]) % AREA_SIZE
            n['pos'][1] = (n['pos'][1] + n['vel'][1]) % AREA_SIZE

    # 3) MAC access (who transmits)
    # Clear previous active flags
    for n in nodes:
        n['active'] = False
    transmitters = mac_access_step(nodes)

    # 4) For each node, compute SNR to a sink at center OR do multi-hop via routing
    sink = [AREA_SIZE/2.0, AREA_SIZE/2.0]
    for i,n in enumerate(nodes):
        # compute received power at sink
        d_sink = math.dist(n['pos'], sink)
        rx_sink = n['tx_power'] * path_loss(d_sink)
        mai = compute_cdma_mai(i, nodes, ACTIVITY_PROB, CROSS_CORR)
        noise = 0.01
        snr_linear = rx_sink / (noise + mai)
        snr_db = 10 * math.log10(max(1e-9, snr_linear))
        n['last_snr_db'] = snr_db
        n['modulation'] = select_modulation(snr_db)
        n['ber'] = clamp(ber_formula(n['modulation'], snr_db), 0.0, 1.0)

    # 5) Transmitters send one packet each (if queue non-empty)
    for tx in transmitters:
        if len(tx['queue']) == 0:
            tx['active'] = False
            continue
        pkt = tx['queue'].popleft()
        # decide route to sink (AODV-like) - using node ids
        src = tx['id']
        # pick the nearest node to sink as destination (or sink as virtual destination)
        # For demo we treat sink as special: if node within range to sink, direct transmission else try multi-hop to a neighbor closer to sink
        d_to_sink = math.dist(tx['pos'], sink)
        if d_to_sink <= COMM_RANGE:
            path = [src]  # direct
        else:
            # pick neighbor with path to sink using routing
            # create a virtual destination node id = -1 (sink) -> we just attempt to route via nodes closer to sink
            # we approximate by finding neighbor nearest to sink and hop to it
            neighbors = []
            for j, other in enumerate(nodes):
                if j == src: continue
                if math.dist(tx['pos'], other['pos']) <= COMM_RANGE:
                    neighbors.append((j, math.dist(other['pos'], sink)))
            if not neighbors:
                path = None
            else:
                # choose neighbor according to routing metric
                if routing_algo == "Hop-count":
                    # choose neighbor with min distance to sink (greedy)
                    neighbors.sort(key=lambda x: x[1])
                    path = [src, neighbors[0][0]]
                else:
                    # energy-aware: prefer neighbor with high battery and nearer to sink
                    neighbors.sort(key=lambda x: (math.dist(nodes[x[0]]['pos'], sink)/ (nodes[x[0]]['battery']+1e-6)))
                    path = [src, neighbors[0][0]]

        # transmission success depends on BER and MAI along chosen path (we simulate direct success probability)
        snr_db = tx.get('last_snr_db', 0.0)
        ber = tx.get('ber', 0.5)
        success_prob = 1.0 - ber
        if random.random() < success_prob:
            # success: count throughput
            bits = modulation_bits(tx['modulation']) * pkt['size']
            tx['stats']['sent'] += 1
            tx['stats']['throughput'] += bits
            tx['stats']['delay_sum'] += (st.session_state['time_step'] + 1 - pkt['arrival'])
            total_throughput += bits
        else:
            # failed -> retransmit attempt later or drop (we'll count as dropped)
            tx['stats']['dropped'] += 1
            total_drops += 1

        # energy consumed proportional to tx_power and bits
        tx['battery'] -= 0.001 * tx['tx_power'] * (1 + bits/8.0)
        tx['battery'] = max(0.0, tx['battery'])

    # 6) Passive battery drain + recharge
    for n in nodes:
        # idle drain
        n['battery'] -= 0.00005
        if enable_recharge:
            n['battery'] = min(1.0, n['battery'] + 0.0002)
        n['battery'] = clamp(n['battery'])

        total_battery += n['battery']

    # 7) Stats aggregation
    avg_throughput = total_throughput / max(1, NUM_NODES)
    avg_delay = sum(n['stats']['delay_sum'] for n in nodes) / max(1, sum(max(1, n['stats']['sent']) for n in nodes))
    avg_batt = total_battery / NUM_NODES

    st.session_state['time_step'] = t
    st.session_state['history']['time'].append(t)
    st.session_state['history']['avg_throughput'].append(avg_throughput)
    st.session_state['history']['avg_delay'].append(avg_delay if not math.isnan(avg_delay) else 0.0)
    st.session_state['history']['avg_battery'].append(avg_batt)
    st.session_state['history']['packet_drops'].append(total_drops)

# -------------------------------
# Controls area
# -------------------------------
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Controls")
    if st.button("Step Simulation"):
        simulation_step()
    run_click = st.button("Run Simulation")
    if run_click and run_mode == "Run":
        st.session_state['running'] = True
        steps = 0
        progress = st.progress(0)
        while steps < MAX_STEPS and st.session_state['running']:
            simulation_step()
            steps += 1
            progress.progress(int(steps*100/MAX_STEPS))
            time.sleep(max(0.01, SIM_STEP_MS/1000.0))
        st.session_state['running'] = False
        st.success(f"Finished run: {steps} steps")
    if st.button("Stop Run"):
        st.session_state['running'] = False
    st.markdown("---")
    st.subheader("Per-node quick view")
    for n in nodes:
        with st.expander(f"Node {n['id']} | Batt: {n['battery']:.2f} | Queue: {len(n['queue'])}"):
            st.write(f"Pos: ({n['pos'][0]:.1f}, {n['pos'][1]:.1f})  Vel: ({n['vel'][0]:.2f},{n['vel'][1]:.2f})")
            st.write(f"Tx power: {n['tx_power']:.2f}  CW: {n['cw']}  Backoff: {n['backoff']}")
            st.write(f"Sent: {n['stats']['sent']}  Dropped: {n['stats']['dropped']}  Throughput (accum): {n['stats']['throughput']:.2f}")
            st.write(f"Last SNR (dB): {n.get('last_snr_db', 0.0):.2f}  Mod: {n.get('modulation','-')}  BER: {n.get('ber',0.0):.3f}")

with col2:
    st.subheader("Live Network Map & Metrics")
    # Plot nodes scatter
    df = pd.DataFrame([{"id": n['id'], "x": n['pos'][0], "y": n['pos'][1], "battery": n['battery']} for n in nodes])
    plt.figure(figsize=(6,6))
    plt.scatter(df['x'], df['y'], s=200*df['battery']+20)
    for _, r in df.iterrows():
        plt.text(r['x']+2, r['y']+2, f"#{int(r['id'])}")
    # plot sink
    plt.scatter([AREA_SIZE/2], [AREA_SIZE/2], marker='*', s=200)
    plt.xlim(0, AREA_SIZE); plt.ylim(0, AREA_SIZE)
    plt.title("Node positions (size ∝ battery) — sink (*)")
    st.pyplot(plt)

    st.subheader("Performance Trends")
    hist = st.session_state['history']
    if len(hist['time']) > 0:
        perf_df = pd.DataFrame({
            "time": hist['time'],
            "throughput": hist['avg_throughput'],
            "delay": hist['avg_delay'],
            "battery": hist['avg_battery'],
            "drops": hist['packet_drops']
        }).set_index("time")
        st.line_chart(perf_df[["throughput", "battery"]])
        st.line_chart(perf_df[["delay", "drops"]])
    else:
        st.info("Run the simulation to populate performance charts.")

# -------------------------------
# BER vs SNR interactive plot
# -------------------------------
st.markdown("---")
st.subheader("BER vs SNR (modulation curves)")
snr_vals = np.linspace(-5, 30, 200)
fig, ax = plt.subplots(figsize=(6,3.5))
for mod in ["BPSK", "QPSK", "16-QAM", "64-QAM"]:
    bers = [clamp(ber_formula(mod, s), 1e-9, 1.0) for s in snr_vals]
    ax.semilogy(snr_vals, bers)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("BER (log scale)")
ax.set_title("BER vs SNR (approximate)")
ax.grid(True, which="both", ls="--", lw=0.5)
st.pyplot(fig)

# -------------------------------
# Export snapshot or CSV
# -------------------------------
st.markdown("---")
st.subheader("Export / Snapshot")
if st.button("Export node snapshot as CSV"):
    # build dataframe
    df_nodes = pd.DataFrame([{
        "id": n['id'],
        "x": n['pos'][0],
        "y": n['pos'][1],
        "battery": n['battery'],
        "tx_power": n['tx_power'],
        "queue_len": len(n['queue']),
        "sent": n['stats']['sent'],
        "dropped": n['stats']['dropped'],
        "throughput": n['stats']['throughput']
    } for n in nodes])
    csv = df_nodes.to_csv(index=False)
    st.download_button("Download CSV", csv, file_name="cl_netsim_nodes_snapshot.csv", mime="text/csv")

# Also allow JSON-like code view
if st.button("Show JSON Snapshot"):
    snapshot = {
        "time_step": st.session_state['time_step'],
        "nodes": [{k: n[k] for k in ['id','pos','battery','tx_power','stats','priority']} for n in nodes],
        "history": st.session_state['history']
    }
    st.code(str(snapshot))

# -------------------------------
# Save params for reproducibility
# -------------------------------
st.session_state['params'] = {
    "NUM_NODES": NUM_NODES, "AREA_SIZE": AREA_SIZE, "COMM_RANGE": COMM_RANGE,
    "NODE_SPEED": NODE_SPEED, "ACTIVITY_PROB": ACTIVITY_PROB, "CROSS_CORR": CROSS_CORR,
    "CW_MIN": CW_MIN, "PAYLOAD_BITS": PAYLOAD_BITS, "DROP_POLICY": DROP_POLICY,
    "mobility_model": mobility_model, "routing_algo": routing_algo, "enable_recharge": enable_recharge
}
st.markdown("### Notes")
st.markdown("""
- This simulator intentionally balances realism and simplicity so it runs interactively.  
- CDMA MAI is modeled as interfering received power from simultaneously active users scaled by a cross-correlation factor.  
- Modulation → BER uses approximate formulas for demonstration.  
- MAC model is a simplified CSMA/CA with contention windows and collisions.  
- Routing is an illustrative AODV-style hop/energy aware selection (not a full protocol implementation).  
""")
