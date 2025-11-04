# app.py
import streamlit as st
import numpy as np
import pandas as pd
import math
import time
import random
from collections import deque

st.set_page_config(page_title="CL-NetSim Pro", layout="wide")
st.title("📡 CL-NetSim Pro — Cross-Layer Wireless Network Simulator")
st.markdown("Interactive simulator: CDMA MAI, adaptive PHY/MAC, mobility & MANET routing. **No datasets, no models.**")

# -----------------------------
# Helpers: modulation, BER approx, MAI, routing metric
# -----------------------------
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

def ber_estimate(mod, snr_db):
    # rough symbolic BER estimate (not exact formulas) to keep model simple
    snr = 10**(snr_db/10)
    if mod=="BPSK":
        return 0.5*math.exp(-snr/2)
    elif mod=="QPSK":
        return 0.4*math.exp(-snr/2)
    elif mod=="16-QAM":
        return 0.2*math.exp(-snr/4)
    else:
        return 0.1*math.exp(-snr/6)

def path_loss(dist):
    # simple path loss: Friis-like power decay (normalized)
    if dist <= 0: return 1.0
    return 1.0/(1 + (dist/50.0)**2)

def compute_MAI(node_idx, nodes, activity_prob=0.7, cross_corr=0.1):
    # Sum of interfering power from other active users scaled by cross-correlation
    target = nodes[node_idx]
    total = 0.0
    for j, other in enumerate(nodes):
        if j==node_idx: continue
        if random.random() < activity_prob:
            d = np.linalg.norm(np.array(target['pos']) - np.array(other['pos']))
            p_rx = other['tx_power'] * path_loss(d)
            # cross-correlation factor simulates code orthogonality imperfection
            total += p_rx * cross_corr
    return total

def routing_metric(path, nodes):
    # Combine residual energy (prefer high energy) and path quality (SNR proxy)
    # Metric: lower is better
    energy_term = sum(1.0/(0.01 + nodes[n]['battery']) for n in path) / len(path)
    snr_term = 0
    for i in range(len(path)-1):
        d = np.linalg.norm(np.array(nodes[path[i]]['pos']) - np.array(nodes[path[i+1]]['pos']))
        snr_est = 10 * path_loss(d) * nodes[path[i]]['tx_power']  # proxy
        snr_term += 1.0/(0.01 + snr_est)
    snr_term = snr_term / max(1, (len(path)-1))
    return 0.6*energy_term + 0.4*snr_term

# -----------------------------
# Simulation Parameters (UI)
# -----------------------------
with st.sidebar:
    st.header("Simulation Controls")
    NUM_NODES = st.slider("Number of nodes (mobile)", 3, 12, 6)
    SIM_STEP_MS = st.slider("Sim step interval (ms)", 100, 2000, 500)
    MAX_STEPS = st.number_input("Max sim steps (for Run)", min_value=10, max_value=10000, value=500)
    NODE_SPEED = st.slider("Max node speed (units/step)", 0.0, 5.0, 1.0)
    ACTIVITY_PROB = st.slider("User activity probability", 0.0, 1.0, 0.7)
    CROSS_CORR = st.slider("CDMA cross-correlation (MAI factor)", 0.0, 0.5, 0.12)
    RECHARGE = st.checkbox("Enable ambient recharge (slow)", value=False)
    st.markdown("---")
    st.write("Adaptive policies")
    QUEUE_LIMIT = st.slider("Queue limit per node", 1, 50, 12)
    DROP_POLICY = st.selectbox("Drop policy when full", ["Tail Drop","Early Drop (prob)","Priority Drop"], index=0)
    st.markdown("---")
    sim_mode = st.radio("Mode", ["Step", "Run"], index=0)
    st.button("Reset Simulation", key="reset")

# -----------------------------
# Setup nodes (session state)
# -----------------------------
if 'nodes' not in st.session_state or st.session_state.get('reset', False):
    st.session_state['nodes'] = []
    st.session_state['time'] = 0
    st.session_state['history'] = {
        'time':[], 'avg_throughput':[], 'avg_delay':[], 'avg_battery':[], 'packet_drops':[]
    }
    # Initialize random nodes: position, battery, queue
    for i in range(NUM_NODES):
        node = {
            'id': i,
            'pos': [random.uniform(0,200), random.uniform(0,200)],
            'velocity': [random.uniform(-NODE_SPEED,NODE_SPEED), random.uniform(-NODE_SPEED,NODE_SPEED)],
            'battery': random.uniform(0.5,1.0),
            'tx_power': random.uniform(0.4,1.0),
            'queue': deque(),
            'stats': {'sent':0,'dropped':0,'delay_sum':0,'throughput':0.0},
            'priority': random.choice([0,1])  # 1 -> high priority (for priority drop)
        }
        st.session_state['nodes'].append(node)
    st.session_state['reset'] = False

# If user changed NUM_NODES or other controls, ensure nodes length matches (simple re-init)
if len(st.session_state['nodes']) != NUM_NODES:
    st.session_state['reset'] = True
    st.experimental_rerun()

nodes = st.session_state['nodes']

# -----------------------------
# Core simulation step
# -----------------------------
def simulate_step():
    t = st.session_state['time'] + 1
    nodes = st.session_state['nodes']
    total_throughput = 0.0
    total_delay = 0.0
    total_batt = 0.0
    total_drops = 0

    # New arrivals: generate packets per node (burstiness modeled by ACTIVITY_PROB and random amount)
    for n in nodes:
        if random.random() < ACTIVITY_PROB:
            # burst size proportional to activity
            burst = np.random.poisson(1 + 3*random.random())
            for _ in range(burst):
                pkt = {
                    'arrival_time': t,
                    'size': random.choice([1,1,2]), # abstract packet size units
                    'priority': n['priority']
                }
                if len(n['queue']) < QUEUE_LIMIT:
                    n['queue'].append(pkt)
                else:
                    # drop policies
                    if DROP_POLICY == "Tail Drop":
                        n['stats']['dropped'] += 1
                        total_drops += 1
                    elif DROP_POLICY == "Early Drop (prob)":
                        if random.random() < 0.5:
                            n['stats']['dropped'] += 1
                            total_drops += 1
                        else:
                            # accept by popping oldest
                            if n['queue']:
                                n['queue'].popleft()
                                n['queue'].append(pkt)
                    else: # Priority Drop
                        # drop a low priority if exists, else tail drop
                        found = False
                        for i,q in enumerate(n['queue']):
                            if q['priority']==0:
                                del n['queue'][i]
                                n['queue'].append(pkt)
                                found = True
                                break
                        if not found:
                            n['stats']['dropped'] += 1
                            total_drops += 1

    # Mobility update + optional small random drift
    for n in nodes:
        # update velocity slightly
        n['velocity'][0] += random.uniform(-0.2,0.2)
        n['velocity'][1] += random.uniform(-0.2,0.2)
        # limit velocity
        n['velocity'][0] = max(-NODE_SPEED, min(NODE_SPEED, n['velocity'][0]))
        n['velocity'][1] = max(-NODE_SPEED, min(NODE_SPEED, n['velocity'][1]))
        n['pos'][0] = (n['pos'][0] + n['velocity'][0]) % 300
        n['pos'][1] = (n['pos'][1] + n['velocity'][1]) % 300

    # For each node, compute channel conditions (proxy SNR from tx_power and path loss to a base station or to neighbors)
    # We'll assume each node transmits to a local "sink" at center [150,150] for throughput metrics, but routing may use node-to-node hops.
    sink = np.array([150.0, 150.0])
    for i,n in enumerate(nodes):
        d_sink = np.linalg.norm(np.array(n['pos']) - sink)
        pl = path_loss(d_sink)
        # received power proxy
        rx_power = n['tx_power'] * pl
        # compute MAI
        mai = compute_MAI(i, nodes, activity_prob=ACTIVITY_PROB, cross_corr=CROSS_CORR)
        # SNR in dB (noise floor normalized)
        noise_floor = 0.01
        snr_linear = (rx_power) / (noise_floor + mai)
        snr_db = 10*math.log10(max(1e-9, snr_linear))
        n['last_snr_db'] = snr_db
        # Cross-layer decision: modulation
        n['modulation'] = select_modulation(snr_db)
        bits = modulation_bits(n['modulation'])
        # Power control: if battery low, reduce power
        if n['battery'] < 0.25:
            n['tx_power'] = max(0.1, n['tx_power'] * 0.85)
        else:
            # opportunistically raise power if SNR poor
            if snr_db < 6:
                n['tx_power'] = min(1.5, n['tx_power'] * 1.05)

        # Admission control: if MAI high and queue long, reject some packets (simple)
        admitted = []
        if n['queue']:
            while n['queue']:
                pkt = n['queue'][0]
                # admission rule: if mai reduces effective bits significantly, reject
                eff_bits = bits * (1.0 / (1.0 + mai*5))
                if len(admitted) > 0 and eff_bits < 0.5:
                    # reject remaining to limit congestion
                    break
                # else transmit this packet
                admitted.append(n['queue'].popleft())

        # Transmit admitted packets: throughput = sum(eff_bits * size) scaled
        sent_bits = 0.0
        sent_delay = 0.0
        for pkt in admitted:
            eff_bits = bits * (1.0 / (1.0 + mai*5))
            # if BER high, some errors -> retransmissions simulated as partial drop
            ber = ber_estimate(n['modulation'], snr_db)
            success_prob = max(0.01, 1.0 - ber)
            if random.random() < success_prob:
                sent_bits += eff_bits * pkt['size']
                n['stats']['sent'] += 1
                sent_delay += (t - pkt['arrival_time'])
            else:
                # unsuccessful -> counted as dropped for throughput but not queue overflow
                n['stats']['dropped'] += 1
                total_drops += 1

            # energy cost per packet (proportional to tx_power and bits)
            n['battery'] -= 0.001 * n['tx_power'] * (1 + eff_bits/4.0)
            n['battery'] = max(0.0, n['battery'])

        n['stats']['throughput'] = sent_bits
        if sent_bits > 0:
            n['stats']['delay_sum'] += sent_delay

        total_throughput += sent_bits
        total_delay += sent_delay
        total_batt += n['battery']

    # Simple MANET routing demonstration: compute best 1-hop neighbor for each node to reach sink
    # and potentially forward packets (not implemented fully as complex forwarding; this is a metric example)
    for i,n in enumerate(nodes):
        # find candidate neighbors within range
        neighbors = []
        for j,m in enumerate(nodes):
            if i==j: continue
            d = np.linalg.norm(np.array(n['pos']) - np.array(m['pos']))
            if d < 120: # arbitrary comm range
                neighbors.append(j)
        # find best neighbor path metric (direct to sink vs 2-hop via neighbor)
        direct_metric = routing_metric([i], nodes)  # direct proxy
        best = ('direct', direct_metric, [i])
        for nb in neighbors:
            metric = routing_metric([i, nb], nodes)
            if metric < best[1]:
                best = ('via', metric, [i, nb])
        n['routing_choice'] = best

    # ambient recharge if enabled
    if RECHARGE:
        for n in nodes:
            n['battery'] = min(1.0, n['battery'] + 0.0005)

    # collect history
    st.session_state['time'] = t
    st.session_state['history']['time'].append(t)
    st.session_state['history']['avg_throughput'].append(total_throughput / max(1, NUM_NODES))
    st.session_state['history']['avg_delay'].append(total_delay / max(1, NUM_NODES))
    st.session_state['history']['avg_battery'].append(total_batt / max(1, NUM_NODES))
    st.session_state['history']['packet_drops'].append(total_drops)

# -----------------------------
# UI Panels: Controls + Visuals
# -----------------------------
col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Node Controls")
    st.write("Click Step to advance one sim step. Run will iterate until max steps.")
    if st.button("Step"):
        simulate_step()
    run_button = st.button("Run")
    if run_button and sim_mode=="Run":
        steps = 0
        progress = st.progress(0)
        while steps < MAX_STEPS:
            simulate_step()
            steps += 1
            progress.progress(int(steps*100/MAX_STEPS))
            # small sleep to allow UI updates
            time.sleep(max(0.01, SIM_STEP_MS/1000.0))
            # break if user interrupts by toggling reset in sidebar (simple check)
            if st.session_state.get('reset', False):
                break
        st.success(f"Run finished: {steps} steps")
    st.markdown("---")
    st.write("Per-node quick stats:")
    for n in nodes:
        with st.expander(f"Node {n['id']} — Battery {n['battery']:.2f} — Mod:{n.get('modulation','-')}"):
            st.write(f"Pos: {tuple(round(x,1) for x in n['pos'])}")
            st.write(f"Tx power: {n['tx_power']:.2f}")
            st.write(f"SNR (dB): {n.get('last_snr_db',0):.2f}")
            st.write(f"Queue len: {len(n['queue'])}  Priority: {n['priority']}")
            st.write(f"Sent: {n['stats']['sent']} | Dropped: {n['stats']['dropped']}")
            st.write("Routing:", n.get('routing_choice',('direct', None,[])))

with col2:
    st.subheader("Live Network View")
    # Scatter plot of nodes and sink
    df_nodes = pd.DataFrame([{
        'id': n['id'],
        'x': n['pos'][0],
        'y': n['pos'][1],
        'battery': n['battery'],
        'tx_power': n['tx_power'],
        'queue': len(n['queue'])
    } for n in nodes])
    # Use Streamlit chart for scatter (plotly style via st.map is not suitable here); use altair via st.altair_chart? simpler: use st.pyplot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    plt.scatter(df_nodes['x'], df_nodes['y'], s=150*df_nodes['battery']+10, alpha=0.8)
    for _,row in df_nodes.iterrows():
        plt.text(row['x']+2, row['y']+2, f"#{int(row['id'])}")
    plt.scatter([150],[150], marker='*', s=200, c='red') # sink
    plt.xlim(0,300); plt.ylim(0,300)
    plt.title("Nodes (size ~ battery)  — Sink marked red")
    st.pyplot(plt)

    st.subheader("Performance Trends")
    hist = st.session_state['history']
    if len(hist['time']) > 0:
        perf_df = pd.DataFrame({
            'time': hist['time'],
            'throughput': hist['avg_throughput'],
            'delay': hist['avg_delay'],
            'battery': hist['avg_battery'],
            'drops': hist['packet_drops']
        }).set_index('time')
        st.line_chart(perf_df[['throughput','battery']])
        st.line_chart(perf_df[['delay','drops']])
    else:
        st.info("Run the simulation to populate charts.")

# -----------------------------
# Save / Export quick snapshot (JSON-like)
# -----------------------------
st.markdown("---")
if st.button("Export Snapshot (to clipboard)"):
    snapshot = {
        'time': st.session_state['time'],
        'nodes': [{k:n[k] for k in ['id','pos','battery','tx_power','stats','priority']} for n in nodes]
    }
    st.code(str(snapshot))

st.markdown("### Notes")
st.markdown("""
- The simulator is intentionally abstract to keep it lightweight and educational.
- **CDMA MAI** is modeled as interfering received power from other active users scaled by a cross-correlation factor.
- **Cross-layer decisions**: PHY (modulation/power) ↔ MAC (admission/queue) ↔ Network (routing metric).
- Extend: add true multi-hop forwarding, more accurate BER formulas, or GUI for packet flows.
""")
