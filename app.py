import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from q_agent import train_q_agent, evaluate_controllers

st.set_page_config(page_title="Adaptive Smart Traffic RL", layout="wide")
st.title("Adaptive Smart Traffic Signal Controller (Tabular Q-Learning)")
st.caption("Custom Gymnasium MDP Environment | Live Comparison: Fixed-Timer vs. Trained Q-Agent")


def render_intersection_html(title: str, info: dict, accent: str) -> str:
    ns_green = (info["phase"] == 0)
    ns_light = "#22c55e" if ns_green else "#ef4444"
    ew_light = "#ef4444" if ns_green else "#22c55e"
    ns_status = "GREEN" if ns_green else "RED"
    ew_status = "RED" if ns_green else "GREEN"

    ns_bars = min(info["q_ns"], 20)
    ew_bars = min(info["q_ew"], 20)
    ns_pct = int((info["q_ns"] / 25) * 100)
    ew_pct = int((info["q_ew"] / 25) * 100)

    return f"""
    <div style="border:2px solid {accent}; border-radius:12px; padding:16px; background:#0f172a; color:#f8fafc;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <h4 style="margin:0; color:{accent};">{title}</h4>
        <span style="background:#1e293b; padding:4px 10px; border-radius:6px; font-size:13px;">
          Waiting Cars: <b>{info['total_queue']}</b>
        </span>
      </div>

      <div style="background:#1e293b; border-radius:10px; padding:14px; margin-bottom:10px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
          <span><b>North–South Axis</b></span>
          <span style="color:{ns_light}; font-weight:700;">● {ns_status} ({info['q_ns']} cars)</span>
        </div>
        <div style="background:#334155; height:12px; border-radius:6px; overflow:hidden;">
          <div style="width:{ns_pct}%; background:{ns_light}; height:100%;"></div>
        </div>
        <div style="font-size:12px; margin-top:4px; letter-spacing:2px;">{"🚗" * min(ns_bars, 12)}</div>
      </div>

      <div style="background:#1e293b; border-radius:10px; padding:14px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
          <span><b>East–West Axis</b></span>
          <span style="color:{ew_light}; font-weight:700;">● {ew_status} ({info['q_ew']} cars)</span>
        </div>
        <div style="background:#334155; height:12px; border-radius:6px; overflow:hidden;">
          <div style="width:{ew_pct}%; background:{ew_light}; height:100%;"></div>
        </div>
        <div style="font-size:12px; margin-top:4px; letter-spacing:2px;">{"🚙" * min(ew_bars, 12)}</div>
      </div>
    </div>
    """


# Sidebar controls
st.sidebar.header("Traffic & RL Parameters")
preset = st.sidebar.selectbox(
    "Traffic Scenario",
    ["Asymmetric Rush Hour (Best Demo)", "Heavy Balanced Traffic", "Custom"],
)

if preset == "Asymmetric Rush Hour (Best Demo)":
    default_ns, default_ew = 1.6, 0.4
elif preset == "Heavy Balanced Traffic":
    default_ns, default_ew = 1.1, 1.1
else:
    default_ns, default_ew = 1.5, 0.6

lambda_ns = st.sidebar.slider("North-South Arrival Rate (cars/step)", 0.2, 2.2, default_ns, 0.1)
lambda_ew = st.sidebar.slider("East-West Arrival Rate (cars/step)", 0.2, 2.2, default_ew, 0.1)
fixed_cycle = st.sidebar.slider("Fixed-Timer Switch Interval (steps)", 2, 8, 3)

st.sidebar.divider()
episodes = st.sidebar.slider("Training Episodes", 2000, 12000, 8000, 1000)
alpha = st.sidebar.slider("Learning Rate (alpha)", 0.05, 0.5, 0.1, 0.05)
gamma = st.sidebar.slider("Discount Factor (gamma)", 0.70, 0.99, 0.95, 0.01)
eval_seed = st.sidebar.number_input("Test Traffic Seed (Fair Comparison)", value=42, step=1)

if st.sidebar.button("Train Q-Agent & Evaluate", type="primary", use_container_width=True):
    with st.spinner("Training Tabular Q-Learning agent..."):
        agent, rewards = train_q_agent(
            lambda_ns=lambda_ns,
            lambda_ew=lambda_ew,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
        )
        fixed_trace, rl_trace = evaluate_controllers(
            q_agent=agent,
            lambda_ns=lambda_ns,
            lambda_ew=lambda_ew,
            fixed_cycle=fixed_cycle,
            eval_seed=int(eval_seed),
        )
    st.session_state["agent"] = agent
    st.session_state["rewards"] = rewards
    st.session_state["fixed_trace"] = fixed_trace
    st.session_state["rl_trace"] = rl_trace

if "agent" not in st.session_state:
    st.info("Click **Train Q-Agent & Evaluate** in the sidebar to train the RL agent (~1 second) and compare it live against a fixed-timer signal.")
    st.stop()

agent = st.session_state["agent"]
rewards = st.session_state["rewards"]
fixed_trace = st.session_state["fixed_trace"]
rl_trace = st.session_state["rl_trace"]

# Summary KPIs
fixed_queues = [s["total_queue"] for s in fixed_trace]
rl_queues = [s["total_queue"] for s in rl_trace]
avg_fixed = float(np.mean(fixed_queues))
avg_rl = float(np.mean(rl_queues))
improvement = ((avg_fixed - avg_rl) / max(avg_fixed, 1e-6)) * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("Fixed-Timer Avg Queue", f"{avg_fixed:.2f} cars")
k2.metric("Q-Learning Avg Queue", f"{avg_rl:.2f} cars")
k3.metric("Congestion Reduction", f"{improvement:.1f}%", delta=f"{avg_fixed - avg_rl:.2f} fewer cars")
k4.metric("Peak Queue (Fixed vs RL)", f"{max(fixed_queues)} vs {max(rl_queues)}")

tab1, tab2, tab3 = st.tabs([
    "Live Side-by-Side Simulation",
    "Convergence & Queue Curves",
    "Learned Policy Heatmap",
])

with tab1:
    col_btn, col_spd = st.columns([1, 2])
    run_anim = col_btn.button("Play Live Simulation", use_container_width=True)
    delay = col_spd.slider("Animation Speed (sec/step)", 0.05, 0.5, 0.15, 0.05)

    step_banner = st.empty()
    c1, c2 = st.columns(2)
    left_box = c1.empty()
    right_box = c2.empty()

    if run_anim:
        for i in range(len(fixed_trace)):
            step_banner.markdown(f"**Simulation Step:** `{i} / {len(fixed_trace) - 1}` (Identical Poisson Traffic Stream)")
            left_box.markdown(render_intersection_html("Fixed-Timer Signal (Baseline)", fixed_trace[i], "#f97316"), unsafe_allow_html=True)
            right_box.markdown(render_intersection_html("Trained Q-Learning Agent (RL)", rl_trace[i], "#38bdf8"), unsafe_allow_html=True)
            time.sleep(delay)
    else:
        final_idx = len(fixed_trace) - 1
        step_banner.markdown(f"**Final Snapshot (Step {final_idx})** — Click *Play Live Simulation* to watch step-by-step.")
        left_box.markdown(render_intersection_html("Fixed-Timer Signal (Baseline)", fixed_trace[final_idx], "#f97316"), unsafe_allow_html=True)
        right_box.markdown(render_intersection_html("Trained Q-Learning Agent (RL)", rl_trace[final_idx], "#38bdf8"), unsafe_allow_html=True)

with tab2:
    g1, g2 = st.columns(2)

    with g1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        window = max(10, len(rewards) // 25)
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(rewards, alpha=0.25, color="#64748b", label="Episode Reward")
        ax1.plot(range(window - 1, len(rewards)), moving_avg, color="#2563eb", linewidth=2, label=f"{window}-Ep Moving Avg")
        ax1.set_title("Q-Learning Convergence Curve")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Cumulative Reward (Higher is Better)")
        ax1.legend()
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)
        plt.close(fig1)

    with g2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        steps = list(range(len(fixed_queues)))
        ax2.plot(steps, fixed_queues, color="#f97316", linewidth=2, label="Fixed-Timer Baseline")
        ax2.plot(steps, rl_queues, color="#0ea5e9", linewidth=2, label="Trained Q-Agent")
        ax2.set_title("Total Waiting Vehicles Over Time")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Total Queued Cars (NS + EW)")
        ax2.legend()
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)

with tab3:
    st.markdown("Shows the optimal action learned by the Q-table when **North–South is currently GREEN** (`0 = Keep NS Green`, `1 = Switch to EW Green`).")
    policy_grid = np.zeros((11, 11))
    for ns in range(11):
        for ew in range(11):
            state_id = (ns * 11 + ew) * 2 + 0  # phase = 0 (NS Green)
            policy_grid[ns, ew] = np.argmax(agent.q_table[state_id])

    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    im = ax3.imshow(policy_grid, origin="lower", cmap="coolwarm", aspect="auto")
    ax3.set_title("Learned Policy Decision Boundary (Current Phase: NS Green)")
    ax3.set_xlabel("East–West Queue Length (cars)")
    ax3.set_ylabel("North–South Queue Length (cars)")
    cbar = fig3.colorbar(im, ax=ax3, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Keep NS Green (0)", "Switch to EW (1)"])
    st.pyplot(fig3)
    plt.close(fig3)