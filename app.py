import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
import time
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Innovista · Smart Water Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

/* ── base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    background-color: #050d14 !important;
    color: #c8e6ff !important;
}
.stApp { background-color: #050d14 !important; }

/* hide default header / footer */
#MainMenu, footer, header { visibility: hidden; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #050d14; }
::-webkit-scrollbar-thumb { background: #1a3a5c; border-radius: 4px; }

/* ── top header bar ── */
.hero-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 28px;
    background: rgba(10,20,40,0.9);
    border: 1px solid rgba(40,120,200,0.2);
    border-radius: 14px;
    margin-bottom: 20px;
}
.hero-logo {
    font-size: 26px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
}
.hero-logo span { color: #00c6ff; }
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #4a7aaa;
    margin-top: 2px;
    letter-spacing: 1px;
}
.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(74,222,128,0.1);
    border: 1px solid rgba(74,222,128,0.35);
    border-radius: 20px;
    padding: 5px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #4ade80;
    letter-spacing: 1.2px;
}
.pulse {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #4ade80;
    display: inline-block;
    animation: blink 1.4s ease-in-out infinite;
}
@keyframes blink {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.3; transform:scale(0.65); }
}

/* ── metric cards ── */
.metric-card {
    background: rgba(10,30,55,0.85);
    border-radius: 12px;
    padding: 20px 22px 16px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent,#00c6ff), transparent);
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #5a8ab0;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 36px;
    font-weight: 600;
    color: #ffffff;
    line-height: 1;
}
.metric-unit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #3a6080;
    margin-top: 4px;
}
.metric-bar-wrap {
    margin-top: 14px;
    height: 3px;
    background: rgba(255,255,255,0.07);
    border-radius: 3px;
    overflow: hidden;
}
.metric-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: var(--accent,#00c6ff);
    transition: width 0.5s ease;
}

/* ── status cards ── */
.status-card {
    border-radius: 12px;
    padding: 14px 18px;
    display: flex;
    align-items: center;
    gap: 14px;
    font-size: 14px;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
}
.status-ok    { background:rgba(20,50,25,0.8);  border:1px solid rgba(74,222,128,0.3); color:#4ade80; }
.status-warn  { background:rgba(60,40,10,0.8);  border:1px solid rgba(251,191,36,0.4); color:#fbbf24; }
.status-danger{ background:rgba(50,10,10,0.85); border:1px solid rgba(239,68,68,0.6);  color:#f87171; }

.status-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    opacity: 0.6;
    font-weight: 400;
    margin-bottom: 2px;
}

/* ── chart panel ── */
.chart-panel {
    background: rgba(10,25,45,0.7);
    border: 1px solid rgba(40,120,200,0.15);
    border-radius: 12px;
    padding: 20px 22px;
    margin-top: 6px;
}
.chart-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #5a8ab0;
    margin-bottom: 14px;
}
.timestamp {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #2a4a65;
    text-align: right;
    margin-top: 8px;
}

/* ── section dividers ── */
.section-gap { margin: 8px 0; }

/* override plotly default white bg */
.js-plotly-plot .plotly, .js-plotly-plot .plotly div { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(
        columns=["time", "flow", "ph", "turbidity"]
    )

# ── Data generation ────────────────────────────────────────────────────────────
def generate_reading():
    return {
        "time":      datetime.now(),
        "flow":      np.random.uniform(10, 50),
        "ph":        np.random.uniform(6.0, 9.0),
        "turbidity": np.random.uniform(1, 10),
    }

new_row = generate_reading()
st.session_state.data = pd.concat(
    [st.session_state.data, pd.DataFrame([new_row])],
    ignore_index=True,
)
df     = st.session_state.data.tail(50).copy()
latest = df.iloc[-1]

# ── AI anomaly detection ───────────────────────────────────────────────────────
is_anomaly = False
if len(df) > 10:
    model      = IsolationForest(contamination=0.1, random_state=42)
    preds      = model.fit_predict(df[["flow", "ph", "turbidity"]])
    is_anomaly = preds[-1] == -1

# ── Alert flags ────────────────────────────────────────────────────────────────
flow_alert = latest["flow"]      > 45
ph_alert   = latest["ph"] < 6.5 or latest["ph"] > 8.5
alert_msgs = []
if flow_alert: alert_msgs.append("Possible Leak Detected")
if ph_alert:   alert_msgs.append("Unsafe pH Level")

# ── Helper: render HTML block ──────────────────────────────────────────────────
def html(content): st.markdown(content, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero bar ───────────────────────────────────────────────────────────────────
html(f"""
<div class="hero-bar">
  <div>
    <div class="hero-logo">Inno<span>vista</span></div>
    <div class="hero-sub">SMART WATER MONITORING SYSTEM</div>
  </div>
  <div class="live-pill">
    <span class="pulse"></span> LIVE &nbsp;·&nbsp; AUTO-REFRESH 2s
  </div>
</div>
""")

# ── Metric cards ───────────────────────────────────────────────────────────────
flow_pct  = ((latest["flow"]      - 10) / 40 * 100)
ph_pct    = ((latest["ph"]        - 6)  /  3 * 100)
turb_pct  = ((latest["turbidity"] - 1)  /  9 * 100)

flow_accent = "#f87171" if flow_alert else "#00c6ff"
ph_accent   = "#f87171" if ph_alert   else "#a78bfa"
turb_accent = "#34d399"

c1, c2, c3 = st.columns(3)

with c1:
    html(f"""
    <div class="metric-card" style="border:1px solid rgba({'239,68,68' if flow_alert else '40,120,200'},0.25); --accent:{flow_accent};">
      <div class="metric-label">⬡ &nbsp;Flow Rate</div>
      <div class="metric-value">{latest['flow']:.1f}</div>
      <div class="metric-unit">L / min &nbsp;·&nbsp; range 10–50</div>
      <div class="metric-bar-wrap">
        <div class="metric-bar-fill" style="width:{flow_pct:.1f}%; background:{flow_accent};"></div>
      </div>
    </div>
    """)

with c2:
    html(f"""
    <div class="metric-card" style="border:1px solid rgba({'239,68,68' if ph_alert else '40,120,200'},0.25); --accent:{ph_accent};">
      <div class="metric-label">⬡ &nbsp;pH Level</div>
      <div class="metric-value">{latest['ph']:.2f}</div>
      <div class="metric-unit">ideal &nbsp;6.5 – 8.5</div>
      <div class="metric-bar-wrap">
        <div class="metric-bar-fill" style="width:{ph_pct:.1f}%; background:{ph_accent};"></div>
      </div>
    </div>
    """)

with c3:
    html(f"""
    <div class="metric-card" style="border:1px solid rgba(40,120,200,0.2); --accent:{turb_accent};">
      <div class="metric-label">⬡ &nbsp;Turbidity</div>
      <div class="metric-value">{latest['turbidity']:.2f}</div>
      <div class="metric-unit">NTU &nbsp;·&nbsp; range 1–10</div>
      <div class="metric-bar-wrap">
        <div class="metric-bar-fill" style="width:{turb_pct:.1f}%; background:{turb_accent};"></div>
      </div>
    </div>
    """)

html('<div class="section-gap"></div>')

# ── Status cards ───────────────────────────────────────────────────────────────
s1, s2 = st.columns(2)

with s1:
    if is_anomaly:
        html("""
        <div class="status-card status-warn">
          <span style="font-size:22px;">⚠</span>
          <div>
            <div class="status-badge">AI Detection</div>
            Anomaly Flagged
          </div>
        </div>""")
    else:
        html("""
        <div class="status-card status-ok">
          <span style="font-size:22px;">✓</span>
          <div>
            <div class="status-badge">AI Detection</div>
            System Normal
          </div>
        </div>""")

with s2:
    if alert_msgs:
        html(f"""
        <div class="status-card status-danger">
          <span style="font-size:22px;">!</span>
          <div>
            <div class="status-badge">Active Alert</div>
            {alert_msgs[0]}{'  +' + str(len(alert_msgs)-1) + ' more' if len(alert_msgs)>1 else ''}
          </div>
        </div>""")
    else:
        html("""
        <div class="status-card status-ok">
          <span style="font-size:22px;">✓</span>
          <div>
            <div class="status-badge">Alerts</div>
            No Active Alerts
          </div>
        </div>""")

html('<div class="section-gap"></div>')

# ── Plotly chart ───────────────────────────────────────────────────────────────
df["time"] = pd.to_datetime(df["time"])
df["time_str"] = df["time"].dt.strftime("%H:%M:%S")

fig = go.Figure()

CHART_STYLE = dict(mode="lines", line_width=1.8)

fig.add_trace(go.Scatter(
    x=df["time_str"], y=df["flow"],
    name="Flow (L/min)", line_color="#00c6ff",
    fill="tozeroy", fillcolor="rgba(0,198,255,0.07)",
    **CHART_STYLE,
))
fig.add_trace(go.Scatter(
    x=df["time_str"], y=df["ph"],
    name="pH Level", line_color="#a78bfa",
    fill="tozeroy", fillcolor="rgba(167,139,250,0.07)",
    **CHART_STYLE,
))
fig.add_trace(go.Scatter(
    x=df["time_str"], y=df["turbidity"],
    name="Turbidity (NTU)", line_color="#34d399",
    fill="tozeroy", fillcolor="rgba(52,211,153,0.07)",
    **CHART_STYLE,
))

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color="#5a8ab0", size=10),
    margin=dict(l=0, r=0, t=10, b=0),
    height=260,
    legend=dict(
        orientation="h", x=0, y=1.15,
        font=dict(color="#6b9fd4", size=11),
        bgcolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(
        gridcolor="rgba(40,120,200,0.08)",
        linecolor="rgba(40,120,200,0.15)",
        tickfont=dict(size=9),
        nticks=8,
    ),
    yaxis=dict(
        gridcolor="rgba(40,120,200,0.08)",
        linecolor="rgba(40,120,200,0.15)",
        tickfont=dict(size=9),
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#08192e",
        bordercolor="rgba(40,120,200,0.4)",
        font=dict(family="JetBrains Mono, monospace", color="#c8e6ff", size=11),
    ),
)

html('<div class="chart-panel">')
html('<div class="chart-title">SENSOR TELEMETRY — LAST 50 READINGS</div>')
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
html(f'<div class="timestamp">Last updated: {latest["time"].strftime("%Y-%m-%d %H:%M:%S")}</div>')
html('</div>')

# ── Auto-refresh ───────────────────────────────────────────────────────────────
time.sleep(2)
st.rerun()