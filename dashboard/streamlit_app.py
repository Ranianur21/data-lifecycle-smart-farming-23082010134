import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Smart Farming Dashboard", page_icon="🌱", layout="wide")

# ========================
# LOAD DATA
# ========================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("outputs/cleaned_data.csv")
    except FileNotFoundError:
        df = pd.read_csv("cleaned_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()

# ========================
# SIDEBAR
# ========================
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("**Time Series Sensor**")
sensor_options = ['soil_moisture_%', 'temperature_C', 'humidity_%']
selected_sensor = st.sidebar.selectbox("Pilih sensor", sensor_options)

soil_threshold = 20
humidity_threshold = 30
temp_threshold = 40

# ========================
# HEADER
# ========================
st.title("🌾 Smart Agriculture Sensor Dashboard")
st.caption(f"Total records: {len(df):,} | {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
st.divider()

# ========================
# METRICS ROW
# ========================
latest = df.sort_values('timestamp').iloc[-1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("🌱 Soil Moisture", f"{round(latest['soil_moisture_%'], 1)}%")
col2.metric("💧 Humidity", f"{round(latest['humidity_%'], 1)}%")
col3.metric("🌡️ Temperature", f"{round(latest['temperature_C'], 1)}°C")
ph_col = 'pH_level' if 'pH_level' in df.columns else 'pH'
col4.metric("⚗️ pH Level", round(latest.get(ph_col, 0), 2))

st.divider()

# ========================
# TIME SERIES
# ========================
st.subheader(f"📈 {selected_sensor.replace('_', ' ').title()} — Time Series")
daily = df.set_index('timestamp')[selected_sensor].resample('D').mean()

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(daily.index, daily.values, color='#2ea043', linewidth=1.8)
ax.fill_between(daily.index, daily.values, alpha=0.1, color='#2ea043')
ax.set_xlabel("Date")
ax.set_ylabel(selected_sensor)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
st.pyplot(fig)

st.divider()

# ========================
# GAUGE
# ========================
st.subheader("🎯 Gauge — Soil Moisture")

gauge_val = round(latest['soil_moisture_%'], 1)
arc_color = '#e74c3c' if gauge_val < soil_threshold else '#2ea043'

gcol_left, gcol_right = st.columns([1, 1])

with gcol_left:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_val,
        number={'suffix': "%", 'font': {'size': 36, 'color': arc_color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
            'bar': {'color': arc_color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': [
                {'range': [0, soil_threshold], 'color': '#fde8e8'},
                {'range': [soil_threshold, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "#e74c3c", 'width': 3},
                'thickness': 0.75,
                'value': soil_threshold
            }
        },
        title={'text': "Soil Moisture (%)", 'font': {'size': 14, 'color': 'gray'}}
    ))
    fig_gauge.update_layout(
        height=250,
        margin=dict(t=40, b=0, l=20, r=20)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with gcol_right:
    st.markdown(f"### {gauge_val}%")
    st.markdown("**Soil Moisture**")
    st.markdown(f"Threshold: `{soil_threshold}%`")
    if gauge_val < soil_threshold:
        st.error("⚠️ Di bawah batas normal")
    else:
        st.success("✅ Normal")

st.divider()

# ========================
# ALERT SYSTEM
# ========================
st.subheader("🚨 Alert System")
alerts = []
if latest['soil_moisture_%'] < soil_threshold:
    alerts.append(f"Soil Moisture rendah: {round(latest['soil_moisture_%'],1)}% < {soil_threshold}%")
if latest['humidity_%'] < humidity_threshold:
    alerts.append(f"Humidity rendah: {round(latest['humidity_%'],1)}% < {humidity_threshold}%")
if latest['temperature_C'] > temp_threshold:
    alerts.append(f"Suhu tinggi: {round(latest['temperature_C'],1)}°C > {temp_threshold}°C")

if alerts:
    for a in alerts:
        st.error(f"⚠️ {a}")
else:
    st.success("✅ Semua parameter dalam batas normal")

st.divider()

# ========================
# HEATMAP + DATA QUALITY
# ========================
col_h, col_q = st.columns([3, 2])

with col_h:
    st.subheader("🔥 Sensor Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['number'])
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax2, fmt=".2f", linewidths=0.5)
    plt.tight_layout()
    st.pyplot(fig2)

with col_q:
    st.subheader("📊 Data Quality Score")

    total = len(df)
    missing = df.isnull().sum().sum()
    non_null = df.notnull().sum().sum()

    accuracy = 1 - (missing / (total * len(df.columns)))
    completeness = non_null / (total * len(df.columns))
    recent = df[df['timestamp'] >= df['timestamp'].max() - pd.Timedelta(days=30)]
    timeliness = len(recent) / total

    st.write("**Accuracy**")
    st.progress(accuracy, text=f"{round(accuracy*100, 1)}%")

    st.write("**Completeness**")
    st.progress(completeness, text=f"{round(completeness*100, 1)}%")

    st.write("**Timeliness**")
    st.progress(timeliness, text=f"{round(timeliness*100, 1)}%")

    overall = (accuracy + completeness + timeliness) / 3
    st.metric("Overall Quality Score", f"{round(overall*100, 1)}%")