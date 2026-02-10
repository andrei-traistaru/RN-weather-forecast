import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import csv
from datetime import datetime

# --- 1. CONFIGURARE CAI  ---
current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, '../'))
if not os.path.exists(os.path.join(project_root, 'config')):

    project_root = os.path.abspath(os.path.join(current_dir, '../../'))

if project_root not in sys.path:
    sys.path.append(project_root)

try:
    import config.config as config
except ImportError:
    st.error(f"Eroare critica: Nu s-a putut gasi modulul 'config'. Root: {project_root}")
    st.stop()

# --- 2. SETARI PAGINA ---
st.set_page_config(
    page_title="Meteo AI Forecast - Real Inference",
    layout="wide"
)

# --- 3. MODUL DATA LOGGING ---
def log_prediction(timestamp_sim, temp_pred, hum_pred, source):
    """
    Salveaza predictiile in CSV.
    """
    log_dir = os.path.join(config.BASE_DIR, 'data', 'generated')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'prediction_logs.csv')
    file_exists = os.path.isfile(log_file)
    
    try:
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Sursa_Date', 'Timestamp_Simulare', 'Timestamp_Logare', 'Temp_Pred_1h', 'Hum_Pred_1h'])
            
            writer.writerow([
                source,
                timestamp_sim, 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                f"{temp_pred:.2f}", 
                f"{hum_pred:.2f}"
            ])
        return True
    except Exception as e:
        st.warning(f"Eroare logging: {e}")
        return False

# --- 4. FUNCTII DE INCARCARE ---
@st.cache_resource
def load_resources():
    # Folosim optimized_model conform cerintei anterioare
    model_path = os.path.join(config.MODELS_PATH, 'optimized_model.keras')
    scaler_x_path = os.path.join(config.MODELS_PATH, 'scaler_x.save')
    scaler_y_path = os.path.join(config.MODELS_PATH, 'scaler_y.save')
    
    if not os.path.exists(model_path):
        # Fallback pe trained_model daca optimized nu exista inca
        fallback_path = os.path.join(config.MODELS_PATH, 'trained_model.keras')
        if os.path.exists(fallback_path):
            model_path = fallback_path
        else:
            st.error(f"[EROARE] Lipsa model la: {model_path}")
            return None, None, None, None

    try:
        model = load_model(model_path)
        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        return model, scaler_x, scaler_y, model_path
    except Exception as e:
        st.error(f"Eroare resurse: {e}")
        return None, None, None, None

@st.cache_data
def load_data(file_source):
    """
    Incarca datele si converteste data la datetime.
    """
    try:
        if isinstance(file_source, str):
            if os.path.exists(file_source):
                df = pd.read_csv(file_source)
            else:
                return None
        else:
            df = pd.read_csv(file_source)
            
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

        # Smoothing
        if 'Humidity' in df.columns:
            df['Humidity'] = df['Humidity'].rolling(window=3, min_periods=1).mean()
        return df
    except Exception as e:
        st.error(f"Nu s-a putut citi fisierul CSV: {e}")
        return None

# --- 5. PREPROCESARE ---
def prepare_input(df_window, scaler_x):
    df = df_window.copy()
    
    # Feature Engineering (Temporal)
    if 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'])
        timestamp_s = df['date'].map(pd.Timestamp.timestamp)
        day = 24 * 60 * 60
        year = (365.2425) * day
        df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    else:
        df['Day sin'] = 0; df['Day cos'] = 0; df['Year sin'] = 0; df['Year cos'] = 0

    # Feature Engineering (Vant)
    if 'WindDirection' in df.columns:
        wd_rad = df['WindDirection'] * np.pi / 180
        df['WindDx'] = np.cos(wd_rad)
        df['WindDy'] = np.sin(wd_rad)
    else:
        df['WindDx'] = 0; df['WindDy'] = 0
        
    # Feature Engineering (Dew Point)
    b = 17.625; c = 243.04
    gamma = (b * df['Temperature']) / (c + df['Temperature']) + np.log((df['Humidity'] + 0.0001) / 100.0)
    df['DewPoint'] = (c * gamma) / (b - gamma)
    
    feature_cols = [
        'Temperature', 'Pressure', 'Humidity', 'WindSpeed', 
        'WindDx', 'WindDy',
        'Day sin', 'Day cos', 'Year sin', 'Year cos', 'DewPoint'
    ]
    
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Coloane lipsa in CSV: {missing_cols}. Fisierul trebuie sa contina: Temperature, Pressure, Humidity, WindSpeed, WindDirection, Date.")
        st.stop()

    X_raw = df[feature_cols].values
    X_scaled = scaler_x.transform(X_raw)
    X_final = X_scaled.reshape(1, config.SEQ_LENGTH, len(feature_cols))
    return X_final

# --- 6. INTERFATA UTILIZATOR (UI) ---

st.title("Sistem de Prognoza Meteo (Bidirectional LSTM)")
st.markdown("### Inferenta cu Incarcare CSV si Selectie Data")

# A. Incarcare Resurse
model, scaler_x, scaler_y, model_path_loaded = load_resources()

if model is None:
    st.stop()

# B. Sidebar: Selector Sursa Date
st.sidebar.header("Sursa Date")
data_source = st.sidebar.radio(
    "Alege metoda de intrare:",
    ["Simulare (Date Istorice)", "Incarca Fisier CSV Propriu"]
)

df_all = None

if data_source == "Simulare (Date Istorice)":
    st.sidebar.info("Mod: Validare pe dataset-ul de antrenare.")
    df_all = load_data(config.DATA_PATH)
    if df_all is None:
        st.error("Nu s-a gasit fisierul de date istorice.")
        st.stop()

else:
    st.sidebar.info("Mod: Predictie pe fisier extern.")
    uploaded_file = st.sidebar.file_uploader("Trage un fisier CSV aici", type=["csv"])
    
    if uploaded_file is not None:
        df_all = load_data(uploaded_file)
    else:
        st.info("Te rog incarca un fisier CSV in sidebar pentru a incepe.")
        st.stop()

# Validare lungime
if len(df_all) < config.SEQ_LENGTH:
    st.error(f"Fisierul selectat are prea putine randuri ({len(df_all)}). Minim necesar pentru istoric: {config.SEQ_LENGTH} (72 ore).")
    st.stop()

# C. Selector Moment
st.sidebar.divider()
st.sidebar.header("Ce moment analizam?")

max_idx_with_gt = len(df_all) - config.SEQ_LENGTH - config.FORECAST_STEPS
has_ground_truth = max_idx_with_gt >= 0

if has_ground_truth:
    max_slider_val = max_idx_with_gt
else:
    max_slider_val = len(df_all) - config.SEQ_LENGTH
    st.sidebar.warning("Atentie: Nu exista date viitoare pentru validare.")

valid_indices = range(0, max_slider_val + 1)

def format_func(idx):
    target_idx = idx + config.SEQ_LENGTH - 1
    if target_idx < len(df_all):
        date_val = df_all['Date'].iloc[target_idx]
        return date_val.strftime('%Y-%m-%d %H:%M')
    return str(idx)

default_idx = max(0, max_slider_val - 200)

start_idx = st.sidebar.select_slider(
    "Selecteaza Data Simularii:",
    options=valid_indices,
    value=default_idx,
    format_func=format_func
)

df_history = df_all.iloc[start_idx : start_idx + config.SEQ_LENGTH].copy()
current_date_str = df_history['Date'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')

ground_truth_future = None
if has_ground_truth and start_idx <= max_idx_with_gt:
    ground_truth_future = df_all.iloc[start_idx + config.SEQ_LENGTH : start_idx + config.SEQ_LENGTH + config.FORECAST_STEPS].copy()

st.info(f"Moment Start Prognoza: {current_date_str} | Lungime fisier: {len(df_all)} randuri")

# Buton Principal
if st.button("Genereaza Prognoza (24h)", type="primary"):
    
    # 1. Preprocesare
    try:
        X_input = prepare_input(df_history, scaler_x)
    except ValueError as e:
        st.error(f"Eroare dimensionala: {e}")
        st.stop()
    
    # 2. Inferenta
    with st.spinner("Analiza Retea Neuronala..."):
        pred_scaled = model.predict(X_input) 
    
    # 3. Post-procesare
    pred_real = scaler_y.inverse_transform(pred_scaled[0])
    
    # 4. Calibrare (Anchor Alignment)
    last_real_temp = df_history['Temperature'].iloc[-1]
    last_real_hum = df_history['Humidity'].iloc[-1]
    
    bias_temp = pred_real[0][0] - last_real_temp
    bias_hum = pred_real[0][1] - last_real_hum
    
    pred_real[:, 0] -= bias_temp
    pred_real[:, 1] -= bias_hum
    pred_real[:, 1] = np.clip(pred_real[:, 1], 0, 100)
    
    df_pred = pd.DataFrame(pred_real, columns=["Temp Pred", "Hum Pred"])
    df_pred["Ora"] = [f"+{i+1}h" for i in range(24)]
    
    # Vizualizare KPI
    col1, col2, col3, col4 = st.columns(4)
    temp_next = df_pred['Temp Pred'].iloc[0]
    hum_next = df_pred['Hum Pred'].iloc[0]

    col1.metric("Temperatura Acum", f"{last_real_temp:.1f} C")
    col3.metric("Umiditate Acum", f"{last_real_hum:.1f}%")
    
    if ground_truth_future is not None:
        mae_temp = np.mean(np.abs(df_pred['Temp Pred'].values - ground_truth_future['Temperature'].values))
        mae_hum = np.mean(np.abs(df_pred['Hum Pred'].values - ground_truth_future['Humidity'].values))
        
        col2.metric("Eroare (MAE)", f"{mae_temp:.2f} C", delta_color="inverse")
        col4.metric("Eroare (MAE)", f"{mae_hum:.2f}%", delta_color="inverse")
    else:
        col2.metric("Prognoza +1h", f"{temp_next:.1f} C")
        col4.metric("Prognoza +1h", f"{hum_next:.1f}%")

    st.subheader("Grafice Interactive")
    tab1, tab2 = st.tabs(["Temperatura", "Umiditate"])
    
    idx_hist = range(len(df_history))
    idx_future = range(len(df_history), len(df_history) + 24)
    
    def plot_graph(ax, hist_data, pred_data, real_data, color_pred, title):
        ax.plot(idx_hist, hist_data, label='Istoric (72h)', color='gray', alpha=0.6)
        
        last_val = hist_data.iloc[-1]
        plot_pred_idx = np.concatenate(([idx_hist[-1]], idx_future))
        plot_pred_val = np.concatenate(([last_val], pred_data))
        ax.plot(plot_pred_idx, plot_pred_val, label='Prognoza AI', color=color_pred, linewidth=2, marker='o', markersize=4)
        
        if real_data is not None:
            ax.plot(idx_future, real_data, label='Realitate (Ground Truth)', color='green', linestyle='--', alpha=0.7)
            
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    with tab1:
        fig, ax = plt.subplots(figsize=(12, 5))
        real_vals = ground_truth_future['Temperature'] if ground_truth_future is not None else None
        plot_graph(ax, df_history['Temperature'], df_pred['Temp Pred'].values, real_vals, 'red', "Evolutie Temperatura")
        st.pyplot(fig)
        
    with tab2:
        fig, ax = plt.subplots(figsize=(12, 5))
        real_vals = ground_truth_future['Humidity'] if ground_truth_future is not None else None
        plot_graph(ax, df_history['Humidity'], df_pred['Hum Pred'].values, real_vals, 'blue', "Evolutie Umiditate")
        st.pyplot(fig)
        
    st.subheader("Valori Prognozate")
    
    st.dataframe(
        df_pred.style.format({
            "Temp Pred": "{:.2f}", 
            "Hum Pred": "{:.2f}"
        }), 
        use_container_width=True
    )
    
    log_source = "UPLOAD_CSV" if data_source == "Incarca Fisier CSV Propriu" else "HISTORY"
    if log_prediction(current_date_str, temp_next, hum_next, log_source):
        st.toast("Salvat in Logs!", icon=None)

else:
    st.info("Alege sursa datelor si apasa 'Genereaza Prognoza'.")