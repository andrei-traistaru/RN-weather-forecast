import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

import config.config as config

def add_gaussian_noise(X_data, noise_level=0.01):
    noise = np.random.normal(loc=0, scale=noise_level, size=X_data.shape)
    return X_data + noise

def get_processed_data():
    if not os.path.exists(config.DATA_PATH):
        print(f"[EROARE] Lipsa fisier date: {config.DATA_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(config.DATA_PATH)
    
    # --- 0. SMOOTHING ---
    # Aplicam o medie mobila pe 3 ore pentru a elimina zgomotul
    if 'Humidity' in df.columns:
        df['Humidity'] = df['Humidity'].rolling(window=3, min_periods=1).mean()
    
    # --- 1. Features Temporale ---
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
        df['Day sin'] = 0; df['Day cos'] = 0
        df['Year sin'] = 0; df['Year cos'] = 0

    # --- 2. Features Vant ---
    if 'WindDirection' in df.columns:
        wd_rad = df['WindDirection'] * np.pi / 180
        df['WindDx'] = np.cos(wd_rad)
        df['WindDy'] = np.sin(wd_rad)
    else:
        df['WindDx'] = 0
        df['WindDy'] = 0

    # --- 3. Dew Point ---
    # Formula Magnus
    b = 17.625
    c = 243.04
    gamma = (b * df['Temperature']) / (c + df['Temperature']) + np.log((df['Humidity'] + 0.0001) / 100.0)
    df['DewPoint'] = (c * gamma) / (b - gamma)
    
    # --- 4. Definire Coloane ---
    feature_cols = [
        'Temperature', 'Pressure', 'Humidity', 'WindSpeed', 
        'WindDx', 'WindDy',
        'Day sin', 'Day cos', 'Year sin', 'Year cos',
        'DewPoint'
    ]
    target_cols = config.TARGETS 

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_x.fit_transform(df[feature_cols].values)
    y_scaled = scaler_y.fit_transform(df[target_cols].values)
    
    X, y = [], []
    
    # --- GENERARE SECVENTE ---
    forecast_steps = config.FORECAST_STEPS
    
    for i in range(config.SEQ_LENGTH, len(df) - forecast_steps):
        X.append(X_scaled[i-config.SEQ_LENGTH:i])
        y.append(y_scaled[i : i + forecast_steps])
        
    X = np.array(X)
    y = np.array(y)
    
    split_test = int(len(X) * (1 - config.TEST_SIZE))
    X_temp, X_test = X[:split_test], X[split_test:]
    y_temp, y_test = y[:split_test], y[split_test:]
    
    split_val = int(len(X_temp) * (1 - config.VAL_SIZE_REL))
    X_train, X_val = X_temp[:split_val], X_temp[split_val:]
    y_train, y_val = y_temp[:split_val], y_temp[split_val:]
    
    print("   [INFO] Se aplica Data Augmentation...")
    X_train = add_gaussian_noise(X_train)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_x, scaler_y