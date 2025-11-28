import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Adăugăm calea către config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from Weather_Prediction_RNN.config import config

def load_and_clean_data():
    """Încarcă datele brute și selectează coloanele necesare."""
    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(f"Fișierul nu a fost găsit la: {config.DATA_PATH}")
    
    df = pd.read_csv(config.DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # Ordonare cronologică (critic pentru serii de timp)
    df = df.sort_values('date')
    
    # Selectare coloane: Target + Features
    cols = [config.TARGET] + config.FEATURES
    return df[cols].values

def create_sequences(data, seq_length):
    """Transformă datele în secvențe pentru LSTM."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0]) # Coloana 0 este Target-ul (Temp Max)
    return np.array(X), np.array(y)

def get_processed_data():
    """Funcția principală apelată din main.py."""
    data = load_and_clean_data()
    
    # Scalare (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Creare secvențe
    X, y = create_sequences(scaled_data, config.SEQ_LENGTH)
    
    # Împărțire Train/Test (Cronologic)
    train_size = int(len(X) * (1 - config.TEST_SPLIT))
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test, scaler