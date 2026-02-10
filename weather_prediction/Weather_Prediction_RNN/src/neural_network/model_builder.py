from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Reshape, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

import config.config as config

def build_regression_model(input_shape, output_vars=2):
    forecast_steps = config.FORECAST_STEPS if hasattr(config, 'FORECAST_STEPS') else 24
    
    model = Sequential([
        Input(shape=input_shape),
        
        # Bidirectional LSTM
        Bidirectional(LSTM(128, return_sequences=True, activation='relu')),
        Dropout(0.3),
        
        Bidirectional(LSTM(64, activation='relu')),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        
        # Output Layer
        Dense(forecast_steps * output_vars, activation='linear'),
        Reshape((forecast_steps, output_vars)) 
    ])
    
    # Huber loss pentru robustete
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss=Huber(delta=1.0), 
        metrics=['mae', 'mse']
    )
    return model

def get_callbacks():
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]