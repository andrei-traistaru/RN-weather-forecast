from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_lstm_model(input_shape, learning_rate=0.001):
    """Construiește și compilează modelul LSTM."""
    model = Sequential([
        # Strat 1 LSTM
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        
        # Strat 2 LSTM
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        
        # Output
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model