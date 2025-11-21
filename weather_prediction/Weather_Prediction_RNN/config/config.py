import os

# Căi către date
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'seattle-weather.csv')
DATA_PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed')

# Parametrii datelor (Etapa 3)
FEATURES = ['temp_max', 'temp_min', 'precipitation', 'wind']
TARGET = 'temp_max'
SEQ_LENGTH = 30  # Fereastra glisantă (zile anterioare)

# Parametrii modelului RNN-LSTM (Etapa 4)
TEST_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 25
LSTM_UNITS = 64
LEARNING_RATE = 0.001