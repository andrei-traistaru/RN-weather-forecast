import os

# Setari Generale
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'weather_multivariate.csv')
MODELS_PATH = os.path.join(BASE_DIR, 'models')
DOCS_PATH = os.path.join(BASE_DIR, 'docs')

# Parametri Model
SEQ_LENGTH = 72       # Istoric: 72 ore
FORECAST_STEPS = 24   # Prognoza: 24 ore
PREDICTION_HORIZON = 24 
TEST_SIZE = 0.15
VAL_SIZE_REL = 0.15   # Din setul de antrenare ramas
BATCH_SIZE = 32
EPOCHS = 50           # Numar maxim de epoci
LEARNING_RATE = 0.001
LSTM_UNITS = 64

# Tinte
TARGETS = ['Temperature', 'Humidity']