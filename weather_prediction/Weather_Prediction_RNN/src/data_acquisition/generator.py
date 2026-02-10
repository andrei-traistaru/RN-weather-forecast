import pandas as pd
import numpy as np
import os
import sys

# Setup cai relative
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import config pentru a sti unde salvam
import config.config as config

def generate_weather_data():
    print("Generating physics-based simulation data...")
    np.random.seed(42)
    
    # Generam mai multe date pentru antrenament robust (ex: 2 ani)
    num_hours = 24 * 365 * 3 
    date_range = pd.date_range(start='2012-01-01', periods=num_hours, freq='H')

    # 1. TEMPERATURA (Simulare fizica: ciclu diurn + anual)
    t = np.linspace(0, 4*np.pi, num_hours) # Trend general
    # Ciclu anual (iarna frig, vara cald)
    yearly_cycle = 10 * np.sin(2 * np.pi * np.arange(num_hours) / (24*365)) 
    # Ciclu zilnic (ziua cald, noaptea frig)
    daily_cycle = 5 * np.sin(2 * np.pi * np.arange(num_hours) / 24)
    
    noise = np.random.normal(0, 1, num_hours)
    temperature = 12 + yearly_cycle + daily_cycle + noise

    # 2. PRESIUNEA (Corelata fizic invers cu temperatura + noise)
    pressure = 1015 - (temperature - 12) * 0.3 + np.random.normal(0, 2, num_hours)

    # 3. UMIDITATEA (Invers proportionala cu temp)
    humidity = 70 - (temperature - 12) * 2 + np.random.normal(0, 5, num_hours)
    humidity = np.clip(humidity, 20, 100)

    # DataFrame
    df = pd.DataFrame({
        'Date': date_range,
        'Temperature': np.round(temperature, 2),
        'Pressure': np.round(pressure, 2),
        'Humidity': np.round(humidity, 2),
        'WindSpeed': np.round(np.random.gamma(2, 2, num_hours), 2),
        'WindDirection': np.round(np.random.uniform(0, 360, num_hours), 0)
    })

    # Asiguram ca folderul exista
    os.makedirs(os.path.dirname(config.DATA_PATH), exist_ok=True)
    
    df.to_csv(config.DATA_PATH, index=False)
    print(f"✅ Data generation complete: {len(df)} samples saved to {config.DATA_PATH}")

if __name__ == "__main__":
    generate_weather_data()