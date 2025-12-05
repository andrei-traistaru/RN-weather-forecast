import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

file_path = r'C:\facultate\RN\weather_prediction\Weather_Prediction_RNN\data\raw\seattle-weather.csv'

try:
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date']) 
except FileNotFoundError:
    print(f"EROARE: Nu găsesc fișierul la calea: {file_path}")
    print("Verifică dacă calea este corectă.")
    exit()

num_rows = 10000

min_precip, max_precip = df['precipitation'].min(), df['precipitation'].max()
min_temp, max_temp = df['temp_min'].min(), df['temp_max'].max()
min_wind, max_wind = df['wind'].min(), df['wind'].max()
unique_weather = df['weather'].unique() 

#generare random
new_precip = np.random.uniform(min_precip, max_precip, num_rows)
new_temp_max = np.random.uniform(min_temp, max_temp, num_rows)
new_temp_min = new_temp_max - np.random.uniform(0, 10, num_rows)
new_wind = np.random.uniform(min_wind, max_wind, num_rows)
new_weather = np.random.choice(unique_weather, num_rows)


first_date = df['date'].min()
print(f"Prima dată din setul original: {first_date}")

new_dates = [first_date - timedelta(days=x) for x in range(1, num_rows + 1)]

df_new = pd.DataFrame({
    'date': new_dates,
    'precipitation': new_precip,
    'temp_max': new_temp_max,
    'temp_min': new_temp_min,
    'wind': new_wind,
    'weather': new_weather
})

df_new = df_new.sort_values(by='date')

#concatenare
df_final = pd.concat([df_new, df], ignore_index=True)

print(f"Număr total rânduri după adăugare: {len(df_final)}")
print(f"Noua perioadă acoperită: {df_final['date'].min()} până la {df_final['date'].max()}")
print("-" * 30)

# normalizarea
cols_to_normalize = ['precipitation', 'temp_max', 'temp_min', 'wind']
scaler = MinMaxScaler(feature_range=(-1, 1))

df_final[cols_to_normalize] = scaler.fit_transform(df_final[cols_to_normalize])

print("Primele 5 rânduri din setul extins (Date sintetice din trecut):")
print(df_final.head())

output_path = r'C:\facultate\RN\weather_prediction\Weather_Prediction_RNN\data\processed\seattle-weather-past-extended-normalized.csv'
df_final.to_csv(output_path, index=False)
print(f"Fișier salvat la: {output_path}")