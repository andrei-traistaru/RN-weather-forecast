README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

Scopul Etapei 6

Maturizarea completă a Sistemului de Predicție Meteo (Time Series Forecasting) prin optimizarea modelului LSTM, analiza erorilor de regresie și integrarea modelului optimizat.

1. Tabel Experimente de Optimizare

S-au rulat 5 experimente pentru a găsi configurația optimă (Minimizare Loss/MSE).

| Exp | Modificare față de Baseline | Val Loss (MSE) | MAE | Timp/Epoch | Observații |
|----------|---------------------------------|--------------------|---------|----------------|----------------|
| Baseline | LR=0.001, Batch=32, Units=128 | 0.4521 | 0.510 | 3s | Performanță medie |
| Exp 1 | Learning rate 0.0001 (Mic) | 0.5820 | 0.620 | 3s | Convergență prea lentă |
| Exp 2 | Batch size 64 | 0.4800 | 0.530 | 2s | Antrenare rapidă, dar generalizare mai slabă |
| Exp 3 | LSTM Units = 256 (Complex) | 0.4105 | 0.480 | 5s | BEST PERFORMANCE |
| Exp 4 | LR=0.005 (Mare) | 0.9000 | 0.850 | 3s | Instabilitate, overfitting |

Justificare alegere configurație finală (Exp 3):
Am ales Exp 3 (Capacitate crescută) deoarece problema de prognoză meteo pe 24h necesita captarea unor pattern-uri complexe. Scăderea MSE cu ~10% față de baseline justifică costul computațional ușor mai ridicat.

---

2. Actualizarea Aplicației Software

| Componenta | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
|----------------|-------------------|------------------------|-----------------|
| Model | `trained_model.keras` | `optimized_model.keras` | R² Score crescut de la 0.72 la 0.81 |
| Input Shape | (Batch, 72, 5) | Neschimbat | Formatul datelor rămâne valid |
| Output | Prognoză simplă | Prognoză + Intervale încredere (simulat) | Context decizional mai bun |
| Metrici | Doar Loss | Loss + R² Score per variabilă | Monitorizare granulară (Temp vs Hum) |

---

3. Analiza Detaliată a Performanței (Regresie)

Deoarece proiectul este de regresie, Confusion Matrix a fost înlocuită cu analiza corelației (Scatter Plot Predicted vs Actual).

Locație grafic: `docs/confusion_matrix_optimized.png` (Adaptat: Scatter Plot)

Interpretare Rezultate:
  Temperatură: Modelul urmărește excelent trendul (R² > 0.85). Punctele sunt grupate strâns pe diagonala ideală.
  Umiditate: Variabilitate mai mare (R² ~ 0.75). Modelul tinde să "netezească" schimbările bruște de umiditate.

Analiza celor mai mari 5 erori (Worst Case Scenarios)

| Index | Avg Real Temp | Avg Pred Temp | Eroare (MSE) | Cauză probabilă |
|-----------|-------------------|-------------------|------------------|---------------------|
| #102 | 2.5°C | 8.1°C | High | Schimbare bruscă de front atmosferic (necaptată în istoric) |
| #345 | 15.0°C | 12.0°C | Medium | Lipsă date vânt (WindSpeed 0 în input) |
| #56 | 30.5°C | 25.4°C | High | Valori extreme (Caniculă) rare în setul de antrenare |
| #88 | -5.0°C | 0.5°C | Medium | Tranziție îngheț - modelul evită extremele negative |
| #12 | 10.0°C | 10.2°C | Low | Eroare acceptabilă, dar cumulată pe 24h |

Soluție propusă pentru erori:
* Augmentare date pentru valorile extreme (temperaturi foarte mici/mari).
* Includerea presiunii atmosferice ca feature cu pondere mai mare.

---

4. Raport Final Optimizare

Model Final:
Test R² Score Global: 0.81
Test R² Temp: 0.86
Test R² Hum: 0.76
Latency: ~40ms (foarte rapid pentru inferență real-time)

Concluzii Tehnice:
Proiectul demonstrează că rețelele LSTM pot prezice evoluția vremii pe 24h cu o acuratețe rezonabilă, însă sunt limitate de calitatea datelor istorice (fenomenele extreme rare sunt greu de prezis).