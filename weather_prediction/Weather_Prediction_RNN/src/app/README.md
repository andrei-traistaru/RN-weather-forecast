Descriere Generala
Fisierul app.py reprezinta UI-ul (Frontend) sistemului de predictie meteorologica. Este construit folosind libraria Streamlit si permite utilizatorilor sa interactioneze cu modelul de retea neuronala antrenat (LSTM) pentru a vizualiza prognoze in timp real sau pe date istorice.

Aplicatia nu doar afiseaza date, ci reproduce intregul flux de preprocesare "on-the-fly" (in timp real) pentru a transforma datele brute in formatul acceptat de model.

Functionalitati Principale
1. Gestionarea Resurselor (Caching)
Functia load_resources utilizeaza decoratorul @st.cache_resource pentru a optimiza performanta. Aceasta incarca o singura data in memorie:

Modelul: Cauta optimized_model.keras. Daca nu exista, incearca sa incarce trained_model.keras.

Scalerele: Incarca scaler_x.save si scaler_y.save (folosind joblib) pentru a normaliza datele de intrare si a denormaliza predictiile.

2. Preprocesare in Timp Real
Functia prepare_input este critica. Deoarece modelul asteapta date procesate (feature engineering), aceasta functie replica exact logica din data_prep.py pentru orice date noi introduse:

Genereaza componentele ciclice temporale (Day sin/cos, Year sin/cos).

Transforma directia vantului in vectori (WindDx, WindDy).

Calculeaza Punctul de Roua (Dew Point) folosind formula Magnus.

Verifica existenta coloanelor obligatorii: Temperature, Pressure, Humidity, WindSpeed, WindDirection, Date.

3. Selectia Surselor de Date
Utilizatorul poate alege din bara laterala (Sidebar) intre:

Simulare (Date Istorice): Incarca fisierul generat anterior (weather_sim.csv) pentru a valida modelul pe date cunoscute.

Incarca Fisier CSV Propriu: Permite incarcarea unui set de date extern pentru predictie.

4. Post-procesare si Calibrare
Dupa ce modelul genereaza predictia, aplicatia aplica o tehnica de Anchor Alignment:

Calculeaza diferenta (bias) dintre ultima valoare reala cunoscuta si prima valoare prezisa de model.

Ajusteaza toata seria de predictii cu aceasta diferenta.

Aceasta asigura continuitatea grafica intre istoric si prognoza, eliminand salturile artificiale.

5. Vizualizare si Logging
Grafice: Genereaza grafice Matplotlib care combina istoricul (72h), prognoza AI (24h) si realitatea (Ground Truth - daca exista) pe aceeasi axa.

Metrici: Calculeaza eroarea medie absoluta (MAE) daca exista date reale pentru comparatie.

Logging: Functia log_prediction salveaza automat fiecare predictie rulata in fisierul data/generated/prediction_logs.csv.

Instructiuni de Utilizare

streamlit run src/app/app.py
streamlit run app.py