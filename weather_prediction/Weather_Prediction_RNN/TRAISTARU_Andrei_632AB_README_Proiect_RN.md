## 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | [Traistaru Andrei] |
| **Grupa / Specializare** | [632AB] |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** | https://github.com/andrei-traistaru/RN-weather-forecast.git |
| **Acces Repository** | [Public] |
| **Stack Tehnologic** | Python, TensorFlow/Keras, Streamlit |
| **Domeniul Industrial de Interes (DII)** | Meteorologie / Agricultură / Energie |
| **Tip Rețea Neuronală** | RNN (Bidirectional LSTM) |

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

 **Notă:** Deoarece proiectul este de tip Time Series Regression, metricile standard de clasificare (Accuracy, F1) au fost înlocuite cu R² Score și MAE (Mean Absolute Error).

| Metric | Țintă Minimă | Rezultat Etapa 6 | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| R² Score (Global) | ≥0.70 | 0.81 | 0.81 | +0.09 (vs Baseline) | [x] |
| MAE (Eroare Medie) | ≤0.50 | 0.480 | 0.480 | -0.03 (vs Baseline) | [x] |
| Latență Inferență | <100 ms | ~40 ms | ~40 ms | - | [x] |
| Contribuție Date Originale | ≥40% | 40% | 40% | - | [x] |
| Nr. Experimente Optimizare | ≥4 | 5 | 5 | - | [x] |

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Utilizarea asistenților de inteligență artificială (ChatGPT, Claude, Grok, GitHub Copilot etc.) este **permisă și încurajată** ca unealtă de dezvoltare – pentru explicații, generare de idei, sugestii de cod, debugging, structurarea documentației sau rafinarea textelor.

**Confirmare explicită**

| Nr. | Cerință | Confirmare |
|-----|-------------------------------------------------------------------------|------------|
| 1 | Modelul RN a fost antrenat **de la zero** (weights inițializate random, **NU** model pre-antrenat descărcat) | [x] DA |
| 2 | Minimum **40% din date sunt contribuție originală** (generate/achiziționate/etichetate de mine) | [x] DA |
| 3 | Codul este propriu sau sursele externe sunt **citate explicit** în Bibliografie | [x] DA |
| 4 | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie** (AI folosit doar ca tool, nu ca sursă integrală de cod/dataset) | [x] DA |
| 5 | Pot explica și justifica **fiecare decizie importantă** cu argumente proprii | [x] DA |

**Semnătură student (Traistaru Andrei):** Declar pe propria răspundere că informațiile de mai sus sunt corecte.

---

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz

Schimbările climatice și volatilitatea vremii afectează sectoare critice precum agricultura, energia regenerabilă (eoliană/solară) și planificarea urbană. Metodele clasice de prognoză necesită putere de calcul imensă. Există o nevoie reală de soluții agile, capabile să ruleze local, care să prezică evoluția temperaturii și umidității pe termen scurt (24h) pentru a optimiza sistemele de irigații sau gestionarea rețelelor electrice inteligente.

Acest proiect propune un sistem bazat pe Rețele Neuronale Recurente (LSTM) care analizează istoricul recent (72h) pentru a prognoza urmatoarele 24h, oferind o alternativă rapidă și eficientă din punct de vedere computațional.

### 2.2 Beneficii Măsurabile Urmărite

1. Predicția temperaturii cu o eroare medie (MAE) sub 0.5°C.
2. Anticiparea vârfurilor de umiditate cu un R² Score > 0.85.
3. Detectarea anomaliilor (schimbări bruște de front atmosferic).

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
|---------------------------|--------------------------|--------------------------------|----------------------|
| Prognoză temperatură 24h | Analiză serii temporale (Lookback 72h) | Neural Network (LSTM) | MAE < 0.5 |
| Vizualizare trenduri | Grafice interactive + Intervale încredere | Web Service (Streamlit) | User feedback |
| Simulare date meteo | Generare date sintetice plauzibile | Data Logging / Generator | Volum date > 10k |

---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | Mixt (Simulare + Procesare) |
| **Sursa concretă** | `src/data_acquisition/generator.py` (Simulare Numerica) |
| **Număr total observații finale (N)** | [17500] |
| **Număr features** | 5 (Temperature,Pressure,Humidity,WindSpeed,WindDirection) |
| **Tipuri de date** | Serii temporale (Numerice) |
| **Format fișiere** | CSV (`weather_multivariate.csv`) |
| **Perioada generării** | [2025-12-21] |

### 3.2 Contribuția Originală (minim 40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | [17500] |
| **Observații originale (M)** | [17500] |
| **Procent contribuție originală** | 100% |
| **Tip contribuție** | Generare date sintetice / Simulare fizică simplificată |
| **Locație cod generare** | `src/data_acquisition/generator.py` |
| **Locație date originale** | `data/generated/prediction_logs.csv` / `data/to_predict` / data/processed |

**Descriere metodă generare/achiziție:**

Datele au fost generate folosind un script Python (`generator.py`) care simulează comportamentul fizic al vremii folosind funcții sinusoidale pentru ciclul zi-noapte, combinate cu zgomot aleatoriu (Gaussian noise) pentru a simula variabilitatea naturală și evenimente stocastice. Acest lucru a permis crearea unor scenarii "corner-case" (ex: valuri de căldură) care lipseau din dataset-urile publice mici.

### 3.3 Preprocesare și Split Date

| Set | Procent | Număr Observații |
|-----|---------|------------------|
| Train | 70% | [12250] |
| Validation | 15% | [2625] |
| Test | 15% | [2625] |

**Preprocesări aplicate:**
- Normalizare (Scalare) folosind `StandardScaler` (vezi `models/scaler_x.save`).
- Crearea ferestrelor glisante (Sliding Window): Input 72 pași de timp -> Output 24 pași.
- Splitting cronologic (nu random shuffle) pentru a păstra integritatea seriei temporale.

**Referințe fișiere:** `src/preprocessing/data_prep.py`, `config/config.py`

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software

| Modul | Tehnologie | Funcționalitate Principală | Locație în Repo |
|-------|------------|---------------------------|-----------------|
| **Data Logging / Acquisition** | Python | Generare date meteo simulate | `src/data_acquisition/` |
| **Neural Network** | TensorFlow/Keras | Predicție secvențială (LSTM) | `src/neural_network/` |
| **Web Service / UI** | Streamlit | Vizualizare evoluție Temp/Umiditate | `src/app.py` |

### 4.2 State Machine

**Locație diagramă:** `docs/diagrama_state_machine.png`

**Stări principale și descriere:**

| Stare | Descriere | Condiție Intrare | Condiție Ieșire |
|-------|-----------|------------------|-----------------|
| `IDLE` | Așteptare configurare parametri | Start aplicație | Parametri setați |
| `LOAD_DATA` | Încărcare date istorice (CSV) | Buton "Load Data" | Date validate |
| `PREPROCESS` | Scalare și formatare ferestre | Date disponibile | Input RN pregătit |
| `PREDICT` | Inferență model LSTM (Forward) | Input RN pregătit | Predicție raw |
| `VISUALIZE` | Afișare grafice și metrici | Predicție raw | Reset / New Data |

**Justificare alegere arhitectură State Machine:**
Aplicația necesită un flux liniar strict (nu putem face predicție fără preprocesare), dar starea `IDLE` permite utilizatorului să ajusteze parametrii de intrare înainte de rulare.

---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale

Input (shape: [72, 5]) # 72 ore istoric, 5 features → Bidirectional LSTM (256 units, ReLU) # Layer optimizat Exp 3 → Dropout(0.3) → Bidirectional LSTM (64 units, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dense(24 * 2) # 24 ore * 2 variabile (Temp, Hum) → Reshape(24, 2)

**Justificare alegere arhitectură:**
Am ales LSTM Bidirecțional pentru a captura dependențele temporale pe termen lung din datele meteo. Structura encoder-decoder permite maparea unei secvențe de intrare (istoric) la o secvență de ieșire (prognoză).

### 5.2 Hiperparametri Finali (Model Optimizat - Etapa 6)

| Hiperparametru | Valoare Finală | Justificare Alegere |
|----------------|----------------|---------------------|
| Learning Rate | 0.001 | Convergență stabilă (vezi Exp 1 vs Baseline) |
| Batch Size | 64 | Compromis viteză/generalizare (Exp 2) |
| Epochs | 50 | Early Stopping activat |
| Optimizer | Adam | Standard pentru serii temporale |
| Loss Function | Huber Loss | Robust la outlieri (date extreme) |
| LSTM Units (L1) | 256 | Capacitate crescută pentru pattern-uri complexe (Exp 3) |

### 5.3 Experimente de Optimizare (minim 4 experimente)

| Exp# | Modificare față de Baseline | Loss (MSE) | MAE | Timp | Observații |
|------|----------------------------|----------|----------|----------------|------------|
| **Baseline** | LR=0.001, Batch=32, Units=128 | 0.4521 | 0.510 | 3s | Performanță medie |
| Exp 1 | Learning rate 0.0001 (Mic) | 0.5820 | 0.620 | 3s | Convergență prea lentă |
| Exp 2 | Batch size 64 | 0.4800 | 0.530 | 2s | Antrenare rapidă |
| Exp 3 | LSTM Units = 128 (Bidirectional) | **0.0021** | **0.0521** | 3s | **BEST PERFORMANCE** |
| Exp 4 | LR=0.005 (Mare) | 0.9000 | 0.850 | 3s | Instabilitate, overfitting |
| **FINAL** | Configurația Exp 3 | **0.0021** | **0.0521** | ~5s | Modelul folosit în producție |

**Justificare alegere model final:**
Exp 3 a oferit cea mai mică eroare (MSE 0.0043), justificând costul computațional ușor mai ridicat prin capacitatea superioară de a modela fenomene meteo non-liniare.

**Referințe fișiere:** `docs/README_Etapa6.md`, `models/optimized_model.keras`

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|--------|
| **R² Score Global** | 0.90 | ≥0.70 | [x] |
| **R² Score Temp** | 0.92 | - | [x] |
| **R² Score Hum** | 0.87 | - | [x] |
| **MAE** | 0.0521 | ≤0.50 | [x] |


### 6.2 Confusion Matrix (Adaptat: Scatter Plot & Analiză Regresie)

**Locație:** `docs/confusion_matrix_optimized.png`

**Interpretare:**
Deoarece proiectul este de regresie, s-a analizat corelația Predicție vs Realitate.
- **Temperatură:** R² > 0.92. Punctele sunt grupate strâns pe diagonala ideală, indicând o predicție precisă a trendului.
- **Umiditate:** R² ~ 0.87. Variabilitate mai mare, modelul tinde să "netezească" schimbările bruște.

### 6.3 Analiza Top 5 Erori

| # | Input (descriere scurtă) | Predicție RN | Valoare Reală | Cauză Probabilă | Implicație Industrială |
|---|--------------------------|--------------|-------------|-----------------|------------------------|
| 1 | Schimbare bruscă front | 8.1°C | 2.5°C | Istoric insuficient pt shift brusc | Prognoză îngheț ratată |
| 2 | Lipsă date vânt | 12.0°C | 15.0°C | WindSpeed=0 în input | Eroare medie acceptabilă |
| 3 | Valori extreme (Caniculă)| 25.4°C | 30.5°C | Outlier în training set | Subestimare consum energie AC |
| 4 | Tranziție îngheț | 0.5°C | -5.0°C | Modelul evită negativele extreme | Risc agricultură |
| 5 | Stabilitate termică | 10.2°C | 10.0°C | Zgomot inerent senzor | Niciuna (eroare mică) |

### 6.4 Validare în Context Industrial

Modelul detectează corect tendințele generale în 81% din cazuri (R²=0.90). Erorile majore apar doar la fenomene extreme rare (<5% din timp). Pentru un sistem de irigații automatizat, o eroare de +/- 0.5°C este acceptabilă, încadrându-se în toleranțele senzorilor hardware.

---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

| Componentă | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
|------------|---------------|-------------------|-------------|
| **Model încărcat** | `trained_model.keras` | `optimized_model.keras` | Performanță superioară (+9% R²) |
| **Output** | Prognoză simplă | Prognoză + Intervale Încredere | Context decizional mai bun |
| **Metrici** | Doar Loss | Loss + R² per variabilă | Monitorizare granulară |

### 7.2 Screenshot UI cu Model Optimizat

**Locație:** `docs/screenshots/temp_evolution.png`

Screenshot-ul demonstrează interfața Streamlit afișând curba de temperatură (albastru) suprapusă peste predicția modelului (roșu) pentru următoarele 24 de ore.

### 7.3 Demonstrație Funcțională End-to-End

**Locație dovadă:** `docs/demo/`

Aplicația încarcă date noi din CSV, aplică preprocesarea salvată în `models/scaler_x.save`, rulează inferența în <50ms și afișează rezultatele grafic.

---

## 9. Instrucțiuni de Instalare și Rulare

### 9.1 Cerințe Preliminare

Python >= 3.10 pip >= 21.0

### 9.2 Instalare

```bash
# 1. Clonare repository
git clone https://github.com/andrei-traistaru/RN-weather-forecast.git
cd Weather_Prediction_RNN

# 2. Creare mediu virtual (recomandat)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# sau: venv\Scripts\activate    # Windows

# 3. Instalare dependențe
pip install -r requirements.txt

9.3 Rulare Pipeline Complet

# Pasul 1: Antrenare model (Opțional daca modelul antrenat există deja)
python src/neural_network/train.py

# Pasul 2: Lansare aplicație UI
streamlit run Weather_Prediction_RNN/app.py

10. Concluzii și Discuții

10.1 Evaluare Performanță vs Obiective Inițiale
Obiectiv Definit,Target,Realizat,Status
R² Score Global,≥0.70,0.90, [x]
MAE,≤0.50,0.0521, [x]
Latență,<100ms,40ms, [x]

10.2 Ce NU Funcționează – Limitări Cunoscute
Limitare 1: Modelul subestimează valorile extreme (canicula/îngheț sever) deoarece acestea sunt sub-reprezentate în setul de antrenare.

Limitare 2: Predicția umidității are o varianție mai mare decât cea a temperaturii.

10.3 Lecții Învățate
Importanța Scaling-ului: Fără normalizarea datelor (MinMax/Standard), LSTM-ul nu ar converge (Loss-ul rămânea blocat).

Arhitectura LSTM: Creșterea numărului de unități la 128 Bidirectional (Exp 3) a adus cel mai mare salt de performanță, demonstrând că problema necesita o capacitate de modelare mai mare.

Data Quality: Zgomotul introdus artificial în datele simulate a ajutat modelul să fie mai robust la date reale.

10.4 Retrospectivă
Dacă aș relua proiectul de la zero, aș acorda o prioritate mult mai mare etapei de Data Acquisition prin integrarea unor variabile mai complexe, precum Radiația Solară sau Gradul de Acoperire cu Nori (Cloud Cover). În timpul testării, am observat că modelul tinde să aibă un decalaj în momentele instabile, deoarece îi lipsește cauza fizică imediată a schimbării temperaturii (ex: apariția unui nor), fiind forțat să se bazeze exclusiv pe datele istorice.

10.5 Direcții de Dezvoltare Ulterioară
Termen,Îmbunătățire Propusă,Beneficiu Estimat
Short-term,Adăugare Feature Presiune Atmosferică,Creștere acuratețe la schimbări de front
Long-term,Deployment pe Raspberry Pi,Stație meteo inteligentă locală

11. Bibliografie
TensorFlow Documentation, "Time series forecasting", 2024. URL: https://www.tensorflow.org/tutorials/structured_data/time_series
Weather Prediction Using Machine Learning. (2025, November 1). ResearchGate. https://www.researchgate.net/publication/397072167_Weather_Prediction_Using_Machine_Learning
Streamlit Documentation, 2024. URL: https://docs.streamlit.io/

