# Raport Etapa 5: Antrenarea si Evaluarea Retelei Neuronale

**Student:** Traistaru Andrei  
**Grupa:** 632AB  
**Disciplina:** Retele Neuronale  

---

## 1. Descrierea Modelului
In aceasta etapa, am antrenat o retea neuronala recurenta (LSTM) pentru a rezolva o problema de **regresie**. Modelul prezice valori numerice continue pentru **Temperatura** si **Umiditate** peste un orizont de timp de **24 de ore**, bazandu-se pe un istoric de 72 de ore.

* **Tip problema:** Regresie Multi-Output (2 variabile tinta)
* **Arhitectura:** Stacked LSTM cu straturi Dense si Dropout
* **Input (Features):** `Temperature`, `Pressure`, `Humidity`, `WindSpeed`, `WindDx`, `WindDy` (6 variabile)
* **Output:** `Temperature`, `Humidity` (2 variabile)

---

## 2. Pregatirea Datelor si Split Stratificat

Dataset-ul a fost impartit respectand proportiile cerute, asigurand ca datele de test nu au fost vazute niciodata de model in timpul antrenarii.

| Set de Date | Procent | Rol |
|:-----------:|:-------:|:----|
| **Train** | **70%** | Folosit pentru ajustarea greutatilor (backpropagation). S-a aplicat augmentare cu zgomot Gaussian. |
| **Validation** | **15%** | Folosit pentru monitorizarea `val_loss` si declansarea Early Stopping / ReduceLROnPlateau. |
| **Test** | **15%** | Folosit exclusiv pentru calculul metricilor finale (R² Score, MAE) dupa antrenare. |

**Tehnici aplicate:**
* **Normalizare:** MinMaxScaler (0-1) aplicat separat pe Features si Targets.
* **Secventiere:** Sliding Window cu lungimea de 72 pasi temporali (ore).
* **Augmentare:** Adaugarea de zgomot Gaussian ($\mu=0, \sigma=0.01$) pe datele de antrenament pentru simularea erorilor de senzor.

---

## 3. Justificare Hiperparametri (OBLIGATORIU)

Tabelul de mai jos detaliaza configuratia finala a modelului si motivele alegerii fiecarui parametru.

| **Hiperparametru** | **Valoare Aleasa** | **Justificare** |
|--------------------|-------------------|-----------------|
| **Learning rate** | `0.001` | Valoare standard pentru optimizatorul Adam. Asigura o convergenta stabila, evitand oscilatiile mari ale gradientului pe datele normalizate. |
| **Batch size** | `32` | Compromis optim intre utilizarea memoriei si stabilitatea gradientului. Avand un dataset de dimensiuni medii (~1000 secvente), 32 permite actualizari frecvente ale greutatilor fara a introduce prea mult zgomot. |
| **Number of epochs** | `50` | Numar suficient pentru convergenta. S-a folosit impreuna cu `EarlyStopping` (patience=5), deci antrenarea se opreste automat daca modelul nu mai invata, prevenind overfitting-ul. |
| **Optimizer** | `Adam` | Adaptive Moment Estimation este standardul in industrie pentru RNN/LSTM, deoarece ajusteaza rata de invatare individual pentru fiecare parametru, gestionand eficient "sparse gradients". |
| **Loss function** | `MSE` (Mean Squared Error) | Fiind o problema de regresie, MSE penalizeaza erorile mari mai drastic (prin ridicare la patrat) decat MAE, fortand modelul sa fie precis pe valorile extreme. |
| **Activation functions** | `ReLU` (LSTM/Dense), `Linear` (Output) | `ReLU` este eficient computational si evita "vanishing gradient". Stratul de iesire are activare `Linear` pentru a putea prezice orice valoare reala din intervalul scalat. |
| **Dropout Rate** | `0.2` | Regularizare aplicata dupa straturile LSTM pentru a preveni memorarea secventelor de antrenament (overfitting). |
| **LR Scheduler** | `ReduceLROnPlateau` | Daca `val_loss` stagneaza timp de 3 epoci, rata de invatare se reduce la jumatate, permitand modelului sa faca ajustari fine ("fine-tuning") spre minimul global. |

---

## 4. Rezultate si Metrici (Test Set)

Evaluarea finala s-a realizat pe setul de TEST (cele 15% din date pastrate intacte). Deoarece este o problema de regresie, "Acuratetea" este masurata prin coeficientul de determinare **R² Score**.

### Metrici Obtinute:
* **R² Score (Temperatura):** `0.9245` (Target atins: > 0.90) ✅
* **R² Score (Umiditate):** `0.8910`
* **Mean Absolute Error (MAE) Scalat:** `0.065`

### Interpretare Grafica:
* **Grafic Loss:** Curbele `loss` (train) si `val_loss` (validare) descresc si se stabilizeaza impreuna, ceea ce indica o invatare corecta fara overfitting major.
* **Imagine generata:** `docs/loss_curve.png` (Curba de pierdere) si `docs/predictie_rezultat.png` (Comparare Predictie vs Realitate).

---

## 5. Analiza Erorilor

Modelul a demonstrat o capacitate ridicata de generalizare, insa exista mici discrepante:
1.  **Varfuri extreme:** Modelul tinde sa "netezeasca" valorile extreme (foarte ridicate sau foarte scazute), un comportament tipic pentru functia de pierdere MSE care mediaza erorile.
2.  **Decalaj temporal:** In anumite secvente, predictia pare usor decalata (lag), fenomen cauzat de natura recurenta a retelei care se bazeaza puternic pe ultimele valori vazute (t-1).

**Concluzie:**
Cu un scor R² de peste 0.90 pentru temperatura, modelul indeplineste criteriile de performanta stabilite pentru aceasta etapa si este pregatit pentru integrarea in interfata Web.

---

### Instructiuni de Rulare pentru Reproducere:

1.  Generare date: `python generate_data.py`
2.  Antrenare: `python main.py`
3.  Vizualizare rezultate: Deschideti folderul `/docs`