1. Arhitectura Modelului (model_builder.py)
Descriere Generala
Fisierul model_builder.py contine functiile necesare pentru construirea modelului de regresie bazat pe retele neuronale recurente (RNN) de tip LSTM (Long Short-Term Memory).

Arhitectura Detaliata
Modelul este construit secvential si include urmatoarele componente:

Input Layer: Primeste datele de intrare sub forma (timesteps, features).

Bidirectional LSTM (128 unitati): Primul strat recurent care proceseaza secventa in ambele directii pentru a capta modele temporale complexe. Foloseste activare 'relu'.

Dropout (0.3): Pentru prevenirea overfitting-ului (supra-ajustarii).

Bidirectional LSTM (64 unitati): Al doilea strat recurent pentru rafinarea caracteristicilor extrase.

Dense Layer (64 unitati): Strat complet conectat pentru interpretarea caracteristicilor.

Output Layer: Strat dens final care proiecteaza predictiile.

Reshape Layer: Formateaza iesirea in dimensiunea (pasi_prognoza, variabile_output) pentru a corespunde formatului cerut (multistep output).

Configurare Compilare
Optimizator: Adam (cu rata de invatare definita in config).

Functie de Pierdere (Loss Function): Huber Loss (delta=1.0). Aceasta a fost aleasa pentru robustetea sa in fata outlier-ilor (valori extreme), fiind un compromis intre MAE (Mean Absolute Error) si MSE (Mean Squared Error).

Metrici: Monitorizeaza MAE si MSE in timpul antrenarii.

Callbacks (Mecanisme de Control)
Functia get_callbacks() returneaza doua mecanisme esentiale:

EarlyStopping: Opreste antrenarea daca val_loss (pierderea pe validare) nu se imbunatateste timp de 6 epoci, restaurand cei mai buni parametri.

ReduceLROnPlateau: Reduce rata de invatare la jumatate daca performanta stagneaza timp de 3 epoci, permitand o convergenta fina.

2. Scriptul de Antrenare (train.py)
Descriere Generala
Fisierul train.py este scriptul principal de executie care controleaza intregul flux de lucru: incarcarea datelor preprocesate, construirea modelului, antrenarea si salvarea rezultatelor.

Incarcare Date: Apeleaza functia get_processed_data din src.preprocessing.data_prep. Se asteapta la seturi de date pentru antrenament, validare si testare, plus obiectele de scalare (scalers).

Initializare Model: Construieste modelul folosind model_builder.py adaptat la dimensiunile datelor de intrare.

Antrenare (Fitting): Ruleaza procesul de invatare pe seturile de antrenament si validare, utilizand numarul de epoci si marimea lotului (batch size) din configuratie.

Evaluare:

Genereaza predictii pe setul de test.

Inverseaza scalarea pentru a aduce datele la valorile reale (grade Celsius, procente umiditate).

Calculeaza scorul R2 (coeficientul de determinare) pentru Temperatura si Umiditate.

Salvare Rezultate:

Grafic: Salveaza curba de invatare (Loss vs Val Loss) in folderul de documentatie.

Model: Salveaza modelul antrenat sub numele optimized_model.keras.

Scalere: Salveaza obiectele scaler_x si scaler_y folosind joblib pentru a fi folosite ulterior la inferenta (predictie).

Dependinte
tensorflow: Pentru operatiile de deep learning.
sklearn: Pentru metrici de evaluare (r2_score).
joblib: Pentru salvarea scalerelor.
matplotlib: Pentru generarea graficelor de performanta.
numpy: Pentru manipularea matricelor.