Descriere Generala
Acest modul este responsabil pentru transformarea datelor brute (din format CSV) intr-un format numeric optimizat pentru antrenarea retelei neuronale. Scriptul gestioneaza curatarea datelor, ingineria caracteristicilor (feature engineering), normalizarea si impartirea in seturi de antrenament, validare si testare.

Functii Principale
1. add_gaussian_noise(X_data, noise_level=0.01)
Scop: Adauga un zgomot aleator (distributie Gaussiana) peste datele de intrare.

Utilitate: Ajuta modelul sa devina mai robust si sa evite overfitting-ul prin expunerea acestuia la variatii mici ale datelor.

2. get_processed_data()
Functia principala care controleaza intregul flux de preprocesare. Pasii executati sunt urmatorii:

A. Incarcare si Curatare (Smoothing)
Verifica existenta fisierului de date definit in configuratie.

Aplica o medie mobila (rolling mean) cu o fereastra de 3 ore asupra coloanei Humidity pentru a elimina fluctuatiile bruste si zgomotul de masurare.

B. Feature Engineering (Generare de Caracteristici)
Scriptul nu foloseste doar datele brute, ci calculeaza derivate fizice si matematice pentru a ajuta reteaua neuronala:

Ciclicitatea Timpului:

Converteste data si ora in semnale sinusoidale si cosinusoidale.

Day sin/cos: Reprezinta ciclul zi(24 ore).

Year sin/cos: Reprezinta ciclul anual (anotimpuri).

Acest lucru permite modelului sa inteleaga continuitatea timpului (ora 23:59 este aproape de 00:00).

Vectorizarea Vantului:

Converteste directia vantului (grade 0-360) in componente vectoriale (WindDx, WindDy) folosind sinus si cosinus. Aceasta rezolva problema discontinuitatii dintre 359 si 0 grade.

Calcul Fizic:

Calculeaza DewPoint folosind Formula Magnus, bazata pe temperatura si umiditate. Aceasta este o caracteristica critica pentru predictia fenomenelor meteorologice (ceata, ploaie).

C. Normalizare (Scaling)
Utilizeaza MinMaxScaler din biblioteca sklearn.

Scaleaza toate caracteristicile (features) si tintele (targets) in intervalul [0, 1] pentru a asigura stabilitatea numerica a retelei neuronale.

Se creeaza doua scalere separate: scaler_x (pentru input) si scaler_y (pentru output), necesare ulterior pentru inversarea predictiilor.

D. Crearea Secventelor (Sliding Window)
Transforma datele tabulare in secvente temporale 3D necesare pentru straturile LSTM:

Format: (Numar_Esantioane, Lungime_Secventa, Numar_Features)

Fereastra glisanta parcurge datele, creand perechi de input (istoric) si output (viitor, definit de forecast_steps).

E. Impartirea Datelor (Train/Val/Test Split)
Imparte setul de date in trei categorii, respectand ordinea cronologica (fara amestecare aleatorie, pentru a pastra structura temporala):

Test Set: Ultimele date din serie (15%).

Validation Set: (15%) din datele totale, folosite pentru monitorizarea antrenarii.

Train Set: Restul datelor, folosite pentru invatarea efectiva (75%).

La final, se aplica add_gaussian_noise doar pe setul de antrenament (X_train).

Valori Returnate
Functia returneaza un tuplu continand 8 elemente esentiale pentru procesul de antrenare:

X_train: Datele de intrare pentru antrenament (cu zgomot adaugat).

y_train: Tintele pentru antrenament.

X_val: Datele de intrare pentru validare.

y_val: Tintele pentru validare.

X_test: Datele de intrare pentru testare finala.

y_test: Tintele pentru testare.

scaler_x: Obiectul de scalare pentru features (necesar pentru inferenta viitoare).

scaler_y: Obiectul de scalare pentru target (necesar pentru denormalizarea predictiilor).

Dependinte
pandas: Manipulare DataFrame si calcule rolling mean.
numpy: Calcule matematice (sin, cos, log, exponentiale).
sklearn: Preprocesare (MinMaxScaler).
config: Fisierul de configurare al proiectului.