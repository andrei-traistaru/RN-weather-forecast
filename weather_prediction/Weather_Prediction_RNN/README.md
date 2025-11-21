Proiect la disciplina: Rețele Neuronale
Student: Traistaru Andrei
Grupa: 632AB

Acest proiect are ca scop dezvoltarea unui sistem de prognoză meteo folosind tehnici moderne de Deep Learning, în special rețele neuronale recurente (RNN/LSTM).
Modelul prezice valorile meteorologice pentru următoarele 24–72 ore, utilizând date istorice despre:temperatură, presiune atmosferică, umiditate, viteza și direcția vântului.
Aplicația este implementată în Python, utilizând librăriile:
TensorFlow, Keras, PyTorch, NumPy, Pandas, Matplotlib.

2.1 Sursa datelor

Origine: Dataset public Kaggle - Seattle Weather 
Modul de achizitie: Fisier extern (CSV)
Perioada / conditiile colectarii: Ianuarie 2012 - Decembrie 2015 (Date zilnice)

2.2 Caracteristicile dataset-ului
Numar total de observatii: 1461
Numar de caracteristici (features): 6
Tipuri de date: Numerice / Temporale
Format fisiere: CSV

Caracteristica,Tip,Unitate,Descriere,Domeniu valori
date,temporal,-,Data inregistrarii,2012-01-01 - 2015-12-31
precipitation,numeric,mm,Cantitatea de ploaie cazuta,0.0 - 55.9
temp_max,numeric,Celsius,Temperatura maxima (Target),-1.6 - 35.6
temp_min,numeric,Celsius,Temperatura minima,-7.1 - 18.3
wind,numeric,m/s,Viteza medie a vantului,0.4 - 9.5
weather,categorial,-,Descriere textuala (ex: rain),"rain, sun, fog, drizzle, snow"

3.1 Statistici descriptive aplicate
Medie, mediana, deviatie standard pentru temperaturi si vant
Min–max pentru stabilirea limitelor de scalare
Distributii pe caracteristici (histograme pentru vant si precipitatii - distributii skewed)
Corelatii: Identificarea corelatiei puternice intre temp_max si temp_min

3.2 Analiza calitatii datelor
Detectarea valorilor lipsa: Dataset-ul este complet (0% valori lipsa in acest CSV specific)
Detectarea valorilor inconsistente: Verificarea temperaturilor extreme negative (nu au fost detectate erori majore)
Verificarea continuitatii temporale: Datele sunt zilnice, fara zile lipsa

3.3 Probleme identificate
Sezonalitate: Datele prezinta o sezonalitate anuala puternica (temperaturi ridicate vara, scazute iarna), necesitand o fereastra de timp (sequence length) adecvata.
Outlieri: Exista valori extreme la precipitatii in zilele cu furtuna.

4.1 Curatarea datelor
Selectie features: S-au pastrat temp_max, temp_min, wind, precipitation. Coloana weather (categoria) a fost exclusa pentru modelul de regresie numerica.

4.2 Transformarea caracteristicilor
Normalizare: Min-Max Scaling (aducerea valorilor in intervalul 0-1) pentru a asigura convergenta algoritmului LSTM.

4.3 Structurarea seturilor de date

5. Fisiere Generate in Aceasta Etapa
data/raw/seattle-weather.csv – date brute
src/preprocessing/data_prep.py – codul de preprocesare si generare secvente
config/config.py – parametrii 