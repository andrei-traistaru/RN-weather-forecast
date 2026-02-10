Descriere Generala
Acest script Python este responsabil pentru generarea datelor meteorologice sintetice utilizate pentru antrenarea si testarea modelului. Scriptul nu preia date reale de pe internet, ci simuleaza fenomene fizice folosind functii matematice pentru a crea un set de date realist.

Scriptul genereaza date orare pentru o perioada de 2 ani (aproximativ 17500 de inregistrari).

Dependinte
Pentru a rula acest script, sunt necesare urmatoarele librarii Python:

pandas: Pentru manipularea structurilor de date (DataFrame) si salvarea in format CSV.

numpy: Pentru calcule numerice si generarea de distributii statistice.

os / sys: Pentru gestionarea cailor de fisiere si directoare.

config: Un modul local care defineste calea de salvare (DATA_PATH).

Logica de Simulare Fizica
Datele nu sunt complet aleatorii; ele respecta corelatii fizice pentru a simula vremea reala:

Temperatura (Temperature):

Este variabila de baza.

Ciclu Anual: Simuleaza diferentele dintre vara si iarna folosind o unda sinusoidala cu perioada de 365 de zile.

Ciclu Diurn: Simuleaza diferentele dintre zi si noapte folosind o unda sinusoidala cu perioada de 24 de ore.

Se adauga zgomot statistic (noise) pentru variatie naturala.

Presiunea Atmosferica (Pressure):

Este corelata invers proportional cu temperatura (fizica gazelor). Cand temperatura creste, presiunea tinde sa scada usor in acest model simplificat.

Valoare de baza: 1015 hPa.

Umiditatea (Humidity):

Este corelata invers proportional cu temperatura (aerul cald poate tine mai multa apa, scazand umiditatea relativa, in timp ce racirea creste umiditatea relativa).

Vantul (WindSpeed & WindDirection):

Viteza: Generata folosind o distributie Gamma pentru a evita valorile negative si a simula rafalele.

Directia: Generata uniform intre 0 si 360 de grade.

Coloana,Descriere,Unitate (estimata)
Date,Data si ora inregistrarii (frecventa orara),YYYY-MM-DD HH:MM:SS
Temperature,Temperatura aerului,Grade Celsius
Pressure,Presiunea atmosferica,hPa
Humidity,Umiditatea relativa,Procent (%)
WindSpeed,Viteza vantului,m/s sau km/h
WindDirection,Directia vantului,Grade (0-360)