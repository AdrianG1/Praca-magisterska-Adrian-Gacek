# README



Poniżej zamieszczam opis struktury projektu

### Algorytmy
W projekcie  wykorzystywane są 4 algorytmy:
- **TD3**
- **SAC**
- **DQN**
- **CQL-SAC**

Każdy z nich ma skojarzone 3 pliki np.:

- **TD3.py**: Wstępne uczenie offline, zawiera konfigurację i pętlę uczenia agenta, na podstawie określonych parametrów
- **TD3_online.py**: Dostrajanie online agenta na podstawie określonych parametrów
- **TD3_optimal**: Optymalizacja przy pomocy optuna
- **TD3_optimal_online**: Optymalizacja przy pomocy optuna - dostrajanie online

Dodatkowo są pliki:
- **PID.py**: Klasa PID do generowania danych
- **utils.py**: Funkcje m. in. do wyświetlania i wczytywania danych z pliku, dyskretyzacja 
- **environmentv3.py**: środowisko oparte o TCLab
- **environmentv3_double.py**: środowisko oparte o TCLab z dwiema grzałkami
- **environmentv3_transmission.py**: środowisko oparte o TCLab dla modelu o większej bezwładności
- **environmental_test(...)**: test środowiska wersja dla akcji dyskretnych, ciągłych (SAC), środowiska z dwiema grzałkami (double), PID (PID) 
- **generating_data.py**: Skrypt generujący dane na podstawie PID i środowiska

W folderze dodatkowe_pliki znajduje się backup pracy, plik requirements.txt i plik z analizą danych.
