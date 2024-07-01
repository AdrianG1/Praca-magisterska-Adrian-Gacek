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

Dodatkowo są pliki:
- **PID.py**: Klasa PID do generowania danych
- **utils.py**: Funkcje m. in. do wyświetlania i wczytywania danych z pliku, dyskretyzacja 
- **environmentv3.py**: środowisko oparte o TCLab
- **generating_data.py**: Skrypt generujący dane na podstawie PID i środowiska

W folderze dodatkowe_pliki znajduje się backup pracy, plik requirements.txt i plik z analizą danych.
