# MnistRecognizer - Instrukcja obsługi
## config.py
Tutaj ustawia się ścieżkę do pliku PATH, nazwę modelu MARK oraz ilość epok EPOCHS.
## resources.py
Zawiera funkcję generującą model i funkcje walidujące. Tutaj ustawia się architekturę sieci.
## training.py
Odpowiada za trenowanie modelu. Po każdej epoce liczy skuteczność oraz generuje wykres funkcji straty dla zbioru treningowego i walidacyjnego, naukę należy przerwać kiedy wykresy zaczynają się rozchodzić. Najlepsze modele zapisują się na bieżąco w lokalizacji PATH więc wystarczy zatrzymać skrypt.
## review.py
Należy uruchomić dla wytrenowanego modelu. Generuje macierz pomyłek oraz zapisuje błędnie zklasyfikowane obrazy w lokalizacji PATH/wrong.
