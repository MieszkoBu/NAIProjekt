# AI Recipe Generator

Generator przepisów kulinarnych wykorzystujący sztuczną inteligencję do rozpoznawania składników ze zdjęć i generowania przepisów.

## 🌟 Funkcjonalności

- Rozpoznawanie składników ze zdjęć przy użyciu modelu AI
- Generowanie przepisów z wykorzystaniem GPT-3.5
- Możliwość tworzenia spersonalizowanych przepisów
- Analiza wartości odżywczych
- Sprawdzanie zgodności z różnymi dietami
- Skalowanie liczby porcji
- Generowanie wariantów przepisu (zdrowsze, szybsze, budżetowe)
- Historia przepisów z możliwością oceniania
- Eksport przepisów do PDF

## 🚀 Instalacja

1. Sklonuj repozytorium:
```bash
git clone https://github.com/twoj-username/NAIProjekt.git
cd NAIProjekt
```

2. Zainstaluj Git LFS (jeśli nie jest zainstalowany):
```bash
git lfs install
```

3. Utwórz i aktywuj wirtualne środowisko:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# lub
venv\Scripts\activate  # Windows
```

4. Zainstaluj zależności:
```bash
pip install -e .
```

5. Utwórz plik `.env` w głównym katalogu projektu i dodaj klucz API OpenAI:
```
OPENAI_API_KEY=twoj-klucz-api
```

6. Pobierz model:
   - Opcja A: Pobierz wytrenowany model z [Google Drive](https://drive.google.com/drive/folders/1MobjEblArzMQ2FGiFK2UGwITGrcN5ERs?usp=sharing) i umieść w `models/`
   - Opcja B: Wytrenuj własny model używając `python train.py` (wymaga pobrania datasetu)

## 📸 Przygotowanie zdjęć

1. Umieść zdjęcia składników w katalogu `Vegetables/`
2. Obsługiwane formaty: JPG, JPEG, PNG
3. Zalecana rozdzielczość: minimum 150x150 pikseli

## 🎯 Użycie

1. Uruchom program:
```bash
python main.py
```

2. Z menu głównego możesz:
   - Generować nowe przepisy ze zdjęć
   - Przeglądać historię przepisów

3. Podczas generowania przepisu:
   - Wybierz standardowy przepis lub własny pomysł
   - Poczekaj na wygenerowanie przepisu
   - Korzystaj z dodatkowych funkcji w oknie przepisu

## 💡 Funkcje dodatkowe

- **Analiza wartości odżywczych**: Oblicza kalorie, białko, tłuszcze i węglowodany
- **Zmiana liczby porcji**: Automatycznie przelicza ilości składników
- **Sprawdzanie diety**: Weryfikuje zgodność z dietami (wegańska, wegetariańska, bezglutenowa, keto)
- **Warianty przepisu**: Generuje zdrowsze, szybsze i budżetowe wersje
- **Eksport do PDF**: Zapisuje przepis w formacie PDF z formatowaniem
- **Historia przepisów**: Przeglądaj i oceniaj wcześniej wygenerowane przepisy

## 🔧 Wymagania systemowe

- Python 3.10 lub nowszy
- Git LFS
- Dostęp do internetu (dla API OpenAI)
- Minimum 2GB RAM
- Około 500MB miejsca na dysku

## 📝 Struktura projektu

- `main.py` - Główny plik programu
- `train.py` - Skrypt do trenowania modelu
- `test.py` - Skrypt do testowania modelu
- `models/` - Katalog na pliki modelu
- `Recipes/` - Katalog na wygenerowane przepisy
- `Vegetables/` - Katalog na zdjęcia składników
- `archive/` - Katalog na dane treningowe
  - `train/` - Zdjęcia do treningu
  - `validation/` - Zdjęcia do walidacji
  - `test/` - Zdjęcia do testów

## 🤝 Współpraca

1. Zrób fork repozytorium
2. Utwórz nową gałąź (`git checkout -b feature/nazwa`)
3. Zatwierdź zmiany (`git commit -am 'Dodano nową funkcję'`)
4. Wypchnij gałąź (`git push origin feature/nazwa`)
5. Utwórz Pull Request

## 📄 Licencja

Ten projekt jest objęty licencją MIT - szczegóły w pliku [LICENSE](LICENSE)

## ⚠️ Znane problemy

- Program może działać wolniej na komputerach bez GPU
- Niektóre błędy CUDA można bezpiecznie zignorować
- Wymagane jest stabilne połączenie internetowe

## 🙋‍♂️ Wsparcie

W razie problemów:
1. Sprawdź sekcję [Issues](https://github.com/twoj-username/NAIProjekt/issues)
2. Utwórz nowe zgłoszenie z dokładnym opisem problemu

## 🤖 Model AI

### Opcja 1: Pobranie wytrenowanego modelu

1. Pobierz wytrenowany model z [Google Drive](https://drive.google.com/drive/folders/1MobjEblArzMQ2FGiFK2UGwITGrcN5ERs?usp=sharing)
2. Umieść plik `trained_model.keras` w katalogu `models/`
3. Pobierz zbiór danych treningowych z [kaggle](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)
4. Rozpakuj archiwum do katalogu `src/`
5. Uruchom program:
```bash
python main.py
```

### Opcja 2: Trenowanie własnego modelu

Model wykorzystuje architekturę MobileNetV2 z transfer learningiem.

1. Pobierz zbiór danych treningowych (Użyłem [tego zbioru danych](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition), ale możesz użyć dowolnego innego)
2. Rozpakuj archiwum do katalogu `src/`
3. Uruchom skrypt trenujący:
```bash
python train.py
```

#### Szczegóły modelu:
- Model bazowy: MobileNetV2 (pre-trained na ImageNet)
- Liczba klas: 36 składników
- Dokładność na zbiorze testowym: 96.48%

#### Parametry treningu:
- Batch size: 128
- Epochs: 100 (z early stopping)
- Optimizer: Adam
- Learning rate: adaptacyjny
- Data augmentation:
  - Rotacja: ±30°
  - Przesunięcia: ±20%
  - Zoom: ±20%
  - Odbicia poziome

#### Dataset:
- Liczba obrazów treningowych: 3115
- Liczba obrazów walidacyjnych: 351
- Liczba obrazów testowych: 359

#### Struktura danych treningowych:
```
archive/
├── train/
│   ├── apple/
│   ├── banana/
│   └── ...
├── validation/
│   ├── apple/
│   ├── banana/
│   └── ...
└── test/
    ├── apple/
    ├── banana/
    └── ...
```

#### Monitorowanie treningu:
- Postęp treningu jest zapisywany w pliku `training_history.png`
- Najlepszy model jest automatycznie zapisywany w `models/trained_model.keras`
- Metryki treningu są wyświetlane w czasie rzeczywistym

#### Wymagania sprzętowe do treningu:
- Minimum 8GB RAM
- GPU z minimum 4GB VRAM (opcjonalnie, ale zalecane)
- Około 2GB wolnego miejsca na dysku
