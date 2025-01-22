# AI Recipe Generator

Generator przepisów kulinarnych wykorzystujący sztuczną inteligencję do rozpoznawania składników ze zdjęć i generowania przepisów.

## 🌟 Funkcjonalności

1. **Rozpoznawanie składników** 
   - Model CNN do klasyfikacji zdjęć składników
   - Obsługa 36 różnych kategorii składników
   - Możliwość przeglądania i wyboru zdjęć

2. **Generowanie przepisów**
   - Wykorzystanie OpenAI GPT do generowania przepisów
   - Dostosowanie do polskich składników i preferencji
   - Generowanie różnych wariantów przepisu

3. **Rekomendacje przepisów**
   - Model Sentence Transformers do znajdowania podobnych przepisów
   - Możliwość dostosowania progu podobieństwa
   - Wyświetlanie top 3 najbardziej podobnych przepisów

4. **Klasyfikacja typu kuchni**
   - Model XLM-RoBERTa do klasyfikacji typu kuchni
   - Rozpoznawanie 8 różnych typów kuchni
   - Wyświetlanie prawdopodobieństwa dla każdego typu

5. **Dodatkowe funkcje**
   - Analiza wartości odżywczych
   - Eksport przepisów do PDF
   - Historia przepisów z ocenami
   - Dostosowanie liczby porcji
   - Sprawdzanie diet

## 📱 Przykłady użycia

### Rozpoznawanie składników
![Rozpoznawanie składników](images/image.png)
*Okno rozpoznawania składników ze zdjęcia*

### Główne okno przepisu
![Główne okno](images/image2.png)
*Wygenerowany przepis z opcjami*

### Klasyfikacja typu kuchni
![Klasyfikacja kuchni](images/image3.png)
*Okno z klasyfikacją typu kuchni*

### Historia przepisów
![Historia](images/image4.png)
*Okno historii przepisów*

## 💻 Wymagania

- Python 3.10+
- Tensorflow 2.13+
- PyTorch 2.1+
- Transformers 4.35+
- Pozostałe zależności w `pyproject.toml`

## 🧠 Modele AI

### 1. Model rozpoznawania składników (CNN)
- Architektura: MobileNetV2
- Dokładność: 96.48%
- Obsługiwane składniki: 36 kategorii

### 2. Model rekomendacji (Sentence Transformers)
- Model: paraphrase-multilingual-MiniLM-L12-v2
- Funkcje: znajdowanie podobnych przepisów
- Język: wielojęzyczny (w tym polski)

### 3. Model klasyfikacji kuchni (XLM-RoBERTa)
- Model: xlm-roberta-base
- Klasyfikowane kuchnie: polska, włoska, azjatycka, meksykańska, francuska, śródziemnomorska, indyjska, amerykańska
- Dokładność: wielojęzyczna analiza tekstu

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
TOKENIZERS_PARALLELISM=true
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

## ❓ FAQ

### Jak dodać własne zdjęcia składników?
Umieść zdjęcia w formacie JPG/PNG w katalogu `Vegetables/`. Minimalna rozdzielczość to 150x150 pikseli.

### Jak wytrenować własny model?
1. Pobierz dataset z [Kaggle](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)
2. Umieść dane w katalogu `archive/`
3. Uruchom `python train.py`

### Jakie są wymagania sprzętowe?
- RAM: minimum 8GB
- GPU: opcjonalnie (przyspiesza działanie)
- Dysk: około 2GB wolnego miejsca

## 📝 Struktura projektu

```
NAIProjekt/
├── main.py                 # Główny plik aplikacji
├── recipe_recommender.py   # Model rekomendacji przepisów
├── cuisine_classifier.py   # Model klasyfikacji typu kuchni
├── src/
│   ├── models/            # Zapisane modele
│   ├── Recipes/           # Wygenerowane przepisy
│   └── Vegetables/        # Zdjęcia składników
└── tests/                 # Testy jednostkowe
```

## 👏 Podziękowania

- OpenAI za model GPT
- Hugging Face za modele transformers
- Kaggle za dataset treningowy
- Społeczność open source za wykorzystane biblioteki

## 📫 Kontakt

- GitHub: [MieszkoBu](https://github.com/MieszkoBu)
- Email: mieszkobu@wp.pl

## 📄 Licencja

Ten projekt jest objęty licencją MIT - szczegóły w pliku [LICENSE](LICENSE)

## ⚠️ Znane problemy

- Program może działać wolniej na komputerach bez GPU
- Niektóre błędy CUDA można bezpiecznie zignorować
- Wymagane jest stabilne połączenie internetowe

## 🙋‍♂️ Wsparcie

W razie problemów:
1. Sprawdź sekcję [Issues](https://github.com/MieszkoBu/NAIProjekt/issues)
2. Utwórz nowe zgłoszenie z dokładnym opisem problemu
