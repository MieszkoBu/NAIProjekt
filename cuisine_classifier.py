"""Moduł do klasyfikacji typu kuchni."""

from transformers import pipeline
from typing import List, Tuple

class CuisineClassifier:
    def __init__(self):
        """Inicjalizuj model klasyfikacji kuchni."""
        self.cuisine_types = [
            "polska", "włoska", "azjatycka", "meksykańska", 
            "francuska", "śródziemnomorska", "indyjska", "amerykańska"
        ]
        
        # Używamy gotowego modelu do klasyfikacji tekstu
        self.classifier = pipeline(
            "text-classification",
            model="papluca/xlm-roberta-base-language-detection",
            top_k=None  # Zwróć wszystkie przewidywania
        )

    def predict_cuisine(self, recipe_text: str) -> List[Tuple[str, float]]:
        """Przewiduj typ kuchni na podstawie przepisu."""
        try:
            # Wykonaj predykcję
            result = self.classifier(recipe_text)
            
            # Przetwórz wyniki
            predictions = []
            for item in result[0]:  # result[0] zawiera listę przewidywań
                cuisine_type = self.map_language_to_cuisine(item['label'])
                score = float(item['score'])
                predictions.append((cuisine_type, score))
            
            # Sortuj według prawdopodobieństwa
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Zwróć top 3 lub wszystkie jeśli jest ich mniej
            return predictions[:3]

        except Exception as e:
            print(f"Błąd podczas klasyfikacji kuchni: {e}")
            import traceback
            traceback.print_exc()
            return []

    def map_language_to_cuisine(self, language: str) -> str:
        """Mapuj język na typ kuchni."""
        mapping = {
            'pl': 'polska',
            'it': 'włoska',
            'zh': 'azjatycka',
            'es': 'meksykańska',
            'fr': 'francuska',
            'el': 'śródziemnomorska',
            'hi': 'indyjska',
            'en': 'amerykańska'
        }
        return mapping.get(language, 'nieznana')

    def get_main_cuisine(self, recipe_text: str) -> Tuple[str, float]:
        """Zwróć najbardziej prawdopodobny typ kuchni."""
        predictions = self.predict_cuisine(recipe_text)
        if predictions:
            return predictions[0]
        return ("nieznana", 0.0) 