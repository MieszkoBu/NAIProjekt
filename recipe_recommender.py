"""Moduł do rekomendacji podobnych przepisów."""

import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class RecipeRecommender:
    def __init__(self):
        """Inicjalizuj model."""
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def create_recipe_features(self, recipe_text: str) -> str:
        """Przygotuj tekst przepisu do analizy podobieństwa."""
        try:
            ingredients = recipe_text.split("📝 SKŁADNIKI:")[1].split("👨‍🍳")[0].lower()
            instructions = recipe_text.split("👨‍🍳 INSTRUKCJE:")[1].lower()
            return f"składniki: {ingredients} instrukcje: {instructions}"
        except Exception as e:
            print(f"Błąd podczas przetwarzania tekstu przepisu: {e}")
            return recipe_text.lower()

    def get_similar_recipes(
        self,
        recipe_id: int, 
        recipes: List[Dict[str, Any]],
        similarity_threshold: float = 0.5,  
        n_recommendations: int = 3
    ) -> List[Dict[str, Any]]:
        """Znajdź podobne przepisy do danego przepisu."""
        if len(recipes) < 2:
            print("Za mało przepisów w bazie do porównania")
            return []
        
        current_recipe = next((r for r in recipes if r["id"] == recipe_id), None)
        if current_recipe is None:
            print(f"Nie znaleziono przepisu o ID: {recipe_id}")
            return []
        
        try:
            print(f"Przetwarzanie przepisu: {current_recipe['recipe_name']}")
            recipe_texts = [self.create_recipe_features(r["instructions"]) for r in recipes]
            
            recipe_embeddings = self.model.encode(recipe_texts)
            
            current_recipe_idx = next(i for i, r in enumerate(recipes) if r["id"] == recipe_id)
            
            similarities = cosine_similarity(
                [recipe_embeddings[current_recipe_idx]], 
                recipe_embeddings
            )[0]
            
            adjusted_threshold = similarity_threshold * 0.5  
            
            similar_indices = [
                i for i, sim in enumerate(similarities)
                if sim >= adjusted_threshold and recipes[i]["id"] != recipe_id
            ]
            
            similar_indices.sort(key=lambda x: similarities[x], reverse=True)
            similar_indices = similar_indices[:n_recommendations]
            
            if not similar_indices:
                print(f"Nie znaleziono przepisów o podobieństwie >= {similarity_threshold}")
                return []
            
            similar_recipes = [recipes[i] for i in similar_indices]
            
            for i, recipe in enumerate(similar_recipes):
                similarity_score = float(similarities[similar_indices[i]])
                recipe["similarity"] = min(5, int(similarity_score * 6))
                print(f"Znaleziono podobny przepis: {recipe['recipe_name']} "
                      f"(podobieństwo: {recipe['similarity']}/5)")
            
            return similar_recipes
            
        except Exception as e:
            print(f"Błąd podczas wyszukiwania podobnych przepisów: {e}")
            import traceback
            traceback.print_exc()
            return []

    def __call__(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Umożliwia wywołanie instancji klasy jak funkcji."""
        return self.get_similar_recipes(*args, **kwargs) 