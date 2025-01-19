"""Moduł do rozpoznawania składników i generowania przepisów."""

import os
import sqlite3
import tkinter as tk
from io import BytesIO
from tkinter import Event, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import openai
import requests
from dotenv import load_dotenv
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageTk
import tensorflow as tf
import sys

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

recipe_history: Dict[str, List[str]] = {}

possible_model_paths = [
    os.path.join("src", "models", "trained_model.keras")  # Używamy tylko jednej, pewnej ścieżki
]

model = None
for model_path in possible_model_paths:
    try:
        print(f"Próba załadowania modelu z: {os.path.abspath(model_path)}")
        # Dodajemy dodatkowe opcje ładowania
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False,  # Wyłączamy tryb bezpieczny
            custom_objects=None  # Pozwalamy na automatyczne wykrycie obiektów
        )
        print(f"Sukces! Model załadowany z: {model_path}")
        break
    except Exception as e:
        print(f"Nie udało się załadować modelu z {model_path}: {e}")
        # Dodajemy więcej szczegółów o błędzie
        import traceback
        traceback.print_exc()

if model is None:
    print("BŁĄD KRYTYCZNY: Nie udało się załadować modelu z żadnej lokalizacji!")
    print("Upewnij się, że plik modelu istnieje w jednej z lokalizacji:")
    for path in possible_model_paths:
        print(f"- {os.path.abspath(path)}")
    sys.exit(1)  

train_dir = os.path.join(os.getcwd(), "src", "archive", "train")
train_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=128, class_mode="categorical"
)


def recognize_ingredient(image_path: str, model: Any, class_labels: Dict[int, str]) -> str:
    """Rozpoznaj składnik na zdjęciu.

    Args:
        image_path: Ścieżka do pliku ze zdjęciem
        model: Model do klasyfikacji
        class_labels: Słownik etykiet klas

    Returns:
        Nazwa rozpoznanego składnika
    """
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class]


def search_recipe_with_gpt(
    ingredient: str, previous_recipes: Optional[List[str]] = None, custom_prompt: Optional[str] = None
) -> str:
    """Wyszukaj przepis używając GPT."""
    if custom_prompt:
        content = (
            f"Stwórz przepis na danie z {ingredient} zgodnie z następującym pomysłem: {custom_prompt}. "
            f"Odpowiedz dokładnie w tym formacie:\n\n"
        )
    else:
        content = "Znajdź przepis na danie z {}. Odpowiedz dokładnie w tym formacie:\n\n".format(ingredient)

    messages = [
        {
            "role": "system",
            "content": (
                "Jesteś pomocnym asystentem kulinarnym. Generuj przepisy w dokładnie określonym formacie. "
                "ZAWSZE podawaj dokładne ilości składników z jednostkami miary (g, ml, łyżki, sztuki itp.) "
                "i konkretną liczbę porcji. Nie pomijaj żadnych ilości."
            ),
        },
        {
            "role": "user",
            "content": content
            + (
                f"🍳 NAZWA DANIA:\n"
                f"[tylko nazwa dania]\n\n"
                f"👥 PORCJE:\n"
                f"[dokładna liczba porcji, np. '4 porcje']\n\n"
                f"📝 SKŁADNIKI:\n"
                f"[lista składników, każdy z dokładną ilością i jednostką miary, np.:\n"
                f"- 500 g mięsa\n"
                f"- 2 łyżki oleju\n"
                f"- 3 ząbki czosnku\n"
                f"- 1 szklanka mleka\n"
                f"- 1/2 łyżeczki soli]\n\n"
                f"👨‍🍳 INSTRUKCJE:\n"
                f"[ponumerowane kroki przygotowania, każdy w nowej linii]"
            ),
        },
    ]

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
    )

    recipe = response.choices[0].message.content

    if "📝 SKŁADNIKI:" in recipe:
        ingredients_section = recipe.split("📝 SKŁADNIKI:")[1].split("👨‍🍳")[0]
        if not all(
            any(char.isdigit() for char in line)
            for line in ingredients_section.split("\n")
            if line.strip() and not line.startswith("[")
        ):
            return search_recipe_with_gpt(ingredient, previous_recipes)

    return recipe


def download_recipe_image(image_url: str, dish_name: str, output_folder: str) -> Optional[str]:
    """Pobierz zdjęcie przepisu z URL."""
    try:
        if "commons.wikimedia.org" in image_url:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(image_url, headers=headers, timeout=10)
            response.raise_for_status()

            import re

            image_matches = re.findall(
                r'https://upload\.wikimedia\.org/wikipedia/commons/[^"]+\.(?:jpg|jpeg|png)', response.text
            )
            if image_matches:
                image_url = image_matches[0]
                print(f"Znaleziono bezpośredni link do obrazu: {image_url}")
            else:
                print("Nie znaleziono bezpośredniego linku do obrazu na stronie Wikimedia")
                return None

        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            print(f"URL nie prowadzi do obrazu: {content_type}")
            return None

        img = Image.open(BytesIO(response.content))
        output_path = os.path.join(output_folder, f"{dish_name}.jpg")
        img.save(output_path)
        print(f"Zapisano zdjęcie w: {output_path}")
        return output_path
    except Exception as e:
        print(f"Błąd podczas pobierania zdjęcia: {e}")
        return None


def save_recipe_to_file(ingredient: str, recipe: str, output_folder: str) -> str:
    """Zapisz przepis do pliku tekstowego."""
    os.makedirs(output_folder, exist_ok=True)

    dish_name = ingredient
    recipe_lines = recipe.split("\n")

    for i, line in enumerate(recipe_lines):
        if "🍳 NAZWA DANIA:" in line:
            if i + 1 < len(recipe_lines):
                next_line = recipe_lines[i + 1].strip()
                if next_line and not next_line.startswith("📝"):
                    dish_name = next_line
                    break

    dish_name = dish_name.strip()
    dish_name = dish_name.replace("1.", "").strip()
    dish_name = dish_name.replace("2.", "").strip()
    dish_name = dish_name.replace("3.", "").strip()
    dish_name = dish_name.replace("4.", "").strip()

    filename = f"{dish_name.replace(' ', '_').replace(':', '').replace(',', '')}_przepis.txt"
    filepath = os.path.join(output_folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(recipe)

    return filepath


def analyze_nutrition(ingredients: str) -> Dict[str, float]:
    """Analizuj wartości odżywcze na podstawie składników."""
    prompt = f"Przeanalizuj wartości odżywcze dla tych składników i podaj kalorie, białko, tłuszcze i węglowodany:\n{ingredients}"
    response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
    return parse_nutrition(response.choices[0].message.content)


def is_recipe_suitable(recipe: str, diet_type: str) -> bool:
    """Sprawdź czy przepis jest odpowiedni dla danej diety."""
    prompt = f"Czy ten przepis jest odpowiedni dla diety {diet_type}? Odpowiedz tak lub nie:\n{recipe}"
    response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
    return "tak" in response.choices[0].message.content.lower()


def scale_recipe(recipe: str, servings: int) -> str:
    """Dostosuj ilości składników do zadanej liczby porcji."""
    prompt = f"Dostosuj ilości składników w przepisie do {servings} porcji:\n{recipe}"
    response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content


def get_recipe_variations(recipe: str) -> List[str]:
    """Generuj alternatywne wersje przepisu."""
    variations = ["zdrowsza", "szybsza", "budżetowa", "tradycyjna"]
    results = []
    for variant in variations:
        prompt = f"Stwórz {variant} wersję tego przepisu:\n{recipe}"
        response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        results.append(response.choices[0].message.content)
    return results


def parse_nutrition(text: str) -> Dict[str, float]:
    """Parsuj tekst z wartościami odżywczymi na słownik."""
    nutrition = {}
    try:
        lines = text.split("\n")
        for line in lines:
            if ":" in line:
                key, value = line.split(":")
                import re

                numbers = re.findall(r"\d+\.?\d*", value)
                if numbers:
                    nutrition[key.strip()] = float(numbers[0])
    except Exception as e:
        print(f"Błąd podczas parsowania wartości odżywczych: {e}")
    return nutrition


class RecipeApp(tk.Tk):
    """Główne okno aplikacji."""

    def __init__(self, parent: tk.Tk, recipe: str, ingredient: str, recipe_id: int) -> None:
        """Inicjalizuj okno przepisu."""
        super().__init__()
        self.title(f"Przepis na danie z {ingredient}")
        self.recipe = recipe
        self.ingredient = ingredient
        self.parent = parent
        self.recipe_id = recipe_id

        window_width = 800
        window_height = 600
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        self.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")

        main_frame = tk.Frame(self)
        main_frame.pack(padx=20, pady=20, expand=True, fill="both")

        recipe_frame = tk.Frame(main_frame)
        recipe_frame.pack(expand=True, fill="both")

        self.recipe_text = ScrolledText(recipe_frame, width=60, height=20, font=("Arial", 12))
        self.recipe_text.pack(expand=True, fill="both", pady=(0, 10))
        recipe_text = recipe.replace("🍳", "\n\n🍳")
        recipe_text = recipe_text.replace("👥", "\n\n👥")
        recipe_text = recipe_text.replace("📝", "\n\n📝")
        recipe_text = recipe_text.replace("👨‍🍳", "\n\n👨‍🍳")
        self.recipe_text.insert(tk.END, recipe_text)
        self.recipe_text.configure(state="disabled")

        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 10))

        self.analyze_btn = tk.Button(
            button_frame, text="Analiza wartości odżywczych", command=self.show_nutrition, width=20, height=2
        )
        self.analyze_btn.pack(side="left", padx=5)

        self.scale_btn = tk.Button(
            button_frame, text="Zmień liczbę porcji", command=self.change_servings, width=20, height=2
        )
        self.scale_btn.pack(side="left", padx=5)

        self.diet_btn = tk.Button(button_frame, text="Sprawdź dietę", command=self.check_diet, width=20, height=2)
        self.diet_btn.pack(side="left", padx=5)

        self.variations_btn = tk.Button(
            button_frame, text="Warianty przepisu", command=self.show_variations, width=20, height=2
        )
        self.variations_btn.pack(side="left", padx=5)

        self.history_btn = tk.Button(
            button_frame, text="Historia przepisów", command=self.show_history, width=20, height=2
        )
        self.history_btn.pack(side="left", padx=5)

        self.export_btn = tk.Button(
            button_frame, text="Eksportuj do PDF", command=self.export_to_pdf, width=20, height=2
        )
        self.export_btn.pack(side="left", padx=5)

        rating_frame = tk.Frame(main_frame)
        rating_frame.pack(fill="x", pady=5)
        tk.Label(rating_frame, text="Oceń przepis:").pack(side="left", padx=5)

        self.rating_var = tk.StringVar(value="5")
        ratings = ["1", "2", "3", "4", "5"]
        for r in ratings:
            tk.Radiobutton(rating_frame, text=r, variable=self.rating_var, value=r, command=self.update_rating).pack(
                side="left"
            )

        result_frame = tk.Frame(main_frame)
        result_frame.pack(expand=True, fill="both")

        self.result_text = ScrolledText(result_frame, width=60, height=10, font=("Arial", 12))
        self.result_text.pack(expand=True, fill="both")

        self.lift()
        self.focus_force()

    def show_nutrition(self) -> None:
        """Pokaż wartości odżywcze."""
        ingredients = self.recipe.split("📝 SKŁADNIKI:")[1].split("👨‍🍳")[0]
        nutrition = analyze_nutrition(ingredients)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Wartości odżywcze:\n\n")
        for key, value in nutrition.items():
            self.result_text.insert(tk.END, f"{key}: {value}\n")

    def change_servings(self) -> None:
        """Zmień liczbę porcji."""
        dialog = tk.Toplevel(self)
        dialog.title("Zmień liczbę porcji")

        tk.Label(dialog, text="Podaj liczbę porcji:").pack(padx=10, pady=5)
        entry = tk.Entry(dialog)
        entry.pack(padx=10, pady=5)

        def update_recipe() -> None:
            try:
                servings = int(entry.get())
                new_recipe = scale_recipe(self.recipe, servings)
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, new_recipe)
                dialog.destroy()
            except ValueError:
                tk.messagebox.showerror("Błąd", "Podaj prawidłową liczbę")

        tk.Button(dialog, text="OK", command=update_recipe).pack(pady=10)

    def check_diet(self) -> None:
        """Sprawdź zgodność z dietami."""
        diets = ["wegańska", "wegetariańska", "bezglutenowa", "ketogeniczna"]
        results = []
        for diet in diets:
            if is_recipe_suitable(self.recipe, diet):
                results.append(f"✓ Przepis jest odpowiedni dla diety: {diet}")
            else:
                results.append(f"✗ Przepis nie jest odpowiedni dla diety: {diet}")

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "\n".join(results))

    def show_variations(self) -> None:
        """Pokaż warianty przepisu."""
        variations = get_recipe_variations(self.recipe)
        self.result_text.delete(1.0, tk.END)
        for i, variant in enumerate(variations, 1):
            self.result_text.insert(tk.END, f"\n=== Wariant {i} ===\n{variant}\n")

    def show_history(self) -> None:
        """Pokaż historię przepisów."""
        history_window = tk.Toplevel(self)
        history_window.title("Historia przepisów")
        history_window.geometry("800x600")

        history = get_recipe_history()

        tree = ttk.Treeview(history_window, columns=("Data", "Składnik", "Nazwa", "Ocena"))
        tree.heading("Data", text="Data")
        tree.heading("Składnik", text="Składnik")
        tree.heading("Nazwa", text="Nazwa przepisu")
        tree.heading("Ocena", text="Ocena")

        for recipe in history:
            tree.insert(
                "",
                "end",
                values=(recipe["created_at"], recipe["ingredient"], recipe["recipe_name"], f"{recipe['rating']}⭐"),
            )

        tree.pack(fill="both", expand=True)

        def show_recipe_details(event: tk.Event) -> None:
            """Pokaż szczegóły przepisu."""
            item = tree.selection()[0]
            recipe = history[tree.index(item)]

            def on_recipe_close() -> None:
                """Zamknij okno przepisu."""
                if isinstance(self.recipe_window, RecipeApp):
                    self.recipe_window.destroy()
                    self.recipe_window = None
                self.deiconify()

            if isinstance(self.recipe_window, RecipeApp):
                self.recipe_window.destroy()
                self.recipe_window = None

            temp_window = RecipeApp(self, recipe["instructions"], recipe["ingredient"], recipe["id"])
            if isinstance(temp_window, RecipeApp):
                temp_window.protocol("WM_DELETE_WINDOW", on_recipe_close)
                self.recipe_window = temp_window
                temp_window.mainloop()

        tree.bind("<Double-1>", show_recipe_details)

        def on_history_close() -> None:
            """Zamknij okno historii."""
            history_window.destroy()
            self.deiconify()

        history_window.protocol("WM_DELETE_WINDOW", on_history_close)

    def export_to_pdf(self) -> None:
        """Eksportuj przepis do PDF."""
        from tkinter import filedialog

        filename = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if filename:
            recipe_data = {
                "recipe_name": self.recipe.split("🍳 NAZWA DANIA:")[1].split("\n")[0],
                "instructions": self.recipe,
                "nutrition": self.result_text.get("1.0", tk.END),
            }
            export_recipe_to_pdf(recipe_data, filename)
            tk.messagebox.showinfo("Sukces", "Przepis został wyeksportowany do PDF!")

    def update_rating(self) -> None:
        """Aktualizuj ocenę przepisu."""
        rating = int(self.rating_var.get())
        update_recipe_rating(self.recipe_id, rating)


def save_to_database(recipe: Dict[str, Any]) -> Optional[int]:
    """Zapisz przepis i wyniki rozpoznawania do bazy danych."""
    conn = sqlite3.connect("recipes.db")
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO recipes (ingredient, recipe_name, instructions, nutrition)
            VALUES (?, ?, ?, ?)
        """,
            (recipe["ingredient"], recipe["name"], recipe["instructions"], recipe["nutrition"]),
        )
        recipe_id = cursor.lastrowid
        conn.commit()
        return recipe_id if recipe_id is not None else 0
    finally:
        conn.close()


def create_database() -> None:
    """Utwórz bazę danych jeśli nie istnieje."""
    conn = sqlite3.connect("recipes.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ingredient TEXT,
            recipe_name TEXT,
            instructions TEXT,
            nutrition TEXT,
            rating INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ingredient TEXT,
            search_type TEXT,
            custom_prompt TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            recipe_id INTEGER,
            FOREIGN KEY (recipe_id) REFERENCES recipes (id)
        )
    """
    )

    conn.commit()
    conn.close()


def get_recipe_history() -> List[Dict[str, Any]]:
    """Pobierz historię przepisów z bazy danych."""
    conn = sqlite3.connect("recipes.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT r.*, h.search_type, h.custom_prompt
        FROM recipes r
        LEFT JOIN search_history h ON r.id = h.recipe_id
        ORDER BY r.created_at DESC
    """
    )
    columns = [description[0] for description in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return results


def update_recipe_rating(recipe_id: int, rating: int) -> None:
    """Aktualizuj ocenę przepisu."""
    conn = sqlite3.connect("recipes.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE recipes SET rating = ? WHERE id = ?", (rating, recipe_id))
    conn.commit()
    conn.close()


def export_recipe_to_pdf(recipe: Dict[str, Any], output_path: str) -> None:
    """Eksportuj przepis do pliku PDF."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("CustomTitle", parent=styles["Heading1"], fontSize=24, spaceAfter=30)
    story.append(Paragraph(recipe["recipe_name"], title_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Składniki:", styles["Heading2"]))
    ingredients = recipe["instructions"].split("📝 SKŁADNIKI:")[1].split("👨‍🍳")[0]
    for line in ingredients.split("\n"):
        if line.strip():
            story.append(Paragraph(f"• {line.strip()}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Instrukcje:", styles["Heading2"]))
    instructions = recipe["instructions"].split("👨‍🍳 INSTRUKCJE:")[1]
    for line in instructions.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), styles["Normal"]))

    if recipe["nutrition"]:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Wartości odżywcze:", styles["Heading2"]))
        story.append(Paragraph(recipe["nutrition"], styles["Normal"]))

    doc.build(story)


class IngredientDialog(tk.Toplevel):
    """Dialog wyboru typu przepisu."""

    def __init__(self, parent: Union[tk.Tk, tk.Toplevel], ingredient: str) -> None:
        """Inicjalizuj dialog wyboru przepisu."""
        super().__init__(parent)
        self.title("Wybór typu przepisu")
        self.ingredient = ingredient
        self.result: Optional[Tuple[str, Optional[str]]] = None

        self.grab_set()

        window_width = 600
        window_height = 400
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        self.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")

        main_frame = tk.Frame(self, padx=30, pady=30)
        main_frame.pack(expand=True, fill="both")

        tk.Label(main_frame, text=f"Rozpoznany składnik: {ingredient}", font=("Arial", 14, "bold")).pack(pady=(0, 20))
        tk.Label(main_frame, text="Wybierz opcję:", font=("Arial", 12)).pack(pady=(0, 10))

        tk.Button(
            main_frame, text="Generuj standardowy przepis", command=self.standard_recipe, width=35, height=2
        ).pack(pady=(0, 15))

        tk.Button(main_frame, text="Własny pomysł na przepis", command=self.custom_recipe, width=35, height=2).pack(
            pady=(0, 15)
        )

        tk.Label(main_frame, text="Wpisz własny pomysł na przepis:", font=("Arial", 10)).pack(pady=(0, 5))

        self.prompt_var = tk.StringVar()
        self.prompt_entry = tk.Entry(main_frame, textvariable=self.prompt_var, width=50)
        self.prompt_entry.pack(pady=(0, 15), ipady=5)
        self.prompt_entry.insert(0, "Np. 'zdrowy przepis wegetariański' lub 'szybkie danie na lunch'")
        self.prompt_entry.bind("<FocusIn>", self.clear_placeholder)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.transient(parent)

        self.wait_window()

    def on_close(self) -> None:
        """Obsługa zamknięcia okna."""
        self.result = None
        self.destroy()

    def clear_placeholder(self, event: tk.Event) -> None:
        """Wyczyść placeholder przy fokusie."""
        if "Np." in self.prompt_entry.get():
            self.prompt_entry.delete(0, tk.END)

    def standard_recipe(self) -> None:
        """Wybrano standardowy przepis."""
        self.result = ("standard", None)
        self.destroy()

    def custom_recipe(self) -> None:
        """Wybrano własny przepis."""
        prompt = self.prompt_var.get()
        if "Np." not in prompt:
            self.result = ("custom", prompt)
            self.destroy()
        else:
            tk.messagebox.showwarning("Uwaga", "Wprowadź własny pomysł na przepis!")


class MainWindow(tk.Tk):
    """Główne okno aplikacji z menu."""

    def __init__(self, directory: str, model: Any, class_labels: Dict[int, str]) -> None:
        """Inicjalizuj główne okno aplikacji."""
        super().__init__()
        if model is None:
            messagebox.showerror("Błąd", "Nie udało się załadować modelu!")
            self.destroy()
            return
        self.title("AI Recipe Generator")
        self.current_photo = None
        self.recipe_window: Optional[RecipeApp] = None
        self.directory = directory
        self.model = model
        self.class_labels = class_labels
        self.image_files = [f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.recipes_folder = os.path.join(os.getcwd(), "Recipes")

        window_width = 400
        window_height = 300
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        self.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")

        main_frame = tk.Frame(self, padx=30, pady=30)
        main_frame.pack(expand=True, fill="both")

        tk.Label(main_frame, text="AI Recipe Generator", font=("Arial", 20, "bold")).pack(pady=(0, 30))

        tk.Button(
            main_frame, text="Generuj nowy przepis", command=self.start_recipe_generation, width=30, height=2
        ).pack(pady=(0, 10))

        tk.Button(
            main_frame, text="Przeglądaj historię przepisów", command=self.show_recipe_history, width=30, height=2
        ).pack(pady=(0, 10))

        tk.Button(main_frame, text="Wyjście", command=self.quit, width=30, height=2).pack(pady=(20, 0))

    def start_recipe_generation(self) -> None:
        """Rozpocznij proces generowania przepisu."""
        self.withdraw()

        ingredient_window = self.show_ingredient_window()

        def on_ingredient_window_close() -> None:
            """Zamknij okno składnika."""
            if ingredient_window is not None:
                ingredient_window.destroy()
            self.deiconify()

        if ingredient_window is not None:
            ingredient_window.protocol("WM_DELETE_WINDOW", on_ingredient_window_close)

    def show_ingredient_window(self) -> Optional[tk.Toplevel]:
        """Pokaż okno ze składnikiem."""
        if self.model is None:
            messagebox.showerror("Błąd", "Model nie jest dostępny!")
            return None
        if not self.image_files:
            tk.messagebox.showwarning("Uwaga", "Brak zdjęć w katalogu!")
            return None

        filename = self.image_files[0]
        image_path = os.path.join(self.directory, filename)
        ingredient = recognize_ingredient(image_path, self.model, self.class_labels)

        img_window = tk.Toplevel(self)
        img_window.title(f"Rozpoznany składnik: {ingredient}")

        window_width = 500
        window_height = 600
        screen_width = img_window.winfo_screenwidth()
        screen_height = img_window.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        img_window.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")

        try:
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            max_size = (400, 400)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            self.current_photo = ImageTk.PhotoImage(img)

            if self.current_photo is not None:
                img_label = tk.Label(img_window, image=self.current_photo)
                img_label.pack(padx=10, pady=10)

            label_text = f"Rozpoznany składnik: {ingredient}"
            tk.Label(img_window, text=label_text, font=("Arial", 12, "bold")).pack(pady=10)

            def show_recipe_dialog() -> None:
                """Pokaż dialog wyboru przepisu."""
                dialog = IngredientDialog(img_window, ingredient)

                if dialog.result:
                    recipe_type, custom_prompt = dialog.result

                    if recipe_type == "custom":
                        recipe = search_recipe_with_gpt(ingredient, previous_recipes=None, custom_prompt=custom_prompt)
                    else:
                        recipe = search_recipe_with_gpt(ingredient, recipe_history.get(ingredient, []))

                    recipe_path = save_recipe_to_file(ingredient, recipe, self.recipes_folder)
                    print(f"Przepis zapisany w: {recipe_path}")

                    img_window.destroy()

                    recipe_id = save_to_database(
                        {
                            "ingredient": ingredient,
                            "name": recipe.split("🍳 NAZWA DANIA:")[1].split("📝")[0].strip(),
                            "instructions": recipe,
                            "nutrition": str(analyze_nutrition(recipe.split("📝 SKŁADNIKI:")[1].split("👨‍🍳")[0])),
                        }
                    )

                    if recipe_id is not None:
                        self.recipe_window = RecipeApp(self, recipe, ingredient, recipe_id)
                    else:
                        tk.messagebox.showerror("Błąd", "Nie udało się zapisać przepisu do bazy danych")

                    def on_recipe_close() -> None:
                        """Zamknij okno przepisu."""
                        if self.recipe_window is not None:
                            self.recipe_window.destroy()
                            self.recipe_window = None
                        self.deiconify()

                    self.recipe_window.protocol("WM_DELETE_WINDOW", on_recipe_close)

            tk.Button(img_window, text="Dalej", command=show_recipe_dialog, width=20, height=2).pack(pady=10)

            return img_window

        except Exception as e:
            print(f"Błąd podczas wyświetlania obrazu: {e}")
            img_window.destroy()
            return None

    def show_recipe_history(self) -> None:
        """Pokaż historię przepisów."""
        self.withdraw()

        history_window = tk.Toplevel(self)
        history_window.title("Historia przepisów")
        history_window.geometry("1200x800")

        history = get_recipe_history()

        tree = ttk.Treeview(history_window, columns=("Data", "Składnik", "Nazwa", "Ocena"))
        tree.column("Data", width=200)
        tree.column("Składnik", width=200)
        tree.column("Nazwa", width=300)
        tree.column("Ocena", width=100)

        tree.heading("Data", text="Data")
        tree.heading("Składnik", text="Składnik")
        tree.heading("Nazwa", text="Nazwa przepisu")
        tree.heading("Ocena", text="Ocena")

        screen_width = history_window.winfo_screenwidth()
        screen_height = history_window.winfo_screenheight()
        center_x = int(screen_width / 2 - 1200 / 2)
        center_y = int(screen_height / 2 - 800 / 2)
        history_window.geometry(f"1200x800+{center_x}+{center_y}")

        for recipe in history:
            tree.insert(
                "",
                "end",
                values=(
                    recipe["created_at"],
                    recipe["ingredient"],
                    recipe["recipe_name"],
                    f"{recipe['rating']}⭐" if recipe["rating"] else "Brak oceny",
                ),
            )

        tree.pack(fill="both", expand=True, padx=10, pady=10)

        def show_recipe_details(event: tk.Event) -> None:
            """Pokaż szczegóły przepisu."""
            item = tree.selection()[0]
            recipe = history[tree.index(item)]

            def on_recipe_close() -> None:
                """Zamknij okno przepisu."""
                if isinstance(self.recipe_window, RecipeApp):
                    self.recipe_window.destroy()
                    self.recipe_window = None
                self.deiconify()

            if isinstance(self.recipe_window, RecipeApp):
                self.recipe_window.destroy()
                self.recipe_window = None

            temp_window = RecipeApp(self, recipe["instructions"], recipe["ingredient"], recipe["id"])
            if isinstance(temp_window, RecipeApp):
                temp_window.protocol("WM_DELETE_WINDOW", on_recipe_close)
                self.recipe_window = temp_window
                temp_window.mainloop()

        tree.bind("<Double-1>", show_recipe_details)

        def on_history_close() -> None:
            """Zamknij okno historii."""
            history_window.destroy()
            self.deiconify()

        history_window.protocol("WM_DELETE_WINDOW", on_history_close)


def process_images_in_directory(directory: str, model: Any, class_labels: Dict[int, str]) -> List[Dict[str, str]]:
    """Przetwórz wszystkie obrazy w katalogu."""
    recipes_folder = os.path.join(os.getcwd(), "src", "Recipes")
    os.makedirs(recipes_folder, exist_ok=True)

    results: List[Dict[str, str]] = []

    app = MainWindow(directory, model, class_labels)
    app.mainloop()

    return results


def update_database_schema() -> None:
    """Aktualizuj schemat bazy danych."""
    conn = sqlite3.connect("recipes.db")
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(recipes)")
    columns = [column[1] for column in cursor.fetchall()]

    if "rating" not in columns:
        cursor.execute("ALTER TABLE recipes ADD COLUMN rating INTEGER DEFAULT 0")
        conn.commit()

    conn.close()


create_database()
update_database_schema()

directory = os.path.join(os.getcwd(), "src", "Vegetables")
os.makedirs(directory, exist_ok=True)
class_labels = {v: k for k, v in train_generator.class_indices.items()}
results = process_images_in_directory(directory, model, class_labels)
