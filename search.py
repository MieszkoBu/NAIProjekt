import os

def check_model_files():
    """Sprawdź dostępne pliki modeli"""
    model_paths = [
        "src/models",
        "models",
        "."
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"\nPliki w {os.path.abspath(path)}:")
            for file in os.listdir(path):
                if file.endswith(('.keras', '.h5')):
                    print(f"- {file}")

check_model_files()