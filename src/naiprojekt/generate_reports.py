"""Moduł do generowania raportów porównawczych dla modeli."""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_test_generator(test_dir: str, preprocess_type: str, target_size=(150, 150)):
    """Utwórz generator danych testowych z odpowiednim preprocessingiem."""
    preprocessing_functions = {
        'resnet': tf.keras.applications.resnet50.preprocess_input,
        'efficientnet': tf.keras.applications.efficientnet.preprocess_input,
        'mobilenet': tf.keras.applications.mobilenet_v2.preprocess_input
    }
    
    preprocess_input = preprocessing_functions.get(preprocess_type, lambda x: x/255.0)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    return test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

def evaluate_model(model, model_name: str, test_generator, reports_dir: str) -> dict:
    """Ewaluuj model i zapisz raporty."""
    # Predykcje
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Nazwy klas
    class_names = list(test_generator.class_indices.keys())
    
    # Generuj raport klasyfikacji
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        digits=4
    )
    
    # Zapisz raport do pliku
    report_path = os.path.join(reports_dir, f'classification_report_{model_name}.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Oblicz macierz pomyłek
    cm = confusion_matrix(y_true, y_pred)
    
    # Wizualizacja macierzy pomyłek
    plt.figure(figsize=(20, 20))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f'Macierz pomyłek - {model_name}')
    plt.xlabel('Predykcja')
    plt.ylabel('Prawdziwa klasa')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Zapisz macierz pomyłek
    cm_path = os.path.join(reports_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(cm_path)
    plt.close()
    
    # Oblicz metryki
    metrics = model.evaluate(test_generator)
    metrics_names = model.metrics_names
    
    # Zbierz wyniki
    results = {}
    for name, value in zip(metrics_names, metrics):
        results[name] = value
        print(f"{name}: {value:.4f}")
    
    return results

def load_model(model_path: str, model_name: str):
    """Załaduj model z obsługą różnych formatów i konfiguracji."""
    try:
        if model_name == 'EfficientNetB0':
            # Utwórz model EfficientNetB0 z właściwą architekturą
            base_model = tf.keras.applications.EfficientNetB0(
                weights=None,
                include_top=False,
                input_shape=(150, 150, 3)
            )
            
            # Dodaj warstwy klasyfikacyjne
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(2048, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(1024, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            predictions = tf.keras.layers.Dense(36, activation='softmax')(x)
            
            model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
            
            # Załaduj wagi
            model.load_weights(model_path)
            return model
        else:
            return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Błąd podczas ładowania modelu {model_name}: {e}")
        raise

def main():
    """Główna funkcja generująca raporty."""
    # Utwórz katalog na raporty
    reports_dir = os.path.join("reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Ścieżka do danych testowych
    test_dir = os.path.join("archive", "test")
    
    # Lista modeli do przetestowania
    models = {
        'ResNet50': ('trained_model_resnet.keras', 'resnet'),
        'EfficientNetB0': ('trained_model_efficientnet.keras', 'efficientnet'),
        'MobileNetV2': ('trained_model.keras', 'mobilenet')
    }
    
    all_results = {}
    
    # Testuj każdy model
    for model_name, (model_file, preprocess_type) in models.items():
        try:
            print(f"\nŁadowanie modelu {model_name}...")
            model_path = os.path.join("src", "models", model_file)
            model = load_model(model_path, model_name)
            
            # Utwórz generator z odpowiednim preprocessingiem
            test_generator = create_test_generator(test_dir, preprocess_type)
            
            print(f"Ewaluacja modelu {model_name}...")
            results = evaluate_model(model, model_name, test_generator, reports_dir)
            all_results[model_name] = results
            
        except Exception as e:
            print(f"Błąd podczas testowania {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Zapisz podsumowanie wyników
    summary_path = os.path.join(reports_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Podsumowanie wyników:\n\n")
        for model_name, results in all_results.items():
            f.write(f"\n{model_name}:\n")
            for metric, value in results.items():
                f.write(f"{metric}: {value:.4f}\n")

if __name__ == "__main__":
    main() 