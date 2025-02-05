"""Moduł do testowania i porównywania wytrenowanych modeli klasyfikacji obrazów."""

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
import random

# Definicje funkcji preprocessingu
preprocessing_functions = {
    'resnet': tf.keras.applications.resnet50.preprocess_input,
    'efficientnet': tf.keras.applications.efficientnet.preprocess_input,
    'mobilenet': tf.keras.applications.mobilenet_v2.preprocess_input
}

def create_resnet_model():
    """Odtwórz architekturę modelu ResNet50."""
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )
    
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    predictions = tf.keras.layers.Dense(36, activation='softmax')(x)
    
    return tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

def get_model_path(filename):
    """Znajdź plik modelu w różnych możliwych lokalizacjach."""
    possible_paths = [
        filename,
        os.path.join("src", "models", filename),
        os.path.join(os.getcwd(), "src", "models", filename),
        os.path.join(os.getcwd(), filename)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(f"Nie znaleziono pliku modelu w żadnej z lokalizacji: {possible_paths}")

# Ścieżki do modeli - sprawdź jakie formaty plików masz dostępne
models_to_test = {
    'ResNet50': ('trained_model_resnet.keras', 'resnet'),
    'EfficientNetB0': ('trained_model_efficientnet.keras', 'efficientnet'),
    'MobileNetV2': ('trained_model.keras', 'mobilenet')
}

def load_model_with_weights(model_path: str, model_type: str):
    """Załaduj model z pliku wag lub pełnego modelu."""
    if model_type == 'efficientnet':
        # Stwórz model EfficientNet od nowa
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(150, 150, 3)
        )
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(2048, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        predictions = tf.keras.layers.Dense(36, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        
        # Załaduj wagi
        model.load_weights(model_path.replace('.keras', '_weights.h5'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall', 'AUC']
        )
        return model
    elif model_path.endswith('.h5'):
        if model_type == 'resnet':
            model = create_resnet_model()
        elif model_type == 'mobilenet':
            base_model = tf.keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(150, 150, 3)
            )
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(1024, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            predictions = tf.keras.layers.Dense(36, activation='softmax')(x)
            model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        
        model.load_weights(model_path)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall', 'AUC']
        )
    else:
        model = tf.keras.models.load_model(model_path)
    return model

# Załaduj wszystkie modele na początku
loaded_models = {}
for model_name, model_info in models_to_test.items():
    try:
        print(f"\nŁadowanie modelu {model_name}...")
        if isinstance(model_info, tuple):
            model_path, model_type = model_info
            full_path = get_model_path(model_path)
            print(f"Ścieżka: {os.path.abspath(full_path)}")
            print(f"Typ modelu: {model_type}")
            print(f"Format pliku: {os.path.splitext(full_path)[1]}")
            loaded_models[model_name] = load_model_with_weights(full_path, model_type)
        else:
            full_path = get_model_path(model_info)
            print(f"Ścieżka: {os.path.abspath(full_path)}")
            loaded_models[model_name] = tf.keras.models.load_model(full_path)
        print(f"Model {model_name} załadowany pomyślnie!")
    except Exception as e:
        print(f"Błąd podczas ładowania modelu {model_name}: {e}")
        import traceback
        traceback.print_exc()

if not loaded_models:
    print("Nie udało się załadować żadnego modelu!")
    exit(1)

# Słownik na wyniki
results = {}

# Testuj załadowane modele
print("\nTestowanie modeli...")
for model_name, model in loaded_models.items():
    print(f"\nTestowanie modelu {model_name}...")
    print(f"Architektura modelu:")
    model.summary()
    print(f"Metryki modelu: {model.metrics_names}")
    try:
        # Stwórz generator z odpowiednim preprocessingiem dla danego modelu
        model_type = models_to_test[model_name][1]
        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocessing_functions[model_type]
        )
        test_generator = test_datagen.flow_from_directory(
            os.path.join(os.getcwd(), "src", "archive", "test"),
            target_size=(150, 150),
            batch_size=32,
            class_mode="categorical",
            shuffle=True
        )
        
        # Rekompiluj model z odpowiednimi metrykami
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall', 'AUC']
        )
        metrics = model.evaluate(test_generator, verbose=1)
        results[model_name] = dict(zip(model.metrics_names, metrics))
        
        print(f"\nWyniki dla {model_name}:")
        for metric_name, value in results[model_name].items():
            print(f"{metric_name}: {value:.4f}")
    except Exception as e:
        print(f"Błąd podczas testowania {model_name}: {e}")

# Porównanie wyników
print("\nPorównanie modeli:")
metrics_to_compare = ['accuracy', 'precision', 'recall', 'auc']
for metric in metrics_to_compare:
    print(f"\n{metric.upper()}:")
    for model_name in results:
        if metric in results[model_name]:
            print(f"{model_name}: {results[model_name][metric]:.4f}")

# Wizualizacja wyników
plt.figure(figsize=(12, 6))
x = np.arange(len(metrics_to_compare))
width = 0.25
multiplier = 0

for model_name, model_results in results.items():
    metric_values = [model_results.get(metric, 0) for metric in metrics_to_compare]
    offset = width * multiplier
    plt.bar(x + offset, metric_values, width, label=model_name)
    multiplier += 1

plt.xlabel('Metryki')
plt.ylabel('Wartość')
plt.title('Porównanie modeli')
plt.xticks(x + width, metrics_to_compare)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Pobierz 10 losowych próbek ze zbioru testowego
print("\nPrzygotowywanie losowych próbek do testów...")
# Załaduj bezpośrednio obrazy z katalogu testowego
test_dir = os.path.join(os.getcwd(), "src", "archive", "test")
test_samples = []
test_labels = []
class_dirs = sorted(os.listdir(test_dir))
for _ in range(10):
    # Wybierz losową klasę
    class_name = random.choice(class_dirs)
    class_path = os.path.join(test_dir, class_name)
    if os.path.isdir(class_path):
        # Wybierz losowy obraz z tej klasy
        image_name = random.choice(os.listdir(class_path))
        image_path = os.path.join(class_path, image_name)
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
        image = tf.keras.preprocessing.image.img_to_array(image)
        test_samples.append(image)
        # Utwórz one-hot encoded label
        label = np.zeros(len(class_dirs))
        label[class_dirs.index(class_name)] = 1
        test_labels.append(label)

# Przykładowe predykcje na losowych próbkach
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Porównanie predykcji modeli na losowych próbkach', fontsize=16)

for i, (ax, image, label) in enumerate(zip(axes.flat, test_samples, test_labels)):
    ax.imshow(array_to_img(image))
    true_label = class_dirs[np.argmax(label)]
    predictions = {}
    print(f"\nPredykcje dla obrazu {i}:")
    print(f"Prawdziwa etykieta: {true_label}")
    
    for model_name, model in loaded_models.items():
        try:
            # Preprocessuj obraz zgodnie z wymaganiami modelu
            model_type = models_to_test[model_name][1]
            preprocess_input = preprocessing_functions[model_type]
            processed_image = preprocess_input(np.expand_dims(image.copy(), axis=0))
            pred = model.predict(processed_image, verbose=0)
            
            # Pokaż top 3 predykcje
            top_3 = np.argsort(pred[0])[-3:][::-1]
            print(f"\n{model_name} top 3 predykcje:")
            for idx in top_3:
                class_name = list(test_generator.class_indices.keys())[idx]
                confidence = pred[0][idx]
                print(f"- {class_name}: {confidence:.2%}")
            
            pred_label = list(test_generator.class_indices.keys())[np.argmax(pred)]
            predictions[model_name] = pred_label
        except Exception as e:
            print(f"Błąd predykcji dla {model_name}: {e}")
            predictions[model_name] = "Error"
    
    title = f'True: {true_label}\n' + '\n'.join([f'{k}: {v}' for k,v in predictions.items()])
    ax.set_title(title, fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.show()

# Pokaż mapowanie klas
print("\nMapowanie klas:")
for class_name, idx in test_generator.class_indices.items():
    print(f"{idx}: {class_name}")