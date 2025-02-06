"""Moduł do wyboru i obsługi modeli klasyfikacji obrazów."""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

class ModelSelector:
    def __init__(self):
        """Inicjalizuj selektor modeli."""
        self.current_model = None
        self.current_model_type = None
        self.model_paths = {
            'ResNet50': os.path.join('src', 'models', 'trained_model_resnet.keras'),
            'EfficientNetB0': os.path.join('src', 'models', 'trained_model_efficientnet.keras'),
            'MobileNetV2': os.path.join('src', 'models', 'trained_model.keras')
        }
        self.preprocessing_functions = {
            'ResNet50': tf.keras.applications.resnet50.preprocess_input,
            'EfficientNetB0': tf.keras.applications.efficientnet.preprocess_input,
            'MobileNetV2': tf.keras.applications.mobilenet_v2.preprocess_input
        }

    def get_available_models(self):
        """Zwraca listę dostępnych modeli."""
        return list(self.model_paths.keys())

    def load_model(self, model_name: str) -> None:
        """Załaduj wybrany model."""
        if model_name not in self.model_paths:
            raise ValueError(f"Nieznany model: {model_name}")
            
        model_path = self.model_paths[model_name]
        try:
            print(f"Ładowanie modelu z: {model_path}")
            if model_name == 'EfficientNetB0':
                # Specjalna obsługa dla EfficientNet
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
                self.current_model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
                # Załaduj wagi
                weights_path = model_path.replace('.keras', '_weights.h5')
                if os.path.exists(weights_path):
                    self.current_model.load_weights(weights_path)
                else:
                    print(f"UWAGA: Nie znaleziono pliku wag: {weights_path}")
            else:
                self.current_model = tf.keras.models.load_model(model_path)
            self.current_model_type = model_name
            print(f"Model załadowany pomyślnie")
        except Exception as e:
            print(f"Błąd podczas ładowania modelu: {e}")
            raise

    def get_preprocessing_function(self):
        """Zwróć funkcję preprocessingu dla aktualnego modelu."""
        if self.current_model_type in self.preprocessing_functions:
            return self.preprocessing_functions[self.current_model_type]
        return None

    def predict_image(self, image_path):
        """Przewiduje klasę dla podanego obrazu."""
        if not self.current_model:
            raise ValueError("Nie wybrano żadnego modelu")
            
        # Wczytaj i przygotuj obraz
        image = load_img(image_path, target_size=(150, 150))
        image = img_to_array(image)
        
        # Preprocessuj obraz
        model_type = self.current_model_type
        preprocess_input = self.preprocessing_functions[model_type]
        processed_image = preprocess_input(image.copy())
        
        # Dodaj wymiar wsadowy
        processed_image = np.expand_dims(processed_image, axis=0)
        
        # Wykonaj predykcję
        predictions = self.current_model.predict(processed_image, verbose=0)
        
        return predictions[0]

    def get_class_names(self):
        """Pobierz nazwy klas z katalogu testowego."""
        try:
            class_names = sorted(os.listdir(os.path.join("archive", "test")))
            return class_names
        except Exception as e:
            print(f"Błąd podczas pobierania nazw klas: {e}")
            return [] 
