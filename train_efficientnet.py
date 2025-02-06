"""Moduł trenujący model EfficientNetB0 do rozpoznawania składników."""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from naiprojekt.model_metrics import ModelMetrics
import time
import numpy as np

def preprocess_image(image):
    """Przygotuj obraz do przetwarzania."""
    # ImageDataGenerator przekazuje obrazy jako tablice numpy
    return tf.keras.applications.efficientnet.preprocess_input(image)

def train_efficientnet_model():
    """Trenuj model EfficientNetB0."""
    # Przygotuj ścieżki
    train_dir = os.path.join("archive", "train")
    test_dir = os.path.join("archive", "test")
    
    # Przygotuj generatory danych
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='constant'
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    # Załaduj model bazowy EfficientNetB0
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    # Zamroź warstwy bazowe
    for layer in base_model.layers:
        layer.trainable = False  # Najpierw zamroź wszystkie

    # Odmroź ostatnie bloki
    for layer in base_model.layers[-50:]:  # Odmrażamy więcej warstw
        layer.trainable = True

    # Dodaj własne warstwy
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

    # Stwórz model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Kompiluj model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall', 'AUC']
    )

    # Trenuj model
    metrics = ModelMetrics("EfficientNetB0")
    start_time = time.time()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        )
    ]

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=100,
        callbacks=callbacks,
        validation_data=test_generator,
        validation_steps=test_generator.samples // test_generator.batch_size
    )
    end_time = time.time()

    # Zapisz model i metryki
    try:
        # Najpierw zapisz metryki
        metrics.measure_training_time(start_time, end_time)
        metrics.calculate_metrics(model, test_generator)
        metrics.save_metrics()
        print("Zapisano metryki modelu")
        
        # Potem spróbuj zapisać model
        try:
            # Zapisz cały model w formacie .keras
            model.save('src/models/trained_model_efficientnet.keras', 
                      save_format='keras_v3',
                      save_traces=True,
                      include_optimizer=True)
            print("Zapisano model w formacie keras")
        except Exception as e:
            print(f"Błąd podczas zapisywania modelu: {e}")
            try:
                # Jeśli nie udało się zapisać całego modelu, spróbuj zapisać same wagi
                model.save_weights('src/models/trained_model_efficientnet_weights.h5')
                print("Zapisano wagi modelu")
            except Exception as e:
                print(f"Błąd podczas zapisywania wag modelu: {e}")
                
        # Zapisz historię treningu
        np.save('training_history_efficientnet.npy', history.history)
        print("Zapisano historię treningu")
        
    except Exception as e:
        print(f"Błąd podczas zapisywania modelu/metryk: {e}")
        import traceback
        traceback.print_exc()

    # Narysuj wykres uczenia
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Dokładność treningu')
    plt.plot(history.history['val_accuracy'], label='Dokładność walidacji')
    plt.title('Dokładność modelu EfficientNetB0')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Strata treningu')
    plt.plot(history.history['val_loss'], label='Strata walidacji')
    plt.title('Strata modelu EfficientNetB0')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_efficientnet.png')
    plt.close()

if __name__ == "__main__":
    train_efficientnet_model() 
