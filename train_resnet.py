"""Moduł trenujący model ResNet50 do rozpoznawania składników."""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from model_metrics import ModelMetrics
import numpy as np

def preprocess_image(image):
    """Przygotuj obraz do przetwarzania."""
    # Konwersja na float32
    image = image.astype('float32')
    # RGB -> BGR
    image = image[..., ::-1]
    # Odejmowanie średnich RGB
    image[..., 0] -= 103.939  # średnia dla B
    image[..., 1] -= 116.779  # średnia dla G
    image[..., 2] -= 123.68   # średnia dla R
    return image

def train_resnet_model():
    """Trenuj model ResNet50."""
    metrics = ModelMetrics("ResNet50")
    start_time = time.time()

    # Przygotuj ścieżki
    train_dir = os.path.join(os.getcwd(), "src", "archive", "train")
    test_dir = os.path.join(os.getcwd(), "src", "archive", "test")
    
    # Przygotuj generatory danych
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
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

    # Załaduj model bazowy ResNet50
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    # Zamroź warstwy bazowe
    for layer in base_model.layers:
        layer.trainable = False

    # Odmroź ostatnie bloki
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    # Dodaj własne warstwy
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    predictions = tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    # Kompiluj model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall', 'AUC']
    )

    # Trenuj model
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
        validation_steps=test_generator.samples // test_generator.batch_size,
        verbose=1
    )

    end_time = time.time()

    # Zapisz model i metryki
    try:
        # Upewnij się, że katalog istnieje
        os.makedirs('src/models', exist_ok=True)
        print(f"Katalog docelowy: {os.path.abspath('src/models')}")
        
        # Najpierw zapisz metryki
        metrics.measure_training_time(start_time, end_time)
        metrics.calculate_metrics(model, test_generator)
        metrics.save_metrics()
        print("Zapisano metryki modelu")
        
        # Potem spróbuj zapisać model
        print("Rozpoczynam zapisywanie modelu...")
        try:
            model.save('src/models/trained_model_resnet.keras')
            print("Zapisano model w formacie SavedModel")
        except Exception as e:
            print(f"Błąd podczas zapisywania modelu w formacie SavedModel: {e}")
            try:
                model.save_weights('src/models/trained_model_resnet.h5')
                print("Zapisano wagi modelu")
            except Exception as e:
                print(f"Błąd podczas zapisywania wag modelu: {e}")
                
        # Zapisz historię treningu
        np.save('training_history_resnet.npy', history.history)
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
    plt.title('Dokładność modelu ResNet50')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Strata treningu')
    plt.plot(history.history['val_loss'], label='Strata walidacji')
    plt.title('Strata modelu ResNet50')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_resnet.png')
    plt.close()

if __name__ == "__main__":
    train_resnet_model() 