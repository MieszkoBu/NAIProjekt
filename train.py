"""Moduł do trenowania modelu klasyfikacji obrazów."""

import os
from typing import Any

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

train_dir = os.path.join(os.getcwd(), "src", "archive", "train")
validation_dir = os.path.join(os.getcwd(), "src", "archive", "validation")
test_dir = os.path.join(os.getcwd(), "src", "archive", "test")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=128, class_mode="categorical"
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=128, class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(150, 150), batch_size=128, class_mode="categorical"
)

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False

model = Sequential(
    [
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(train_generator.num_classes, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

history = model.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr],
)

model_save_path = os.path.join(os.getcwd(), "models", "trained_model.keras")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)

os.makedirs("src/models", exist_ok=True)

try:
    print("Zapisywanie modelu w formacie keras...")
    model.save('src/models/trained_model.keras', save_format='keras')
    print("Model zapisany pomyślnie!")
except Exception as e:
    print(f"Błąd podczas zapisywania modelu keras: {e}")

try:
    print("Zapisywanie modelu w formacie h5...")
    model.save('src/models/trained_model.h5', save_format='h5')
    print("Model h5 zapisany pomyślnie!")
except Exception as e:
    print(f"Błąd podczas zapisywania modelu h5: {e}")

try:
    print("Zapisywanie modelu w formacie SavedModel...")
    tf.saved_model.save(model, 'src/models/saved_model')
    print("Model SavedModel zapisany pomyślnie!")
except Exception as e:
    print(f"Błąd podczas zapisywania SavedModel: {e}")

def plot_training_history(history: Any) -> None:
    """Generuj wykres historii treningu.

    Args:
        history: Historia treningu z Keras
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.close()


plot_training_history(history)
