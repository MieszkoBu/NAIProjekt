"""Moduł do testowania wytrenowanego modelu klasyfikacji obrazów."""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
from tensorflow.keras.preprocessing.image import img_to_array, load_img

test_dir = os.path.join(os.getcwd(), "src", "archive", "test")
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(150, 150), batch_size=128, class_mode="categorical", shuffle=False
)

model = load_model("src/models/trained_model.keras")

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Dokładność na zbiorze testowym: {test_accuracy * 100:.2f}%")

images, labels = next(test_generator)

predictions = model.predict(images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(labels, axis=1)

class_labels = {v: k for k, v in test_generator.class_indices.items()}

plt.figure(figsize=(15, 10))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(array_to_img(images[i]))
    true_label = class_labels[true_classes[i]]
    predicted_label = class_labels[predicted_classes[i]]
    plt.title(f"True: {true_label}\nPred: {predicted_label}", color="green" if true_label == predicted_label else "red")
    plt.axis("off")
plt.tight_layout()
plt.show()
