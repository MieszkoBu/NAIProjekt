"""Moduł do obliczania i porównywania metryk modeli."""

import time
import numpy as np
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

class ModelMetrics:
    def __init__(self, model_name: str):
        """Inicjalizuj metryki dla modelu."""
        self.model_name = model_name
        self.training_time = 0
        self.model_size = 0
        self.inference_time = 0
        self.history = None
        self.metrics = {}

    def measure_training_time(self, start_time: float, end_time: float) -> None:
        """Zmierz czas treningu."""
        self.training_time = end_time - start_time

    def measure_model_size(self, model) -> None:
        """Zmierz rozmiar modelu."""
        import os
        model.save(f'temp_{self.model_name}.keras')
        self.model_size = os.path.getsize(f'temp_{self.model_name}.keras') / (1024 * 1024)  # MB
        os.remove(f'temp_{self.model_name}.keras')

    def calculate_metrics(self, model, test_generator) -> None:
        """Oblicz metryki modelu."""
        # Dodaj metryki
        precision_metric = Precision()
        recall_metric = Recall()
        auc_metric = AUC()

        # Zmierz czas inferencji
        start_time = time.time()
        predictions = model.predict(test_generator)
        self.inference_time = (time.time() - start_time) / len(test_generator)

        # Oblicz true labels
        true_labels = test_generator.classes
        predicted_labels_one_hot = tf.one_hot(np.argmax(predictions, axis=1), depth=len(test_generator.class_indices))

        # Oblicz metryki
        self.metrics = {
            'Precision': precision_metric(tf.one_hot(true_labels, depth=len(test_generator.class_indices)), 
                                        predicted_labels_one_hot).numpy(),
            'Recall': recall_metric(tf.one_hot(true_labels, depth=len(test_generator.class_indices)), 
                                  predicted_labels_one_hot).numpy(),
            'AUC': auc_metric(tf.one_hot(true_labels, depth=len(test_generator.class_indices)), 
                             predictions).numpy(),
            'Training Time': self.training_time,
            'Model Size (MB)': self.model_size,
            'Inference Time (s)': self.inference_time
        }

        # Stwórz macierz pomyłek
        predicted_labels = np.argmax(predictions, axis=1)
        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Macierz pomyłek - {self.model_name}')
        plt.ylabel('Prawdziwa etykieta')
        plt.xlabel('Przewidziana etykieta')
        plt.savefig(f'confusion_matrix_{self.model_name}.png')
        plt.close()

        # Zapisz raport klasyfikacji
        report = classification_report(true_labels, predicted_labels, target_names=test_generator.class_indices.keys())
        with open(f'classification_report_{self.model_name}.txt', 'w') as f:
            f.write(report)

    def save_metrics(self) -> None:
        """Zapisz wszystkie metryki do pliku."""
        with open(f'metrics_{self.model_name}.txt', 'w') as f:
            f.write(f"Metryki dla modelu {self.model_name}:\n")
            f.write("-" * 50 + "\n")
            for metric_name, value in self.metrics.items():
                f.write(f"{metric_name}: {value}\n") 