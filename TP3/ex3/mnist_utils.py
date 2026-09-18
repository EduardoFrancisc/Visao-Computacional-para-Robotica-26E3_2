"""
Funções auxiliares compartilhadas pelos exemplos da aula.

A ideia é reduzir repetição nos scripts, mantendo cada exemplo curto e didático.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


IMG_SIZE = 28
NUM_CLASSES = 10


def load_mnist(flatten: bool = False, one_hot: bool = False):
    """Carrega MNIST e aplica normalização para [0, 1]."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    if flatten:
        x_train = x_train.reshape((-1, 28 * 28))
        x_test = x_test.reshape((-1, 28 * 28))
    else:
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]

    if one_hot:
        y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
        y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    return (x_train, y_train), (x_test, y_test)


def build_mlp() -> tf.keras.Model:
    """MLP com pelo menos duas camadas densas."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28 * 28,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="mlp_mnist",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(dropout: float = 0.0) -> tf.keras.Model:
    """CNN com dois blocos Conv2D + MaxPooling."""
    layers = [
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
    ]
    if dropout > 0:
        layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"))

    model = tf.keras.Sequential(layers, name="cnn_mnist")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class EpochTimer(tf.keras.callbacks.Callback):
    """Mede o tempo de cada época para comparação didática."""

    def on_train_begin(self, logs=None):
        self.epoch_times: List[float] = []

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.perf_counter() - self._start)


def plot_history(history, title: str, output_path: str | Path | None = None) -> None:
    """Plota acurácia e loss de treino vs. validação."""
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist["accuracy"], label="treino")
    plt.plot(epochs, hist["val_accuracy"], label="validação")
    plt.title(f"{title} - Acurácia")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist["loss"], label="treino")
    plt.plot(epochs, hist["val_loss"], label="validação")
    plt.title(f"{title} - Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    plt.show()


def preprocess_digit_image(image_path: str | Path, invert_if_needed: bool = True):
    """Pré-processa uma foto de dígito para o formato esperado pelo MNIST.

    Pipeline exigido no enunciado:
    escala de cinza -> binarização Otsu -> resize 28x28 -> normalização.
    """
    image_path = Path(image_path)
    original = cv2.imread(str(image_path))
    if original is None:
        raise FileNotFoundError(f"Não foi possível abrir: {image_path}")

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu encontra automaticamente um limiar para separar fundo e traço.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # MNIST tem fundo preto e dígito claro. Muitas fotos têm o contrário.
    if invert_if_needed and np.mean(binary) > 127:
        binary = 255 - binary

    # Recorte automático da região do dígito para reduzir fundo sobrando.
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = binary[y : y + h, x : x + w]
    else:
        digit = binary

    # Coloca o dígito em uma imagem quadrada antes de redimensionar.
    side = max(digit.shape[:2])
    square = np.zeros((side, side), dtype=np.uint8)
    y0 = (side - digit.shape[0]) // 2
    x0 = (side - digit.shape[1]) // 2
    square[y0 : y0 + digit.shape[0], x0 : x0 + digit.shape[1]] = digit

    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    model_input = normalized.reshape(1, 28, 28, 1)

    return original, gray, binary, resized, model_input


def predict_real_digit(model: tf.keras.Model, image_path: str | Path) -> Dict[str, object]:
    """Executa pré-processamento e inferência em uma imagem real."""
    original, gray, binary, resized, model_input = preprocess_digit_image(image_path)
    probs = model.predict(model_input, verbose=0)[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
    return {
        "path": str(image_path),
        "original": original,
        "gray": gray,
        "binary": binary,
        "resized": resized,
        "input": model_input,
        "prediction": pred,
        "confidence": conf,
    }


def list_real_digit_images(folder: str | Path) -> List[Tuple[Path, int]]:
    """Procura arquivos 0.jpg, 1.jpg, ..., 9.jpg e retorna pares (caminho, rótulo)."""
    folder = Path(folder)
    samples = []
    for label in range(10):
        for ext in ("jpg", "jpeg", "png", "bmp"):
            path = folder / f"{label}.{ext}"
            if path.exists():
                samples.append((path, label))
                break
    return samples


def annotate_prediction(original_bgr, prediction: int, confidence: float):
    """Escreve a predição em cima da imagem original."""
    annotated = original_bgr.copy()
    text = f"Predito: {prediction} ({confidence:.1%})"
    cv2.putText(
        annotated,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def model_summary_row(name: str, model: tf.keras.Model, epoch_times, test_acc: float) -> dict:
    """Cria uma linha para tabela comparativa."""
    return {
        "modelo": name,
        "parametros": int(model.count_params()),
        "tempo_medio_epoca_s": float(np.mean(epoch_times)),
        "acuracia_teste": float(test_acc),
    }


def save_comparison_table(rows, path: str | Path = "resultados/comparacao_modelos.csv"):
    df = pd.DataFrame(rows)
    print("\nTabela comparativa:")
    print(df.to_string(index=False))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
