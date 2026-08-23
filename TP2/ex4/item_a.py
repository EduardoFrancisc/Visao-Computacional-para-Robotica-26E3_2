"""
TP2 - Exercício 4 - Item A
CNN do zero com TensorFlow/Keras para classificação de imagens (Fashion-MNIST).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EPOCHS = 15
BATCH_SIZE = 64
NUM_CLASSES = 10

CLASS_NAMES = [
    "Camiseta", "Calça", "Pullover", "Vestido", "Casaco",
    "Sandália", "Camisa", "Tênis", "Bolsa", "Bota"
]

# ====================== CARREGAMENTO E PRÉ-PROCESSAMENTO ======================

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normaliza para [0, 1] e adiciona canal (28x28x1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Separa 10% do treino para validação
val_split = int(0.1 * len(x_train))
x_val, y_val = x_train[:val_split], y_train[:val_split]
x_train, y_train = x_train[val_split:], y_train[val_split:]

print(f"Treino: {x_train.shape} | Validação: {x_val.shape} | Teste: {x_test.shape}")

# ====================== ARQUITETURA DA CNN ======================
# 2 blocos Conv2D + MaxPooling, Flatten, Dense, Softmax

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),

    # Bloco 1
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Bloco 2
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Classificador
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax"),
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ====================== TREINAMENTO ======================

history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    verbose=1,
)

# ====================== AVALIAÇÃO NO TESTE ======================

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n{'=' * 50}")
print(f"  Acurácia no conjunto de teste: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  Loss no conjunto de teste:     {test_loss:.4f}")
print(f"{'=' * 50}\n")

# ====================== ANÁLISE DE OVERFITTING ======================
#
# Para identificar overfitting, comparamos as curvas de treino e validação:
#   - Se a acurácia de treino continua subindo enquanto a de validação estagna
#     ou cai, há overfitting (o modelo memorizou o treino).
#   - Se o loss de treino cai mas o de validação sobe, é outro sinal clássico.
#
# Mitigações aplicadas nesta arquitetura:
#   - Dropout(0.3) antes da camada final para regularização.
#   - Arquitetura relativamente compacta para o tamanho do dataset.
#
# Se as curvas divergirem significativamente, considerar:
#   - Aumentar Dropout ou adicionar BatchNormalization
#   - Aplicar data augmentation
#   - Reduzir o número de épocas (early stopping)

train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

gap_acc = train_acc[-1] - val_acc[-1]
gap_loss = val_loss[-1] - train_loss[-1]

print("ANÁLISE DE OVERFITTING:")
if gap_acc > 0.05:
    print(f"  ⚠ INDÍCIO DE OVERFITTING: gap de acurácia = {gap_acc:.4f}")
    print(f"    Treino={train_acc[-1]:.4f} vs Validação={val_acc[-1]:.4f}")
else:
    print(f"  ✓ Sem indício significativo de overfitting (gap = {gap_acc:.4f})")
if gap_loss > 0.1:
    print(f"  ⚠ Loss de validação divergindo: gap = {gap_loss:.4f}")
else:
    print(f"  ✓ Loss estável (gap = {gap_loss:.4f})")

# ====================== GRÁFICOS ======================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, EPOCHS + 1)

# Acurácia
ax1.plot(epochs_range, train_acc, "b-o", label="Treino", markersize=4)
ax1.plot(epochs_range, val_acc, "r-o", label="Validação", markersize=4)
ax1.set_title("Acurácia — Treino vs Validação")
ax1.set_xlabel("Época")
ax1.set_ylabel("Acurácia")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(epochs_range, train_loss, "b-o", label="Treino", markersize=4)
ax2.plot(epochs_range, val_loss, "r-o", label="Validação", markersize=4)
ax2.set_title("Loss — Treino vs Validação")
ax2.set_xlabel("Época")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

plot_path = os.path.join(SCRIPT_DIR, "curvas_treinamento.png")
plt.savefig(plot_path, dpi=150)
print(f"\nGráficos salvos em: {plot_path}")
plt.show()
