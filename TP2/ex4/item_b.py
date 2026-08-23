import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import PCA
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "cnn_fashion_mnist.keras")
NUM_CLASSES = 10
SAMPLES_PER_CLASS = 4
TOTAL_SAMPLES = NUM_CLASSES * SAMPLES_PER_CLASS  # 40 imagens (≥20, ≥4 por classe)

CLASS_NAMES = [
    "Camiseta", "Calça", "Pullover", "Vestido", "Casaco",
    "Sandália", "Camisa", "Tênis", "Bolsa", "Bota"
]

# ====================== MODELO CNN (mesmo do Item A) ======================

def build_model():
    return keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])


def get_trained_model():
    """Carrega modelo salvo ou treina do zero."""
    if os.path.exists(MODEL_PATH):
        print(f"Carregando modelo de: {MODEL_PATH}")
        return keras.models.load_model(MODEL_PATH)

    print("Modelo não encontrado. Treinando do zero (10 épocas)...")
    (x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)

    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)
    model.save(MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")
    return model

# ====================== DADOS DE TESTE ======================

(_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_test_norm = x_test.astype("float32") / 255.0
x_test_norm = np.expand_dims(x_test_norm, -1)

# Seleciona SAMPLES_PER_CLASS imagens por classe
selected_idx = []
for c in range(NUM_CLASSES):
    idx = np.where(y_test == c)[0][:SAMPLES_PER_CLASS]
    selected_idx.extend(idx)
selected_idx = np.array(selected_idx)

x_selected = x_test_norm[selected_idx]
y_selected = y_test[selected_idx]
x_selected_raw = x_test[selected_idx]  # uint8 para ORB

print(f"Imagens selecionadas: {len(selected_idx)} ({SAMPLES_PER_CLASS} por classe)")

# ====================== 1. FEATURES DA CNN (penúltima Dense) ======================

model = get_trained_model()

# Extrator: remove a última Dense(softmax) → saída da Dense(128)
feature_extractor = keras.Sequential(model.layers[:-1])

cnn_features = feature_extractor.predict(x_selected, verbose=0)
print(f"Features CNN: shape = {cnn_features.shape}")  # (40, 128)

# ====================== 2. PCA 2D — FEATURES CNN ======================

pca_cnn = PCA(n_components=2)
cnn_2d = pca_cnn.fit_transform(cnn_features)
print(f"Variância explicada PCA (CNN): {pca_cnn.explained_variance_ratio_.sum():.2%}")

# ====================== 3. DESCRITORES ORB ======================

orb = cv2.ORB_create(nfeatures=128)

def orb_descriptor_vector(img_gray, target_dim=128):
    """Extrai descritores ORB e retorna vetor de dimensão fixa."""
    kp, desc = orb.detectAndCompute(img_gray, None)
    if desc is None or len(desc) == 0:
        return np.zeros(target_dim * 32, dtype=np.float32)
    if len(desc) < target_dim:
        desc = np.vstack([desc, np.zeros((target_dim - len(desc), 32), dtype=np.uint8)])
    else:
        desc = desc[:target_dim]
    return desc.flatten().astype(np.float32)


orb_features = np.array([orb_descriptor_vector(img) for img in x_selected_raw])
print(f"Features ORB: shape = {orb_features.shape}")  # (40, 4096)

pca_orb = PCA(n_components=2)
orb_2d = pca_orb.fit_transform(orb_features)
orb_var = np.nansum(pca_orb.explained_variance_ratio_)
print(f"Variância explicada PCA (ORB): {orb_var:.2%}")

# ====================== 4. SCATTER PLOTS ======================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
cmap = plt.get_cmap("tab10")

for c in range(NUM_CLASSES):
    mask = y_selected == c
    ax1.scatter(cnn_2d[mask, 0], cnn_2d[mask, 1], c=[cmap(c)],
                label=CLASS_NAMES[c], s=60, edgecolors="k", linewidths=0.5)
    ax2.scatter(orb_2d[mask, 0], orb_2d[mask, 1], c=[cmap(c)],
                label=CLASS_NAMES[c], s=60, edgecolors="k", linewidths=0.5)

ax1.set_title("PCA 2D — Features CNN (Dense 128)")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.legend(fontsize=7, loc="best")
ax1.grid(True, alpha=0.3)

ax2.set_title("PCA 2D — Descritores ORB")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.legend(fontsize=7, loc="best")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(SCRIPT_DIR, "pca_cnn_vs_orb.png")
plt.savefig(plot_path, dpi=150)
print(f"\nGráfico salvo em: {plot_path}")
plt.show()


print("\n" + "=" * 60)
print("  RESUMO")
print("=" * 60)
print(f"  CNN features: {cnn_features.shape[1]}D → PCA 2D "
      f"(variância: {pca_cnn.explained_variance_ratio_.sum():.2%})")
print(f"  ORB features: {orb_features.shape[1]}D → PCA 2D "
      f"(variância: {orb_var:.2%})")
print("  → CNN produz clusters mais separados = representações semânticas melhores")
print("  → ORB captura padrões locais, útil para matching, não para classificação")
print("=" * 60)
