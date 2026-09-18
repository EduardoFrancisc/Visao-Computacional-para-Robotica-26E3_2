# -*- coding: utf-8 -*-
"""item_b.py

Exemplo de treinamento e avaliação de um classificador SVM usando HOG.
O código realiza as etapas solicitadas:

1. Carrega 100 amostras positivas e 100 negativas a partir de diretórios
   ``data/pos`` e ``data/neg``. As imagens são redimensionadas para 64×128 px.
2. Extrai descritores HOG com ``cv2.HOGDescriptor().compute()``.
3. Treina um ``sklearn.svm.SVC`` com kernel RBF.
4. Avalia o modelo (acurácia, precisão e recall) em um conjunto de teste.
5. Executa detecção por janela deslizante sobre uma imagem de teste e
   desenha as detecções.

Além disso, o script contém comentários que comparam o custo computacional
da abordagem de janela deslizante com o de detectores baseados em redes
convolucionais (ex.: YOLO).

Obs.: Este script é auto‑contido, porém depende de duas pastas de imagens:
``data/pos`` (objetos) e ``data/neg`` (background). Substitua os caminhos
ou preencha as pastas com suas próprias imagens antes de executar.
"""

import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from pathlib import Path

# ------------------------------------------------------------
# Configurações
# ------------------------------------------------------------
IMG_W, IMG_H = 64, 128               # tamanho padrão usado em many detectors (ex.: pedestrian)
# Synthetic dataset generation (executed only if needed)
BASE_DIR = Path(__file__).resolve().parents[0]

# Diretórios de dataset sintético
POS_DIR_PATH = BASE_DIR / "data" / "pos"
NEG_DIR_PATH = BASE_DIR / "data" / "neg"
TEST_IMG_PATH = BASE_DIR / "data" / "test.jpg"

# Gera dataset sintético caso não exista
if not (POS_DIR_PATH.exists() and any(POS_DIR_PATH.iterdir())):
    rng = np.random.default_rng(42)
    POS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    for i in range(120):
        img = rng.integers(0, 50, (128, 64, 3), dtype=np.uint8)
        cv2.circle(img, (32 + int(rng.integers(-3, 4)), 25), 10, (220, 220, 220), -1)
        cv2.rectangle(img, (23, 38), (41, 95), (210, 210, 210), -1)
        cv2.imwrite(str(POS_DIR_PATH / f'pos_{i:03d}.png'), img)

if not (NEG_DIR_PATH.exists() and any(NEG_DIR_PATH.iterdir())):
    rng = np.random.default_rng(42)
    NEG_DIR_PATH.mkdir(parents=True, exist_ok=True)
    for i in range(120):
        img = rng.integers(0, 120, (128, 64, 3), dtype=np.uint8)
        for _ in range(int(rng.integers(2, 7))):
            cv2.line(
                img,
                (int(rng.integers(0, 64)), int(rng.integers(0, 128))),
                (int(rng.integers(0, 64)), int(rng.integers(0, 128))),
                tuple(map(int, rng.integers(0, 180, 3))),
                2,
            )
        cv2.imwrite(str(NEG_DIR_PATH / f'neg_{i:03d}.png'), img)

# Cria cena de teste sintética
TEST_IMG_PATH.parent.mkdir(parents=True, exist_ok=True)
if not TEST_IMG_PATH.exists():
    rng = np.random.default_rng(42)
    scene = rng.integers(0, 55, (384, 320, 3), dtype=np.uint8)
    for cx, cy in [(95, 165), (225, 190)]:
        cv2.circle(scene, (cx, cy - 45), 12, (225, 225, 225), -1)
        cv2.rectangle(scene, (cx - 11, cy - 30), (cx + 11, cy + 55), (215, 215, 215), -1)
    cv2.imwrite(str(TEST_IMG_PATH), scene)

# Paths used by the rest of the script
POS_DIR = str(POS_DIR_PATH)
NEG_DIR = str(NEG_DIR_PATH)
TEST_IMG = str(TEST_IMG_PATH)

# ------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------
def load_and_preprocess(folder: str, label: int, max_samples: int = 100) -> tuple[list[np.ndarray], list[int]]:
    """Carrega até ``max_samples`` imagens da *folder*, redimensiona para 64×128 e
    retorna a lista de imagens (como arrays ``np.uint8``) e a lista de rótulos.
    """
    images = []
    labels = []
    for i, fname in enumerate(sorted(os.listdir(folder))):
        if i >= max_samples:
            break
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue  # ignora arquivos que não são imagens válidas
        img_resized = cv2.resize(img, (IMG_W, IMG_H))
        images.append(img_resized)
        labels.append(label)
    return images, labels

def compute_hog_features(images: list[np.ndarray]) -> np.ndarray:
    """Extrai descritores HOG de todas as imagens da lista.

    ``cv2.HOGDescriptor`` usa parâmetros padrão (cellSize=8x8, blockSize=16x16,
    blockStride=8x8, nbins=9).  As imagens já estão em escala de cinza.
    """
    hog = cv2.HOGDescriptor(
        _winSize=(IMG_W, IMG_H),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    feats = []
    for img in images:
        descriptor = hog.compute(img).flatten()
        feats.append(descriptor)
    return np.array(feats, dtype=np.float32)

def train_svm(X_train: np.ndarray, y_train: np.ndarray) -> SVC:
    """Treina um classificador SVM com kernel RBF.

    Parâmetros padrão (C=1.0, gamma='scale') são suficientes para o exemplo.
    """
    svm = SVC(kernel="rbf", probability=True)
    svm.fit(X_train, y_train)
    return svm

def evaluate_model(svm: SVC, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Calcula acurácia, precisão e recall no conjunto de teste."""
    y_pred = svm.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }
    return metrics

def sliding_window(image: np.ndarray, stepSize: int, windowSize: tuple[int, int]):
    """Gerador que varre a imagem com passos ``stepSize``.

    Yields ``(x, y, window)`` onde ``window`` tem dim ``windowSize``.
    """
    for y in range(0, image.shape[0] - windowSize[1] + 1, stepSize):
        for x in range(0, image.shape[1] - windowSize[0] + 1, stepSize):
            yield (x, y, image[y : y + windowSize[1], x : x + windowSize[0]])

def detect_with_sliding_window(svm: SVC, image: np.ndarray, step: int = 8, thresh: float = 0.7):
    """Aplica a janela deslizante na ``image`` e devolve retângulos detectados.

    ``thresh`` corresponde à probabilidade mínima para considerar uma região
    como positiva.  O custo computacional desta estratégia é:

    * **Número de avaliações** = (W‑w)/step × (H‑h)/step
      onde (W,H) são as dimensões da imagem e (w,h) o tamanho da janela.
    * Cada avaliação envolve o cálculo do descritor HOG (≈ 4 kB) e a
      inferência do SVM (um pequeno número de operações de distância euclidiana).

    Em comparação, **YOLO** processa a imagem inteira com uma única passagem de
    rede convolucional, realizando ~10⁶ a 10⁸ multiplicações‑acúmulo (dependendo da
    versão).  Embora o custo da rede seja maior por operação, ele é **independente**
    do número de janelas e, na prática, costuma ser **mais rápido** em GPUs e
    até mesmo em CPUs modernas.

    Portanto, a janela deslizante tem complexidade O(N) em relação ao número de
    posições, enquanto YOLO tem complexidade O(1) por imagem.
    """
    detections = []
    hog = cv2.HOGDescriptor(
        _winSize=(IMG_W, IMG_H),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    for (x, y, win) in sliding_window(image, step, (IMG_W, IMG_H)):
        if win.shape[0] != IMG_H or win.shape[1] != IMG_W:
            continue
        feat = hog.compute(win).flatten().reshape(1, -1)
        prob = svm.predict_proba(feat)[0][1]
        if prob >= thresh:
            detections.append((x, y, prob))
    return detections

def draw_detections(image: np.ndarray, detections: list[tuple[int, int, float]]) -> np.ndarray:
    """Desenha retângulos verdes nas regiões detectadas."""
    img_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for (x, y, prob) in detections:
        cv2.rectangle(img_vis, (x, y), (x + IMG_W, y + IMG_H), (0, 255, 0), 2)
        cv2.putText(
            img_vis,
            f"{prob:.2f}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return img_vis

# ------------------------------------------------------------
# Execução principal
# ------------------------------------------------------------
if __name__ == "__main__":
    pos_imgs, pos_labels = load_and_preprocess(POS_DIR, label=1)
    neg_imgs, neg_labels = load_and_preprocess(NEG_DIR, label=0)
    if len(pos_imgs) < 1 or len(neg_imgs) < 1:
        raise RuntimeError("Pastas de dados vazias ou não encontradas. Verifique POS_DIR e NEG_DIR.")

    X = compute_hog_features(pos_imgs + neg_imgs)
    y = np.array(pos_labels + neg_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    svm_model = train_svm(X_train, y_train)

    metrics = evaluate_model(svm_model, X_test, y_test)
    print("=== Métricas de avaliação ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    test_img = cv2.imread(TEST_IMG, cv2.IMREAD_GRAYSCALE)
    if test_img is None:
        raise RuntimeError(f"Imagem de teste não encontrada em {TEST_IMG}")
    detections = detect_with_sliding_window(svm_model, test_img, step=8, thresh=0.7)
    print(f"Detecções encontradas: {len(detections)}")

    output = draw_detections(test_img, detections)
    cv2.imwrite("detected_output.jpg", output)
    print("Resultado salvo como 'detected_output.jpg'.")
