import cv2
import numpy as np
from pathlib import Path
from synthetic_utils import (
    generate_dataset, calibrate_from_paths, reprojection_errors
)

ROOT = Path(__file__).resolve().parent

#Gerar 15 imagens sintéticas de calibração
paths = generate_dataset(ROOT, n=15)

print(f"Foram geradas {len(paths)} imagens.")
for p in paths:
    print(" -", p.name)

# Calibrar a câmera usando as imagens geradas, obtém a matriz intrínseca K, coeficientes de distorção e os erros de reprojeção por imagem.
_, K, dist, rvecs, tvecs, objpoints, imgpoints, used, _ = \
    calibrate_from_paths(paths)

errors = reprojection_errors(
    K, dist, rvecs, tvecs, objpoints, imgpoints
)

for p, e in zip(used, errors):
    print(f"{p.name}: {e:.4f} px")

_, K, dist, *_ = calibrate_from_paths(paths)

img = cv2.imread(str(paths[6]))
corrigida = cv2.undistort(img, K, dist)

painel = np.hstack([img, corrigida])

cv2.putText(painel, "ORIGINAL", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
cv2.putText(painel, "CORRIGIDA", (img.shape[1]+30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,120,0), 3)

saida = ROOT / "saidas" / "7_painel_original_corrigida.png"
cv2.imwrite(str(saida), painel)
print("Painel salvo em:", saida)

#Imprimir a matriz intrínseca K, os 5 primeiros coeficientes de distorção e o erro médio de reprojeção.
print("K:\n", K)
print("\n5 coeficientes:", dist.ravel()[:5])
print(f"\nErro médio: {np.mean(errors):.4f} px")

print("\nA matriz intrínseca K define a geometria interna da câmera, convertendo coordenadas da cena em pixels.")
print("Os coeficientes de distorção descrevem como a lente distorce a imagem, e o erro médio indica a precisão da calibração.")
print("O valor de 0.0246 px sugere que a calibração foi bastante precisa, com pequenas discrepâncias entre os pontos projetados e os observados. O valor de erro considerável para aplicações robóticas é geralmente menor que 1 px, então 0.0246 px é excelente.")