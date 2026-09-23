from pathlib import Path
import cv2
from seg_utils import salvar_cenas, colorir_mascara, sobrepor, porcentagens

ROOT = Path(__file__).resolve().parent

salvar_cenas(ROOT, 5)

img = cv2.imread(str(ROOT / "imagens" / "cena_01.png"))
mask = cv2.imread(str(ROOT / "imagens" / "cena_01_mask.png"), cv2.IMREAD_GRAYSCALE)
color = colorir_mascara(mask)
out = sobrepor(img, mask, alpha=0.45)

cv2.imwrite(str(ROOT / "saidas" / "segmentacao_classes_coloridas.png"), color)
print("Máscara colorida salva em saidas/segmentacao_classes_coloridas.png")

cv2.imwrite(str(ROOT / "saidas" / "mascara_semi_transparente.png"), out)
print("Overlay semitransparente salva em saidas/mascara_semi_transparente.png")

out_p = sobrepor(img, mask)
y = 25

for k, v in porcentagens(mask).items():
    print(f"{k:12s}: {v:5.2f}%")
    cv2.putText(out_p, f"{k}: {v:.1f}%", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    y += 25

cv2.imwrite(str(ROOT / "saidas" / "proporcoes.png"), out_p)
print("Frame final anotado salvo em saidas/proporcoes.png")