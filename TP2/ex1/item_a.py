import cv2
import numpy as np
import os

def create_stereo_images(left_path, right_path):
    if os.path.exists(left_path) and os.path.exists(right_path):
        print("Imagens estéreo já existem.")
        return

    print("Criando imagens estéreo sintéticas...")
    #fundo preto
    img_left = np.zeros((300, 400), dtype=np.uint8)
    img_right = np.zeros((300, 400), dtype=np.uint8)

    noise = np.random.randint(0, 255, (300, 400), dtype=np.uint8)
    img_left[:] = noise
    img_right[:] = noise

    cv2.rectangle(img_left, (150, 100), (250, 200), 255, -1)
    cv2.rectangle(img_right, (120, 100), (220, 200), 255, -1)

    cv2.imwrite(left_path, img_left)
    cv2.imwrite(right_path, img_right)
    print("Imagens criadas com sucesso.")

def main():
    left_img_path = 'left.png'
    right_img_path = 'right.png'

    create_stereo_images(left_img_path, right_img_path)

    imgL = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    if imgL is None or imgR is None:
        print("Erro ao carregar as imagens.")
        return

    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    
    disparity = stereo.compute(imgL, imgR)

    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    valid_mask = disparity > 0

    disp_for_min = np.copy(disparity).astype(np.float32)
    disp_for_min[~valid_mask] = np.inf
    _, _, min_loc, _ = cv2.minMaxLoc(disp_for_min)

    disp_for_max = np.copy(disparity)
    disp_for_max[~valid_mask] = -1
    _, _, _, max_loc = cv2.minMaxLoc(disp_for_max)

    closest_val = disparity_normalized[max_loc[1], max_loc[0]]
    farthest_val = disparity_normalized[min_loc[1], min_loc[0]]

    print("--- Estimativa de Profundidade Relativa ---")
    print(f"Pixel mais próximo (maior disparidade): Coordenadas (x, y) = {max_loc}, Valor Normalizado = {closest_val}")
    print(f"Pixel mais distante (menor disparidade): Coordenadas (x, y) = {min_loc}, Valor Normalizado = {farthest_val}")
    print("-------------------------------------------")

    disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)

    cv2.imshow('Imagem Esquerda', imgL)
    cv2.imshow('Imagem Direita', imgR)
    cv2.imshow('Mapa de Disparidade (JET)', disparity_color)

    print("Pressione qualquer tecla para sair...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

'''
Em uma câmera estéreo embarcada em um drone, essa estimativa de profundidade se comportaria 
da seguinte forma:
1. Os objetos no solo muito abaixo ou obstáculos muito distantes apresentariam baixa disparidade (tendendo a zero).
2. Obstáculos iminentes (como galhos, paredes ou outras aeronaves) teriam alta disparidade, sendo detectados como os "pixels mais próximos".
3. As vibrações do motor e os movimentos rápidos (motion blur) poderiam prejudicar a correlação de blocos (StereoBM), gerando ruídos no mapa de disparidade (muitos pixels com disparidade inválida ou errada).
4. Ao apontar a câmera para o chão plano, a disparidade formaria um gradiente linear que ajudaria a estimar a altitude e inclinação (pitch/roll) do drone.
'''