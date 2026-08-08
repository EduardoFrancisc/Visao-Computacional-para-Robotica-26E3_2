import cv2
import numpy as np 
from pathlib import Path

def obter_imagem_base(nome_arquivo: str = "paisagem.jpg"):
    caminho = Path(__file__).resolve().parent / nome_arquivo
    
    # Tenta carregar uma fotografia já existente.
    imagem = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    
    altura, largura = imagem.shape[:2]
    
    if altura < 480 or largura < 480:
        escala = max(480 / altura, 480 / largura)
        nova_largura = int(round(largura * escala))
        nova_altura = int(round(altura * escala))
        
        n_imagem = cv2.resize(
            imagem,
            (nova_largura, nova_altura),
            interpolation=cv2.INTER_CUBIC,
        )
        return n_imagem
    else:
        return imagem

def redimensionar_quadrado(imagem, lado: int = 320):
    """Redimensiona a imagem para facilitar a montagem de painéis didáticos."""
    return cv2.resize(imagem, (lado, lado), interpolation=cv2.INTER_AREA)

def adicionar_titulo(imagem, titulo: str):
    """Escreve um título na parte superior de uma cópia da imagem."""
    saida = imagem.copy()
    cv2.rectangle(saida, (0, 0), (saida.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        saida,
        titulo,
        (10, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return saida

def canal_para_bgr(canal):
    """Converte um canal de uma dimensão para três canais BGR."""
    return cv2.cvtColor(canal, cv2.COLOR_GRAY2BGR)

def alterar_saturacao(imagem_bgr, fator: float):
    """Multiplica somente o canal S do espaço HSV pelo fator informado."""
    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)

    # Converte temporariamente para float32 para evitar estouro de uint8.
    hsv_float = hsv.astype(np.float32)
    hsv_float[:, :, 1] *= fator

    # Saturação deve permanecer no intervalo de 0 a 255.
    hsv_float[:, :, 1] = np.clip(hsv_float[:, :, 1], 0, 255)

    hsv_alterado = hsv_float.astype(np.uint8)
    return cv2.cvtColor(hsv_alterado, cv2.COLOR_HSV2BGR)

def main() -> None:
    imagem = obter_imagem_base()
    
    # OpenCV carrega imagens coloridas na ordem BGR.
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
    
    # HSV:
    # H (Hue / Matiz): identifica a família de cor. No OpenCV varia de 0 a 179.
    # S (Saturation / Saturação): intensidade ou pureza da cor, de 0 a 255.
    # V (Value / Valor): brilho do pixel, de 0 a 255.
    canal_h, canal_s, canal_v = cv2.split(hsv)
    
    # LAB:
    # L: luminosidade perceptual, de escuro para claro.
    # A: eixo cromático aproximado entre verde e vermelho.
    # B: eixo cromático aproximado entre azul e amarelo.
    canal_l, canal_a, canal_b = cv2.split(lab)
    
    # BLOCO DESCOMENTADO E ADAPTADO
    sat_0 = alterar_saturacao(imagem, 0.0)
    sat_50 = alterar_saturacao(imagem, 0.5)
    sat_150 = alterar_saturacao(imagem, 1.5)
    
    imagens_sat = [
        adicionar_titulo(redimensionar_quadrado(sat_0), "Saturacao: 0%"),
        adicionar_titulo(redimensionar_quadrado(sat_50), "Saturacao: 50%"),
        adicionar_titulo(redimensionar_quadrado(sat_150), "Saturacao: 150%"),
    ]
    
    original = redimensionar_quadrado(imagem)
    imagens = [
        adicionar_titulo(original, "BGR original"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(canal_h)), "HSV - H (matiz)"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(canal_s)), "HSV - S (saturacao)"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(canal_v)), "HSV - V (brilho)"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(canal_l)), "LAB - L (luminosidade)"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(canal_a)), "LAB - A (verde-vermelho)"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(canal_b)), "LAB - B (azul-amarelo)"),
    ]
    
    # Cria células pretas para completar uma grade 3 x 4.
    celula_vazia = imagens[0] * 0
    celula_vazia_txt = celula_vazia.copy()
    cv2.putText(
        celula_vazia_txt,
        "10 imagens no total",
        (28, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    
    # MONTAGEM DO NOVO GRID 3x4
    linha_1 = cv2.hconcat(imagens[0:4])
    linha_2 = cv2.hconcat(imagens[4:7] + [celula_vazia_txt])
    linha_3 = cv2.hconcat(imagens_sat + [celula_vazia])
    
    painel = cv2.vconcat([linha_1, linha_2, linha_3])
    
    print("Shapes dos espaços de cor:")
    print(f"BGR: {imagem.shape}")
    print(f"HSV: {hsv.shape}")
    print(f"LAB: {lab.shape}")
    print("Pressione Q ou Esc para fechar.")
    
    cv2.imshow("Exercicio - canais HSV e LAB + Saturacao", painel)
    while True:
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q") or tecla == ord("Q"):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()