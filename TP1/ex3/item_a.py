import cv2
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

def redimensionar_quadrado(imagem, lado: int = 320):
    """Redimensiona a imagem para facilitar a montagem de painéis didáticos."""
    return cv2.resize(imagem, (lado, lado), interpolation=cv2.INTER_AREA)

def main() -> None:
    imagem = obter_imagem_base()
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Suavização leve reduz ruído antes da binarização.
    cinza_suave = cv2.GaussianBlur(cinza, (5, 5), 0)

    # 1) Limiar global fixo: o valor 127 é aplicado a toda a imagem.
    _, global_bin = cv2.threshold(
        cinza_suave,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    # 2) Limiar adaptativo: cada região utiliza um limiar local.
    adaptativo = cv2.adaptiveThreshold(
        cinza_suave,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,  # Tamanho ímpar da vizinhança.
        5,   # Constante subtraída da média ponderada local.
    )

    # 3) Otsu: o valor inicial de threshold é ignorado e calculado automaticamente.
    # O método de Otsu é superior ao limiar global fixo em imagens com iluminação
    # não uniforme (ou variações globais de luminosidade) porque ele não depende de 
    # um valor estático (como 127). Ele calcula estatisticamente o melhor ponto 
    # de separação (minimizando a variância intra-classe no histograma), adaptando-se 
    # à distribuição real de pixels daquela imagem específica.
    limiar_otsu, otsu = cv2.threshold(
        cinza_suave,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    print(f"Limiar calculado pelo método de Otsu: {limiar_otsu:.2f}")

    painel = cv2.hconcat([
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(global_bin), 360), "Global: 127"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(adaptativo), 360), "Adaptativo"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(otsu), 360), f"Otsu: {limiar_otsu:.1f}"),
    ])

    cv2.imshow("Exercicio 3A - limiarizacao", painel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()