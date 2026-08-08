import cv2
import numpy as np
from pathlib import Path

def obter_imagem_base(nome_arquivo: str = "blur_image.jpg"):
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

def redimensionar_quadrado(imagem, lado: int = 320):
    """Redimensiona a imagem para facilitar a montagem de painéis didáticos."""
    return cv2.resize(imagem, (lado, lado), interpolation=cv2.INTER_AREA)

def variancia_laplaciano(imagem_bgr) -> float:

    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)

    return float(cv2.Laplacian(cinza, cv2.CV_64F).var())

def main() -> None:
    
    original = obter_imagem_base()
    
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1],
    ], dtype=np.float32)
    
    manual = cv2.filter2D(original, -1, kernel)
    blur = cv2.GaussianBlur(original, (0, 0), sigmaX=1.2)
    unsharp = cv2.addWeighted(original, 2.0, blur, -1.0, 0)
    
    metricas = {
        "Original": variancia_laplaciano(original),
        "Kernel manual": variancia_laplaciano(manual),
        "Unsharp masking": variancia_laplaciano(unsharp),
    }
    
    print("Variância do Laplaciano:")
    for nome, valor in metricas.items():
        print(f"- {nome:16s}: {valor:.2f}")
        
    metodo_mais_nitido = max(metricas, key=metricas.get)
    
    print()
    print(f"Maior valor de nitidez: {metodo_mais_nitido}")
    print(
        "Observação: valores altos também podem indicar ruído; "
        "a métrica deve ser interpretada junto à inspeção visual."
    )

    painel = cv2.hconcat([
        adicionar_titulo(redimensionar_quadrado(original, 360), "Original"),
        adicionar_titulo(redimensionar_quadrado(manual, 360), "Kernel manual"),
        adicionar_titulo(redimensionar_quadrado(unsharp, 360), "Unsharp masking"),

    ])

    cv2.imshow("Exercicio 2B - comparacao de nitidez", painel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()