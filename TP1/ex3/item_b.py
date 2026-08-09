import cv2
from pathlib import Path

def obter_imagem_base(nome_arquivo: str = "paisagem.jpg"):
    caminho = Path(__file__).resolve().parent / nome_arquivo
    
    imagem = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    altura, largura = imagem.shape[:2]
    
    if altura < 480 or largura < 480:
        escala = max(480 / altura, 480 / largura)
        nova_largura = int(round(largura * escala))
        nova_altura = int(round(altura * escala))
        
        return cv2.resize(
            imagem,
            (nova_largura, nova_altura),
            interpolation=cv2.INTER_CUBIC,
        )
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

def redimensionar_proporcional(img, largura_alvo=420):
    """Redimensiona mantendo a proporção para padronizar o painel."""
    proporcao = largura_alvo / img.shape[1]
    altura_alvo = int(img.shape[0] * proporcao)
    return cv2.resize(img, (largura_alvo, altura_alvo), interpolation=cv2.INTER_AREA)

def main() -> None:
    imagem = obter_imagem_base()
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    cinza_suave = cv2.GaussianBlur(cinza, (5, 5), 0)

    adaptativo = cv2.adaptiveThreshold(
        cinza_suave,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )

    canny_sensivel = cv2.Canny(adaptativo, 50, 150)
    canny_restritivo = cv2.Canny(adaptativo, 150, 250)

    contornos, _ = cv2.findContours(canny_sensivel, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    imagem_contornos = imagem.copy()
    
    total_validos = 0
    maior_area = 0.0

    for c in contornos:
        area = cv2.contourArea(c)
        
        if area > 200:
            total_validos += 1
            
            if area > maior_area:
                maior_area = area
                
            if area < 1000:
                cor = (255, 0, 0)      # Azul
            elif area < 5000:
                cor = (0, 255, 0)      # Verde
            else:
                cor = (0, 0, 255)      # Vermelho
                
            cv2.drawContours(imagem_contornos, [c], -1, cor, 2)

    print("-" * 40)
    print("ANÁLISE DE CONTORNOS")
    print("-" * 40)
    print(f"Total de contornos encontrados (> 200 px²): {total_validos}")
    print(f"Área do maior contorno: {maior_area:.2f} px²")
    print("-" * 40)

    # Preparação para o painel:
    # 1. Converter imagens de 1 canal (cinza) para 3 canais (BGR) para permitir concatenação
    canny_sensivel_bgr = cv2.cvtColor(canny_sensivel, cv2.COLOR_GRAY2BGR)
    canny_restritivo_bgr = cv2.cvtColor(canny_restritivo, cv2.COLOR_GRAY2BGR)

    # 2. Redimensionar para tamanho igual e adicionar título
    img_1 = adicionar_titulo(redimensionar_proporcional(canny_sensivel_bgr), "Canny (50, 150)")
    img_2 = adicionar_titulo(redimensionar_proporcional(canny_restritivo_bgr), "Canny (150, 250)")
    img_3 = adicionar_titulo(redimensionar_proporcional(imagem_contornos), "Contornos > 200px")

    # 3. Concatenar horizontalmente
    painel = cv2.hconcat([img_1, img_2, img_3])

    cv2.imshow("Painel Integrado - Bordas e Contornos", painel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()