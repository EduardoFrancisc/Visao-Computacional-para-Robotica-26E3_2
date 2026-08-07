
import cv2
from pathlib import Path


def main() -> None:
    caminho_saida = Path(__file__).resolve().parent / "item_b_frame.png"

    colorida = cv2.imread(
            str(caminho_saida),
            cv2.IMREAD_COLOR
            )

    if colorida is None:
        print(f"Não foi possível abrir a imagem: {caminho_saida}")
        return
    
    # Converte a imagem colorida de BGR para escala de cinza.
    cinza = cv2.cvtColor(
        colorida,
        cv2.COLOR_BGR2GRAY
    )
    
    # Exibe os metadados dos arrays NumPy.
    print()
    print("Metadados dos arrays NumPy:")
    print(
        f"Colorida -> shape: {colorida.shape}, "
        f"dtype: {colorida.dtype}"
    )
    print(
        f"Cinza    -> shape: {cinza.shape}, "
        f"dtype: {cinza.dtype}"
    )
    
        # A imagem colorida possui três canais: (altura, largura, 3).
    # A imagem cinza possui apenas dois eixos: (altura, largura).
    #
    # Para usar cv2.hconcat, as duas imagens precisam ter a mesma
    # quantidade de canais. Por isso, convertemos temporariamente
    # a imagem cinza para BGR apenas para montar o painel.
    cinza_bgr = cv2.cvtColor(
        cinza,
        cv2.COLOR_GRAY2BGR
    )
    
    # Junta horizontalmente a imagem colorida e a versão cinza.
    painel = cv2.hconcat(
        [colorida, cinza_bgr]
    )
    
    # Adiciona identificações ao painel.
    cv2.putText(
        painel,
        "Colorida",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    
    
    cv2.putText(
        painel,
        "Escala de cinza",
        (colorida.shape[1] + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    
    print("Pressione Q ou Esc para fechar o painel.")
    
    # Exibe as duas versões lado a lado.
    cv2.imshow(
        "Item B - colorida e cinza",
        painel
    )
    
    # Mantém o painel aberto até Q ou Esc ser pressionado.
    while True:
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), ord("Q"), 27):
            break

    cv2.destroyAllWindows()
    
    print("Janelas fechadas corretamente.")


if __name__ == "__main__":
    main()