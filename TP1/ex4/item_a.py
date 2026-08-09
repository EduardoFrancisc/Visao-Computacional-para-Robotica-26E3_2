import cv2
import numpy as np

def preparar_mascara(mascara: np.ndarray) -> np.ndarray:
    """Remove pequenos ruídos e fecha falhas da máscara binária."""
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Abertura para remover ruídos isolados
    limpa = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Fechamento para preencher buracos no objeto detectado
    limpa = cv2.morphologyEx(limpa, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return limpa

def main() -> None:
    # Abre a câmera principal do computador
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Não foi possível abrir a câmera.")
        return
        
    print("Iniciando rastreamento de cor. Pressione Q para encerrar.")

    # Range de cor HSV (Exemplo configurado para objetos AZUIS)
    # H varia de 0 a 179, S e V de 0 a 255
    limite_inferior = np.array([100, 150, 50])
    limite_superior = np.array([140, 255, 255])

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            altura, largura = frame.shape[:2]
            centro_x_frame = largura // 2
            centro_y_frame = altura // 2
            
            # Converte a imagem BGR para HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 1) Segmentar o objeto de cor definida via range HSV
            mascara_cor = cv2.inRange(hsv, limite_inferior, limite_superior)
            mascara_limpa = preparar_mascara(mascara_cor)
            
            # Localiza os contornos externos na máscara
            contornos, _ = cv2.findContours(
                mascara_limpa, 
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # 2) Encontrar o maior blob e calcular seu centroide
            if contornos:
                # Prioriza o contorno com maior área
                maior_contorno = max(contornos, key=cv2.contourArea)
                area = cv2.contourArea(maior_contorno)
                
                # Aplica um filtro de área mínima (ex: 500 px) para ignorar falsos positivos
                if area > 500:
                    # Calcula os momentos espaciais do contorno para achar o centro
                    momentos = cv2.moments(maior_contorno)
                    if momentos["m00"] != 0:
                        centroide_x = int(momentos["m10"] / momentos["m00"])
                        centroide_y = int(momentos["m01"] / momentos["m00"])
                        
                        # 3) Desenhar bounding box e centroide
                        x, y, w, h = cv2.boundingRect(maior_contorno)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.circle(frame, (centroide_x, centroide_y), 5, (0, 0, 255), -1)
                        
                        # 4) Imprimir posição normalizada no terminal
                        # Normaliza a distância do centroide ao centro entre -1.0 (esq) e 1.0 (dir)
                        desvio_x = (centroide_x - centro_x_frame) / centro_x_frame
                        
                        if desvio_x < -0.15:
                            direcao = "esquerda"
                        elif desvio_x > 0.15:
                            direcao = "direita"
                        else:
                            direcao = "centro"
                            
                        print(f"Objeto à {direcao}, desvio X = {desvio_x:.2f}")
                        
                        # 5) Simular correção de robô desenhando uma seta no frame
                        # A seta sai do centro geográfico da imagem e aponta para o objeto
                        cv2.arrowedLine(
                            frame, 
                            (centro_x_frame, centro_y_frame), 
                            (centroide_x, centro_y_frame), 
                            (255, 0, 0), 
                            3, 
                            tipLength=0.2
                        )
                        
            # Adiciona linha guia central para facilitar a visualização do desvio nulo
            cv2.line(frame, (centro_x_frame, 0), (centro_x_frame, altura), (255, 255, 255), 1)

            # Exibe o feed em tempo real com as sobreposições
            cv2.imshow("Rastreador de Cor e Controle de Trajetoria", frame)
            
            # Aguarda tecla para encerrar
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), ord("Q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()