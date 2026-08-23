import cv2
import numpy as np

def main():
    # Inicializa a captura de vídeo da webcam padrão (índice 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a câmera/vídeo.")
        return

    # (1) Definir o range de cor HSV para segmentação 
    # Exemplo configurado para a cor AZUL.
    lower_color = np.array([100, 150, 50])
    upper_color = np.array([140, 255, 255])

    # Elemento estruturante (kernel) para as operações morfológicas
    kernel = np.ones((5, 5), np.uint8)

    print("Iniciando a pipeline de segmentação. Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo ou falha ao capturar o frame.")
            break
            
        frame_height, frame_width = frame.shape[:2]
        total_area = frame_width * frame_height

        # (1) Segmentar o objeto de interesse por range HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # (2) Aplicar operações morfológicas (erode + dilate) para limpar a máscara
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Encontrar os contornos na máscara limpa
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Cria uma cópia do frame para desenhar a máscara colorida semitransparente
        overlay = frame.copy()

        if contours:
            # Pegar o maior contorno (assume-se que seja o objeto principal de interesse)
            largest_contour = max(contours, key=cv2.contourArea)
            area_contour = cv2.contourArea(largest_contour)
            
            # Filtro para ignorar ruídos muito pequenos
            if area_contour > 500:
                # (3) Extrair a ROI (Bounding Box)
                x, y, w, h = cv2.boundingRect(largest_contour)
                roi_area = w * h
                
                # Destacar a ROI com um retângulo ao redor
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Preencher a máscara colorida apenas sobre o objeto (ex: na cor verde)
                cv2.drawContours(overlay, [largest_contour], -1, (0, 255, 0), -1)
                
                # (4) Calcular e imprimir a proporção da área da ROI em relação ao frame total
                proportion = (roi_area / total_area) * 100
                print(f"Proporção da ROI na tela: {proportion:.2f}% (Tamanho da Bounding Box: {w}x{h})")

        # Aplica a semitransparência na máscara colorida
        alpha = 0.4 # Transparência do overlay (40%)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Exibir os resultados em duas janelas
        cv2.imshow('Pipeline de Segmentacao (Resultado)', frame)
        cv2.imshow('Mascara HSV Limpa', mask)

        # Para interromper o loop pressionando a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera a captura de vídeo e fecha todas as janelas do OpenCV
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
