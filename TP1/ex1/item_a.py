import cv2
from pathlib import Path

def main() -> None:
    
    indice_camera = 0
    cap = cv2.VideoCapture(indice_camera)
    fonte_video = "Câmera"
    caminho_saida = Path(__file__).resolve().parent / "item_b_frame.png"
    
    if not cap.isOpened():
        print("Não foi possível abrir a câmera nem o arquivo 'video.mp4'.")
        print("Verifique se o arquivo existe ou cheque suas permissões.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    numero_frame = 0
    tick_anterior = cv2.getTickCount()
    frequencia_ticks = cv2.getTickFrequency()
    
    print(f"{fonte_video} aberta(o) corretamente.")
    print("Pressione S na janela para salvar o frame atual.")
    print("Pressione Q na janela para encerrar.")
    print("-" * 50)
    print("Resolução atual | número do frame | FPS estimado")
    
    try:
        while True:
            
            ret, frame = cap.read()
        
            if not ret:
                print("Falha na leitura do frame ou fim do vídeo.")
                break
            
            numero_frame += 1
            tick_atual = cv2.getTickCount()
            
            tempo_frame = (
                            tick_atual - tick_anterior
                        ) / frequencia_ticks
            
            tick_anterior = tick_atual
            
            if tempo_frame > 0:
                fps_estimado = 1.0 / tempo_frame
            else:
                fps_estimado = 0.0
            
            altura, largura = frame.shape[:2]
            
            print(
                f"Resolução: {largura}x{altura} | "
                f"Frame: {numero_frame:06d} | "
                f"FPS estimado: {fps_estimado:7.2f}"
            )
            
            texto = (
                f"{largura}x{altura} | "
                f"Frame {numero_frame} | "
                f"FPS {fps_estimado:.1f}"
            )
            
            cv2.putText(
                frame,                       # Imagem
                texto,                       # Texto
                (10, 30),                    # Posição inicial
                cv2.FONT_HERSHEY_SIMPLEX,    # Fonte
                0.65,                        # Tamanho da fonte
                (0, 255, 0),                 # Cor BGR: verde
                2,                           # Espessura
                cv2.LINE_AA,                 # Suavização das letras
            )
            
            cv2.imshow(
                f"{fonte_video} - pressione Q para sair, S para salvar",
                frame,
            )
            
            tecla = cv2.waitKey(1) & 0xFF
            
            if tecla == ord("s") or tecla == ord("S"):
                cv2.imwrite(caminho_saida, frame)
                print(f"*** Imagem salva com sucesso: {caminho_saida} ***")
            
            elif tecla == ord("q") or tecla == ord("Q"):
                print("Encerramento solicitado pelo usuário.")
                break
            
    finally:
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"{fonte_video} liberada(o).")
        print("Janelas fechadas corretamente.")

if __name__ == "__main__":
    main()