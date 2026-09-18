import os, argparse, time, cv2, numpy as np

p = argparse.ArgumentParser(description='Detecta pedestres usando HOG em um vídeo')
p.add_argument('--video', default='video_exemplo.mp4', help='Caminho para o vídeo de entrada')
a = p.parse_args()

video_path = os.path.join(os.path.dirname(__file__), a.video)

# Configura o detector HOG pré‑treinado do OpenCV
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Definição dos dois cenários de parâmetros
cenarios = {
    'rapido': dict(winStride=(12, 12), padding=(8, 8), scale=1.10),
    'preciso': dict(winStride=(4, 4),   padding=(8, 8), scale=1.03)
}

def processar_cenario(nome):
    """Processa o vídeo usando o cenário indicado.

    Retorna:
        (nome, frames, det_por_frame, tempo_medio_ms, fps)
    """
    cap = cv2.VideoCapture(video_path)
    tempos = []                # tempo de inferência por frame (ms)
    total_deteccoes = 0
    frames = 0
    cfg = cenarios[nome]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        rects, _ = hog.detectMultiScale(
            frame,
            winStride=cfg['winStride'],
            padding=cfg['padding'],
            scale=cfg['scale']
        )
        dur = (time.perf_counter() - t0) * 1000  # ms
        tempos.append(dur)
        total_deteccoes += len(rects)
        frames += 1

        # Desenhar bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Informar no console
        print(f'[{nome}] frame={frames:04d} det={len(rects):2d} inferencia={dur:7.2f} ms')

        # Mostrar o frame (opcional – pode ser comentado em ambientes sem tela)
        cv2.imshow('HOG People Detector', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
            break

    cap.release()
    cv2.destroyAllWindows()

    media_ms = float(np.mean(tempos)) if tempos else 0.0
    fps = 1000.0 / media_ms if media_ms else 0.0
    deteccoes_por_frame = total_deteccoes / frames if frames else 0.0
    return nome, frames, deteccoes_por_frame, media_ms, fps

# Executa ambos os cenários
resultados = [processar_cenario('rapido'), processar_cenario('preciso')]

# Tabela de resumo
print('\n' + '=' * 72)
header = f"{'Cenario':<12}{'Frames':>10}{'Det/frame':>14}{'ms/frame':>14}{'FPS':>12}"
print(header)
print('-' * 72)
for nome, frm, det_fp, ms_fp, fps in resultados:
    print(f'{nome:<12}{frm:>10d}{det_fp:>14.2f}{ms_fp:>14.2f}{fps:>12.2f}')
print('=' * 72)
