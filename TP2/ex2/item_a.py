"""
TP2 - Exercício 2 - Item A
Detector facial em tempo real usando Haar Cascade.

Teclas: [1/2] trocar config | [S] salvar ROIs | [F] +1 FP | [R] reset | [Q/ESC] sair
"""

import cv2
import os
import time
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces_48x48")
os.makedirs(OUTPUT_DIR, exist_ok=True)

cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise IOError(f"Erro ao carregar cascade: {cascade_path}")

# ======================== TRADE-OFF: SENSIBILIDADE vs. ESPECIFICIDADE ========================
#
# Config 1 (scaleFactor=1.1, minNeighbors=3) — Alta Sensibilidade
#   Reduz a imagem em 10% por escala → mais janelas → detecta mais rostos (alto recall),
#   porém aceita detecções com poucas confirmações → mais falsos positivos.
#
# Config 2 (scaleFactor=1.3, minNeighbors=6) — Alta Especificidade
#   Reduz a imagem em 30% por escala → menos janelas → pode perder rostos intermediários,
#   mas exige muitas confirmações → praticamente elimina falsos positivos (alta precisão).
#
# Resumo: Config 1 prioriza não perder faces (sensibilidade); Config 2 prioriza não gerar
#         alarmes falsos (especificidade). A escolha depende do custo relativo de cada erro.
# =============================================================================================

configs = {
    1: {
        "name": "Config 1 — Alta Sensibilidade",
        "scaleFactor": 1.1,
        "minNeighbors": 3,
        "color": (0, 255, 0),
        "total_detections": 0,
        "total_frames": 0,
        "false_positives": 0,
        "faces_saved": 0,
        "description": "scaleFactor=1.1, minNeighbors=3 (mais detecções, mais FP)",
    },
    2: {
        "name": "Config 2 — Alta Especificidade",
        "scaleFactor": 1.3,
        "minNeighbors": 6,
        "color": (255, 165, 0),
        "total_detections": 0,
        "total_frames": 0,
        "false_positives": 0,
        "faces_saved": 0,
        "description": "scaleFactor=1.3, minNeighbors=6 (menos detecções, menos FP)",
    },
}

current_config = 1
face_counter = 0
auto_save = True


def detect_faces(frame_gray, cfg):
    """Detecta rostos usando os parâmetros da configuração fornecida."""
    return face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=cfg["scaleFactor"],
        minNeighbors=cfg["minNeighbors"],
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )


def save_face_roi(frame, x, y, w, h, cfg):
    """Extrai ROI facial, redimensiona para 48x48 e salva em disco."""
    global face_counter
    roi = frame[y:y+h, x:x+w]
    roi_48 = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)

    face_counter += 1
    cfg["faces_saved"] += 1
    filename = f"face_{face_counter:04d}_cfg{current_config}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(filepath, roi_48)
    return filepath, roi_48


def draw_detections(frame, faces, cfg):
    """Desenha bounding boxes no frame."""
    color = cfg["color"]
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"Face #{i+1}", (x, y - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def draw_hud(frame, cfg, fps, num_faces):
    """Desenha HUD com informações da configuração atual."""
    h, w = frame.shape[:2]
    color = cfg["color"]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    texts = [
        f"{cfg['name']}  |  {cfg['description']}",
        f"Faces: {num_faces}  |  FPS: {fps:.1f}  |  Total: {cfg['total_detections']}",
        f"FP: {cfg['false_positives']}  |  Salvas: {cfg['faces_saved']}  |  "
        f"Taxa: {cfg['total_detections'] / max(cfg['total_frames'], 1):.2f} det/frame",
    ]
    for i, text in enumerate(texts):
        cv2.putText(frame, text, (10, 25 + i * 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)

    instructions = "[1/2] Config  |  [S] Salvar  |  [F] +1 FP  |  [R] Reset  |  [Q/ESC] Sair"
    cv2.putText(frame, instructions, (10, h - 15),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def print_report():
    """Imprime relatório final com resultados por configuração."""
    print("\n" + "=" * 75)
    print("  RELATÓRIO FINAL — DETECÇÃO FACIAL COM HAAR CASCADE")
    print("=" * 75)

    for key, cfg in configs.items():
        total_frames = max(cfg["total_frames"], 1)
        detection_rate = cfg["total_detections"] / total_frames

        print(f"\n{'─' * 75}")
        print(f"  {cfg['name']}")
        print(f"  scaleFactor={cfg['scaleFactor']}, minNeighbors={cfg['minNeighbors']}")
        print(f"{'─' * 75}")
        print(f"  Frames processados:         {cfg['total_frames']}")
        print(f"  Total de detecções:         {cfg['total_detections']}")
        print(f"  Taxa de detecção (det/frame): {detection_rate:.2f}")
        print(f"  Falsos positivos visíveis:  {cfg['false_positives']}")
        print(f"  Faces salvas em disco:      {cfg['faces_saved']}")

        # Trade-off resumido no terminal
        if key == 1:
            print("\n  TRADE-OFF: scaleFactor baixo + minNeighbors baixo →")
            print("    Alta sensibilidade (detecta quase tudo), mas mais falsos positivos.")
        else:
            print("\n  TRADE-OFF: scaleFactor alto + minNeighbors alto →")
            print("    Alta especificidade (poucos FPs), mas pode perder rostos reais.")

    print(f"\n{'=' * 75}")
    print(f"  Faces salvas em: {OUTPUT_DIR}")
    print(f"  Total global: {face_counter}")
    print(f"{'=' * 75}\n")


def main():
    global current_config, face_counter

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: não foi possível abrir a câmera.")
        return

    print("=" * 60)
    print("  DETECTOR FACIAL — Haar Cascade")
    print("  [1/2] config | [F] +FP | [Q/ESC] sair")
    print("=" * 60)

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        cfg = configs[current_config]

        faces = detect_faces(gray, cfg)
        num_faces = len(faces)
        cfg["total_detections"] += num_faces
        cfg["total_frames"] += 1

        # Salva ROIs a cada 15 frames
        if auto_save and num_faces > 0 and cfg["total_frames"] % 15 == 0:
            for (x, y, w, h) in faces:
                save_face_roi(frame, x, y, w, h, cfg)

        frame_display = draw_detections(frame, faces, cfg)

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-6))
        prev_time = curr_time

        frame_display = draw_hud(frame_display, cfg, fps, num_faces)

        # Miniaturas 48x48 no canto
        if num_faces > 0:
            thumb_x = frame_display.shape[1] - 60
            thumb_y = 130
            for (x, y, w, h) in faces[:3]:
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    roi_48 = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
                    cv2.rectangle(frame_display,
                                  (thumb_x - 2, thumb_y - 2),
                                  (thumb_x + 50, thumb_y + 50), cfg["color"], 1)
                    frame_display[thumb_y:thumb_y+48, thumb_x:thumb_x+48] = roi_48
                    thumb_y += 58

        cv2.imshow("Detector Facial - Haar Cascade (TP2 Ex2 Item A)", frame_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('1'):
            current_config = 1
            print(f"\n>> {configs[1]['name']}")
        elif key == ord('2'):
            current_config = 2
            print(f"\n>> {configs[2]['name']}")
        elif key == ord('s'):
            if num_faces > 0:
                for (x, y, w, h) in faces:
                    filepath, _ = save_face_roi(frame, x, y, w, h, cfg)
                    print(f"   Salvo: {filepath}")
            else:
                print("   Nenhuma face detectada.")
        elif key == ord('f'):
            cfg["false_positives"] += 1
            print(f"   FP +1 ({cfg['name']}): total = {cfg['false_positives']}")
        elif key == ord('r'):
            for c in configs.values():
                c["total_detections"] = 0
                c["total_frames"] = 0
                c["false_positives"] = 0
            print("   Contadores resetados.")

    cap.release()
    cv2.destroyAllWindows()
    print_report()


if __name__ == "__main__":
    main()
