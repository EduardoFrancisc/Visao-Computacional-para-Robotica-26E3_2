import time
import cv2
from pathlib import Path
from ultralytics import YOLO
from vision_utils import draw_detections, IoUTracker

ROOT = Path(__file__).resolve().parent

video = ROOT / "dados" / "video_real.mp4"
modelo_path = ROOT / "modelos" / "yolov8n.pt"
saida = ROOT / "saidas" / "yolo_tracking.avi"

if not video.exists():
    raise FileNotFoundError(f"Vídeo não encontrado: {video}")

if not modelo_path.exists():
    raise FileNotFoundError(f"Modelo não encontrado: {modelo_path}")


cap = cv2.VideoCapture(str(video))
if not cap.isOpened():
    raise RuntimeError(f"Não foi possível abrir o vídeo: {video}")

largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 20.0

writer = cv2.VideoWriter(
    str(saida),
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (largura, altura),
)

model = YOLO(str(modelo_path))
tracker = IoUTracker(iou_thr=0.30, trail_len=30)
latencias = []
frames_processados = 0

linha_x = largura // 2
lado_anterior = {}
entradas = 0
saidas = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break  # chegou ao fim de dados/video_real.mp4

    t0 = time.perf_counter()

    # O YOLO já aplica NMS internamente com o parâmetro iou.
    result = model(frame, conf=0.25, iou=0.40, verbose=False)[0]

    dets = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        cls_id = int(box.cls[0])

        dets.append(
            {
                "cls": model.names[cls_id],
                "conf": float(box.conf[0]),
                "bbox": [x1, y1, x2, y2],
            }
        )

    tracked = tracker.update(dets)
    for t in tracked:
        x1, y1, x2, y2 = t["bbox"]
        cx = int((x1 + x2) / 2)
        tid = t["id"]

        lado_atual = "esq" if cx < linha_x else "dir"

        if tid in lado_anterior and lado_anterior[tid] != lado_atual:
            if lado_atual == "dir":
                entradas += 1  # cruzou esquerda → direita
            else:
                saidas += 1  # cruzou direita → esquerda

        lado_anterior[tid] = lado_atual

    out = draw_detections(
        frame,
        tracked,
        show_id=True,
        tracks=tracker.trails,
    )

    cv2.line(out, (linha_x, 0), (linha_x, altura), (0, 255, 255), 2)

    cv2.putText(
        out,
        f"Entradas: {entradas} | Saidas: {saidas}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )

    latencias.append((time.perf_counter() - t0) * 1000)
    writer.write(out)
    frames_processados += 1

cap.release()
writer.release()

duracao_segundos = frames_processados / fps
duracao_minutos = duracao_segundos / 60

ids_por_minuto = tracker.total_created / duracao_minutos

print("Vídeo final salvo em:", saida)

print("\nTotal de IDs criados:", tracker.total_created)
print(f"Duração do vídeo: {duracao_segundos:.2f} s")
print(f"Taxa de IDs criados: {ids_por_minuto:.2f} IDs/min")

print("\nEntradas cumulativas:", entradas)
print("Saídas cumulativas:", saidas)