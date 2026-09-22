from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO

from vision_utils import (
    CLASSES,
    ensure_dirs,
    Timer,
    apply_nms,
    draw_detections,
    write_csv,
)

ROOT = Path(__file__).resolve().parent
ensure_dirs(ROOT)

VIDEO = ROOT / "dados" / "video_real.mp4"

YOLO_PATH = ROOT / "modelos" / "yolov8n.pt"
SSD_PB = ROOT / "modelos" / "frozen_inference_graph.pb"
SSD_PBTXT = ROOT / "modelos" / "ssd_mobilenet_v2_coco.pbtxt"

CONF_THR = 0.25
NMS_THR = 0.40

SSD_CLASSES = {
    1: CLASSES[0],  # person
    2: CLASSES[2],  # bicycle
    3: CLASSES[1],  # car
    6: CLASSES[3],  # bus
}

faltando = [
    arquivo for arquivo in [YOLO_PATH, SSD_PB, SSD_PBTXT] if not arquivo.exists()
]

if faltando:
    print("Modelos ausentes na pasta modelos/:")
    for arquivo in faltando:
        print("-", arquivo.name)
    raise SystemExit

modelo_yolo = YOLO(str(YOLO_PATH))

rede_ssd = cv2.dnn.readNetFromTensorflow(str(SSD_PB), str(SSD_PBTXT))

resultados = []

for nome_modelo in ["YOLOv8n", "SSD MobileNetV2"]:
    cap = cv2.VideoCapture(str(VIDEO))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 20.0
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    nome_saida = nome_modelo.lower().replace(" ", "_")
    video_saida = ROOT / "saidas" / f"{nome_saida}_anotado.avi"

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_saida), fourcc, fps_video, (largura, altura))

    latencias = []
    frames_processados = 0

    print(f"\nProcessando vídeo com {nome_modelo}...")

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        timer = Timer()
        detections = []

        if nome_modelo == "YOLOv8n":
            result = modelo_yolo(frame, conf=CONF_THR, iou=NMS_THR, verbose=False)[0]
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                classe_id = int(box.cls[0])

                detections.append(
                    {
                        "cls": modelo_yolo.names[classe_id],
                        "conf": float(box.conf[0]),
                        "bbox": [x1, y1, x2, y2],
                    }
                )
        else:
            blob = cv2.dnn.blobFromImage(
                frame, size=(300, 300), swapRB=True, crop=False
            )

            rede_ssd.setInput(blob)
            saidas = rede_ssd.forward()

            for det in saidas[0, 0]:
                confianca = float(det[2])
                classe_id = int(det[1])

                if confianca < CONF_THR:
                    continue

                if classe_id not in SSD_CLASSES:
                    continue

                x1 = int(det[3] * largura)
                y1 = int(det[4] * altura)
                x2 = int(det[5] * largura)
                y2 = int(det[6] * altura)

                detections.append(
                    {
                        "cls": SSD_CLASSES[classe_id],
                        "conf": confianca,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        detections = apply_nms(detections, score_thr=CONF_THR, nms_thr=NMS_THR)

        frame_saida = draw_detections(frame, detections)

        latencia_ms = timer.ms()
        latencias.append(latencia_ms)
        frames_processados += 1

        fps_atual = 1000 / latencia_ms if latencia_ms > 0 else 0

        cv2.putText(
            frame_saida,
            f"{nome_modelo} | FPS: {fps_atual:.1f} | {latencia_ms:.1f} ms",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

        writer.write(frame_saida)

        cv2.imshow(nome_modelo, frame_saida)

        # Para uma comparação justa, tem que deixar o vídeo terminar.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if not latencias:
        print("Nenhum frame foi processado.")
        continue

    latencia_media = sum(latencias) / len(latencias)
    fps_medio = 1000 / latencia_media

    if nome_modelo == "YOLOv8n":
        tamanho_mb = YOLO_PATH.stat().st_size / (1024 * 1024)
        parametros_m = 3.2
    else:
        tamanho_mb = SSD_PB.stat().st_size / (1024 * 1024)
        parametros_m = 4.3

    resultados.append(
        {
            "modelo": nome_modelo,
            "fps": round(fps_medio, 2),
            "latencia_ms": round(latencia_media, 2),
            "parametros_M": parametros_m,
            "tamanho_MB": round(tamanho_mb, 2),
            "frames": frames_processados,
        }
    )

    print(f"Vídeo anotado salvo em: {video_saida}")
    print(f"Latência média: {latencia_media:.2f} ms")
    print(f"FPS médio: {fps_medio:.2f}")

df = pd.DataFrame(resultados)

print("\nTABELA COMPARATIVA")
print(df.to_string(index=False))

csv_saida = ROOT / "relatorios" / "metricas_yolov8n_ssd.csv"

write_csv(
    csv_saida,
    resultados,
    ["modelo", "fps", "latencia_ms", "parametros_M", "tamanho_MB", "frames"],
)

print("\nCSV salvo em:", csv_saida)

if len(resultados) == 2:
    melhor_fps = max(resultados, key=lambda x: x["fps"])
    menor_latencia = min(resultados, key=lambda x: x["latencia_ms"])
    menor_tamanho = min(resultados, key=lambda x: x["tamanho_MB"])

    print("\nCONCLUSÃO")
    print("Maior FPS:", melhor_fps["modelo"])
    print("Menor latência:", menor_latencia["modelo"])
    print("Menor tamanho em disco:", menor_tamanho["modelo"])
