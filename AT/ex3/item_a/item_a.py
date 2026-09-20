import os
import urllib.request
import tarfile
from pathlib import Path
import cv2
import time
import numpy as np

# Importando do vision_utils (conforme instruído, códigos base 01 a 35)
from vision_utils import ensure_dirs, create_synthetic_video, apply_nms, COLORS

try:
    from ultralytics import YOLO
except ImportError:
    print("Instalando ultralytics...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def setup_models_and_data():
    """Cria pastas (dados, modelos, saidas, etc.) e prepara os artefatos necessários."""
    # 1. Cria ambiente de pastas conforme 01_ambiente_e_pastas.py
    ensure_dirs(ROOT)
    
    # 2. Prepara video sintético para simular o stream de câmera/vídeo
    video_path = ROOT / "dados" / "video_teste.avi"
    if not video_path.exists():
        print("Criando vídeo sintético para teste...")
        create_synthetic_video(video_path, n_frames=120)
        
    # 3. Prepara modelo SSD MobileNetV2 (OpenCV DNN)
    modelo_dir = ROOT / "modelos"
    pb_path = modelo_dir / "frozen_inference_graph.pb"
    pbtxt_path = modelo_dir / "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
    
    if not pb_path.exists() or not pbtxt_path.exists():
        print("Baixando SSD MobileNetV2 (pesos e configuração)...")
        url_tar = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
        tar_path = modelo_dir / "ssd_mobilenet.tar.gz"
        urllib.request.urlretrieve(url_tar, tar_path)
        
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith("frozen_inference_graph.pb"):
                    member.name = "frozen_inference_graph.pb"
                    tar.extract(member, path=modelo_dir)
                    break
        os.remove(tar_path)
        
        url_pbtxt = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
        urllib.request.urlretrieve(url_pbtxt, pbtxt_path)
        
    return video_path, pb_path, pbtxt_path


def get_file_size_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0


def formatar_caixa(frame, bbox, classe, conf, cor):
    """Desenha bounding box, rótulo e confiança no frame."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
    label = f"{classe} {conf:.2f}"
    cv2.putText(frame, label, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2)


def main():
    print("="*60)
    print("Iniciando Pipeline de Comparação: YOLOv8n vs SSD MobileNetV2")
    print("="*60)
    
    video_path, pb_path, pbtxt_path = setup_models_and_data()
    
    print("\nCarregando modelos na memória...")
    
    # Modelo 1: YOLOv8n via Ultralytics
    model_yolo = YOLO("yolov8n.pt") # Será baixado se não existir na raiz
    size_yolo_mb = get_file_size_mb("yolov8n.pt")
    params_yolo = "3.2M"  # Aproximação arquitetura YOLOv8n
    
    # Modelo 2: SSD MobileNetV2 via OpenCV DNN
    net_ssd = cv2.dnn.readNetFromTensorflow(str(pb_path), str(pbtxt_path))
    size_ssd_mb = get_file_size_mb(pb_path)
    params_ssd = "4.3M"  # Aproximação arquitetura MobileNetV2 (backbone + head)
    
    # Dicionário reduzido do COCO (índices típicos do MobileNetV2)
    coco_classes = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
    
    resultados = []
    
    for nome_modelo in ["YOLOv8n", "SSD MobileNetV2"]:
        print(f"\n[ Processando vídeo com {nome_modelo} ]")
        cap = cv2.VideoCapture(str(video_path))
        
        # Configurar VideoWriter para salvar a saída comprobatória
        saida_video_path = ROOT / "saidas" / f"resultado_{nome_modelo.replace(' ', '_')}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fps_video_original = cap.get(cv2.CAP_PROP_FPS) or 20.0
        largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(saida_video_path), fourcc, fps_video_original, (largura, altura))
        
        latencias = []
        frames_processados = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            t0 = time.perf_counter()
            detections_para_nms = []
            
            if nome_modelo == "YOLOv8n":
                # YOLOv8 aplica NMS internamente através do parâmetro iou
                results = model_yolo(frame, conf=0.25, iou=0.40, verbose=False)
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        classe = model_yolo.names[cls_id]
                        formatar_caixa(frame, (x1, y1, x2, y2), classe, conf, COLORS.get(classe, (0, 255, 120)))
                        
            else:
                # SSD MobileNetV2 via OpenCV DNN
                blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
                net_ssd.setInput(blob)
                out = net_ssd.forward()
                
                h, w = frame.shape[:2]
                # Coletar predições válidas
                for i in range(out.shape[2]):
                    conf = float(out[0, 0, i, 2])
                    if conf > 0.25:
                        class_id = int(out[0, 0, i, 1])
                        classe = coco_classes.get(class_id, f"obj_{class_id}")
                        
                        x1 = int(out[0, 0, i, 3] * w)
                        y1 = int(out[0, 0, i, 4] * h)
                        x2 = int(out[0, 0, i, 5] * w)
                        y2 = int(out[0, 0, i, 6] * h)
                        
                        detections_para_nms.append({
                            "bbox": [x1, y1, x2, y2],
                            "conf": conf,
                            "cls": classe
                        })
                
                # Aplicar NMS customizado com threshold 0.4 (conforme vision_utils.py)
                det_finais = apply_nms(detections_para_nms, score_thr=0.25, nms_thr=0.40)
                
                # Desenhar apenas as caixas que sobreviveram ao NMS
                for d in det_finais:
                    formatar_caixa(frame, d["bbox"], d["cls"], d["conf"], COLORS.get(d["cls"], (255, 100, 0)))
                        
            # Salvar o frame anotado no vídeo de saída
            writer.write(frame)
            
            # (Opcional) Descomente as duas linhas abaixo para ver em tempo real na tela
            cv2.imshow(f"Real-Time - {nome_modelo}", frame)
            if cv2.waitKey(1) == ord('q'): break
            
            # Registro de latência
            tempo_frame = (time.perf_counter() - t0) * 1000
            latencias.append(tempo_frame)
            frames_processados += 1
                
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"-> Vídeo salvo em: {saida_video_path}")
        
        lat_media = sum(latencias) / len(latencias)
        fps_medio = 1000 / lat_media
        
        if nome_modelo == "YOLOv8n":
            resultados.append({"modelo": nome_modelo, "fps": fps_medio, "latencia": lat_media, "params": params_yolo, "size": size_yolo_mb})
        else:
            resultados.append({"modelo": nome_modelo, "fps": fps_medio, "latencia": lat_media, "params": params_ssd, "size": size_ssd_mb})
            
    # TABELA COMPARATIVA
    print("\n" + "="*80)
    print("TABELA DE COMPARAÇÃO DOS MODELOS:")
    print(f"{'Modelo':<20} | {'FPS Médio':<12} | {'Latência (ms)':<15} | {'Parâmetros':<12} | {'Tam. Disco (MB)':<15}")
    print("-" * 80)
    for r in resultados:
        print(f"{r['modelo']:<20} | {r['fps']:<12.2f} | {r['latencia']:<15.2f} | {r['params']:<12} | {r['size']:<15.2f}")
    print("="*80)
    
    # CONCLUSÃO
    melhor_fps = max(resultados, key=lambda x: x["fps"])
    menor_tamanho = min(resultados, key=lambda x: x["size"])
    
    print("\nCONCLUSÃO E JUSTIFICATIVA PARA ROBÓTICA EMBARCADA:")
    print(f"-> Modelo com maior FPS e menor latência: {melhor_fps['modelo']} ({melhor_fps['fps']:.2f} FPS).")
    print(f"-> Modelo mais leve em disco: {menor_tamanho['modelo']} ({menor_tamanho['size']:.2f} MB).")
    
    print("\nJustificativa Técnica:")
    if melhor_fps['modelo'] == "YOLOv8n":
        print("- O YOLOv8n demonstrou maior velocidade (FPS) e menor latência por frame nesta execução.")
        print("- Sua arquitetura baseada em PyTorch e implementações otimizadas (Ultralytics) garantem alta eficiência em processamento de vídeo.")
        print("- Possui menos peso em disco comparado ao Grafo Congelado (Frozen Graph) do SSD e usa eficientemente a CPU/GPU.")
        print("- Para robótica embarcada (ex: Jetson Nano, Raspberry Pi), onde tempo de resposta é crítico para evitar obstáculos (latência baixa), o YOLOv8n é a escolha mais adequada dos dois testados.")
    else:
        print("- O SSD MobileNetV2 demonstrou maior velocidade (FPS) nesta execução, provavelmente devido à otimização do OpenCV DNN para CPUs x86/ARM.")
        print("- O backbone MobileNetV2 usa Depthwise Separable Convolutions, desenhadas especificamente para minimizar operações computacionais em dispositivos móveis/embarcados.")
        print("- Portanto, para um robô equipado apenas com uma CPU de baixo custo (Raspberry Pi, microcontroladores com Linux), este modelo é o ideal para garantir fluidez (alto FPS).")

    print("\nFim do pipeline.")

if __name__ == "__main__":
    main()
