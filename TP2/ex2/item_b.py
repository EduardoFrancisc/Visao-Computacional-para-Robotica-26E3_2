import cv2
import os
import time
import numpy as np
from deepface import DeepFace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(SCRIPT_DIR, "known_faces")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise IOError(f"Erro ao carregar cascade: {cascade_path}")

MODEL_NAME = "VGG-Face"
DISTANCE_METRIC = "cosine"
THRESHOLD = 0.40


def load_known_faces():
    """Carrega fotos de known_faces/ e computa embeddings por identidade."""
    known = []
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

    for filename in os.listdir(KNOWN_FACES_DIR):
        if not filename.lower().endswith(valid_ext):
            continue
        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        name = os.path.splitext(filename)[0]
        try:
            result = DeepFace.represent(
                img_path=filepath, model_name=MODEL_NAME,
                detector_backend="skip", enforce_detection=False,
            )
            if result:
                embedding = np.array(result[0]["embedding"])
                known.append({"name": name, "embedding": embedding})
                print(f"  [OK] {name} ({filename})")
        except Exception as e:
            print(f"  [ERRO] {filename}: {e}")
    return known


def get_embedding_from_roi(roi_bgr):
    """Computa embedding de uma ROI facial já recortada."""
    try:
        result = DeepFace.represent(
            img_path=roi_bgr, model_name=MODEL_NAME,
            detector_backend="skip", enforce_detection=False,
        )
        if result:
            return np.array(result[0]["embedding"])
    except Exception:
        pass
    return None


def find_best_match(embedding, known_faces):
    """Compara embedding com identidades cadastradas via distância cosseno."""
    best_name, best_dist = "Desconhecido", float("inf")
    for person in known_faces:
        dist = 1 - np.dot(embedding, person["embedding"]) / (
            np.linalg.norm(embedding) * np.linalg.norm(person["embedding"])
        )
        if dist < best_dist:
            best_dist = dist
            best_name = person["name"]
    if best_dist > THRESHOLD:
        best_name = "Desconhecido"
    return best_name, best_dist


def main():
    print("=" * 60)
    print("  RECONHECIMENTO FACIAL — DeepFace + VGG-Face")
    print("=" * 60)

    print(f"\nCarregando modelo {MODEL_NAME}...")
    known_faces = load_known_faces()

    if len(known_faces) < 3:
        print(f"\n[AVISO] {len(known_faces)} identidade(s). Adicione fotos em: {KNOWN_FACES_DIR}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: não foi possível abrir a câmera.")
        return

    inference_times = []
    frame_count = 0
    RECOGNITION_INTERVAL = 5
    cached_results = []

    print("\nStream iniciado. [Q/ESC] para sair.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60)
        )

        if frame_count % RECOGNITION_INTERVAL == 0 and len(faces) > 0:
            t_start = time.time()
            results = []
            for (x, y, w, h) in faces:
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:
                    results.append(("Desconhecido", 1.0))
                    continue
                roi_resized = cv2.resize(roi, (224, 224))
                embedding = get_embedding_from_roi(roi_resized)
                if embedding is not None and len(known_faces) > 0:
                    results.append(find_best_match(embedding, known_faces))
                else:
                    results.append(("Desconhecido", 1.0))
            inference_times.append((time.time() - t_start) * 1000)
            cached_results = results

        for i, (x, y, w, h) in enumerate(faces):
            name, dist = cached_results[i] if i < len(cached_results) else ("Desconhecido", 1.0)
            color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{name} ({dist:.2f})"
            lbl_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - lbl_sz[1] - 10), (x + lbl_sz[0], y), color, -1)
            cv2.putText(frame, label, (x, y - 5),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # HUD
        h_f, w_f = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_f, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        avg_ms = np.mean(inference_times) if inference_times else 0
        cv2.putText(frame,
                     f"Identidades: {len(known_faces)} | Faces: {len(faces)} | Latencia: {avg_ms:.1f} ms",
                     (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Modelo: {MODEL_NAME} | Limiar: {THRESHOLD}",
                     (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Reconhecimento Facial - DeepFace (TP2 Ex2 Item B)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Relatório final
    print("\n" + "=" * 60)
    print("  RELATÓRIO — RECONHECIMENTO FACIAL")
    print("=" * 60)
    print(f"  Modelo:              {MODEL_NAME}")
    print(f"  Métrica:             {DISTANCE_METRIC}")
    print(f"  Limiar:              {THRESHOLD}")
    print(f"  Identidades:         {len(known_faces)}")
    print(f"  Frames processados:  {frame_count}")
    print(f"  Inferências:         {len(inference_times)}")
    if inference_times:
        print(f"  Latência média:      {np.mean(inference_times):.1f} ms")
        print(f"  Latência mín/máx:    {np.min(inference_times):.1f} / {np.max(inference_times):.1f} ms")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
