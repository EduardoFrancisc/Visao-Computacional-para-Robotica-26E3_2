import argparse
import csv
import time
from collections import deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import open_capture, resize_keep_aspect, overlay_text


def parse_roi(values):
    if values is None:
        return None
    if len(values) != 4:
        raise ValueError("ROI deve ser x y w h")
    return tuple(map(int, values))


def initialize_camshift(frame, roi):
    x, y, w, h = roi
    track_window = (x, y, w, h)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi_hsv = hsv[y:y + h, x:x + w]
    mask = cv2.inRange(roi_hsv, np.array((0, 40, 30)), np.array((180, 255, 255)))
    hist = cv2.calcHist([roi_hsv], [0], mask, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return track_window, hist


def create_kalman(dt=1.0, q=0.03, r=8.0, p=1.0):
    """Cria um filtro de Kalman 2D com estado [x, y, vx, vy].

    Q = processNoiseCov:
        Representa a incerteza do modelo de movimento. Se Q aumenta, o filtro
        passa a confiar menos no modelo de velocidade constante e reage mais
        rapidamente a mudanças bruscas, porém a trajetória pode ficar menos suave.

    R = measurementNoiseCov:
        Representa a incerteza da medição. Se R aumenta, o filtro confia menos
        no centroide medido pelo CamShift e suaviza mais a saída.

    P = errorCovPost:
        Representa a incerteza inicial do estado estimado. Um P maior indica que
        o filtro começa pouco confiante na posição/velocidade inicial e aceita
        correções mais fortes no início.
    """
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)
# Q (processNoiseCov): representa a incerteza no modelo de movimento (processo).
#   - Aumentar Q faz o filtro confiar menos no modelo de velocidade constante,
#     reagindo mais rapidamente a mudanças bruscas, porém pode tornar a trajetória
#     menos suave e introduzir mais ruído nas predições.
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * q
# R (measurementNoiseCov): representa a incerteza das medições (CamShift).
#   - Valores maiores de R fazem o filtro confiar menos nas medições e suavizar mais a saída.
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r
# P (errorCovPost): incerteza inicial do estado estimado.
#   - Um P maior indica baixa confiança inicial, permitindo correções mais fortes nas primeiras iterações.
    kf.errorCovPost = np.eye(4, dtype=np.float32) * p
    return kf


def main():
    parser = argparse.ArgumentParser(description="CamShift + Filtro de Kalman.")
    parser.add_argument("--source", default="ex2\data\sample_video.mp4")
    parser.add_argument("--roi", nargs="*", type=int, help="ROI opcional: x y w h")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--q", type=float, default=0.03, help="Ruído de processo Q")
    parser.add_argument("--r", type=float, default=8.0, help="Ruído de medição R")
    parser.add_argument("--output", default="output/trajectory_camshift_kalman.csv")
    args = parser.parse_args()

    cap = open_capture(args.source)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Fonte sem frames.")
    frame = resize_keep_aspect(frame, args.width)

    roi = parse_roi(args.roi)
    if roi is None:
        roi = cv2.selectROI("Selecione a ROI do objeto", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Selecione a ROI do objeto")

    track_window, hist = initialize_camshift(frame, roi)
    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    kf = create_kalman(q=args.q, r=args.r)

    # Inicializa o estado com o centro da ROI.
    x, y, w, h = roi
    init_cx = x + w / 2
    init_cy = y + h / 2
    kf.statePost = np.array([[init_cx], [init_cy], [0], [0]], dtype=np.float32)

    measured_points = []
    predicted_points = []
    times = deque(maxlen=120)
    frame_idx = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = output_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "measured_x", "measured_y", "kalman_x", "kalman_y", "loop_ms"])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_keep_aspect(frame, args.width)
        frame_idx += 1

        tic = time.perf_counter()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        backproj = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
        rotated_rect, track_window = cv2.CamShift(backproj, track_window, term_criteria)

        measured_x, measured_y = rotated_rect[0]
        measurement = np.array([[np.float32(measured_x)], [np.float32(measured_y)]])

        prediction = kf.predict()
        predicted_x, predicted_y = float(prediction[0][0]), float(prediction[1][0])
        kf.correct(measurement)

        elapsed_ms = (time.perf_counter() - tic) * 1000
        times.append(elapsed_ms)
        avg_ms = sum(times) / len(times)

        measured_points.append((measured_x, measured_y))
        predicted_points.append((predicted_x, predicted_y))
        writer.writerow([frame_idx, measured_x, measured_y, predicted_x, predicted_y, elapsed_ms])

        out = frame.copy()
        cv2.ellipse(out, rotated_rect, (0, 255, 0), 2)
        cv2.circle(out, (int(measured_x), int(measured_y)), 7, (0, 0, 255), -1)   # vermelho: medição CamShift
        cv2.circle(out, (int(predicted_x), int(predicted_y)), 7, (255, 0, 0), -1) # azul: predição Kalman

        out = overlay_text(out, [
            "CamShift + Kalman",
            f"medido/vermelho: ({measured_x:.1f}, {measured_y:.1f})",
            f"predito/azul: ({predicted_x:.1f}, {predicted_y:.1f})",
            f"Q={args.q} | R={args.r} | media={avg_ms:.2f} ms",
        ])
        cv2.imshow("CamShift + Kalman", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()

    if measured_points and predicted_points:
        measured = np.array(measured_points)
        predicted = np.array(predicted_points)
        plt.figure(figsize=(8, 5))
        plt.plot(measured[:, 0], measured[:, 1], "r.-", label="CamShift medido")
        plt.plot(predicted[:, 0], predicted[:, 1], "b.-", label="Kalman predito")
        plt.gca().invert_yaxis()
        plt.xlabel("x [pixels]")
        plt.ylabel("y [pixels]")
        plt.title("Trajetórias: medição CamShift vs predição Kalman")
        plt.grid(True)
        plt.legend()
        plot_path = output_path.with_suffix(".png")
        plt.savefig(plot_path, dpi=160, bbox_inches="tight")
        print(f"CSV salvo em: {output_path}")
        print(f"Gráfico salvo em: {plot_path}")
        plt.show()


if __name__ == "__main__":
    main()