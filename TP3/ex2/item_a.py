import argparse
import time
from collections import deque

import cv2
import numpy as np

from utils import open_capture, resize_keep_aspect, clean_foreground_mask, contours_from_mask, annotate_contours, overlay_text


def process(method_name, subtractor, frame, min_area):
    tic = time.perf_counter()
    mask = subtractor.apply(frame)
    clean = clean_foreground_mask(mask)
    contours = contours_from_mask(clean, min_area=min_area)
    elapsed_ms = (time.perf_counter() - tic) * 1000
    annotated = annotate_contours(frame, contours, color=(0, 255, 0) if method_name == "MOG2" else (255, 128, 0))
    return clean, annotated, contours, elapsed_ms


def main():
    parser = argparse.ArgumentParser(description="Compara MOG2 e KNN lado a lado.")
    parser.add_argument("--source", default="ex2\data\sample_video.mp4")
    parser.add_argument("--min-area", type=float, default=500)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()

    cap = open_capture(args.source)
    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    knn = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)

    times_mog2 = deque(maxlen=120)
    times_knn = deque(maxlen=120)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_keep_aspect(frame, args.width)
        frame_idx += 1

        mask_mog2, ann_mog2, cont_mog2, t_mog2 = process("MOG2", mog2, frame, args.min_area)
        mask_knn, ann_knn, cont_knn, t_knn = process("KNN", knn, frame, args.min_area)
        times_mog2.append(t_mog2)
        times_knn.append(t_knn)
        avg_mog2 = sum(times_mog2) / len(times_mog2)
        avg_knn = sum(times_knn) / len(times_knn)

        ann_mog2 = overlay_text(ann_mog2, [f"MOG2 | objetos: {len(cont_mog2)}", f"media: {avg_mog2:.2f} ms"])
        ann_knn = overlay_text(ann_knn, [f"KNN | objetos: {len(cont_knn)}", f"media: {avg_knn:.2f} ms"])

        masks = np.hstack([cv2.cvtColor(mask_mog2, cv2.COLOR_GRAY2BGR), cv2.cvtColor(mask_knn, cv2.COLOR_GRAY2BGR)])
        frames = np.hstack([ann_mog2, ann_knn])

        print(
            f"frame={frame_idx:04d} | "
            f"MOG2 objetos={len(cont_mog2)} tempo={t_mog2:.2f}ms media={avg_mog2:.2f}ms | "
            f"KNN objetos={len(cont_knn)} tempo={t_knn:.2f}ms media={avg_knn:.2f}ms"
        )

        cv2.imshow("Mascaras: MOG2 esquerda | KNN direita", masks)
        cv2.imshow("Frames anotados: MOG2 esquerda | KNN direita", frames)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
