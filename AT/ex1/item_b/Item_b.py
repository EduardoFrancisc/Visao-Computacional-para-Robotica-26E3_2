from pathlib import Path
import cv2
import numpy as np
from synthetic_utils import (
    generate_view, find_refined_corners, object_points,
    TRUE_K, TRUE_DIST, SQUARE_SIZE_M
)

ROOT = Path(__file__).resolve().parent
saida = ROOT / "saidas" / "tabuleiro_video_faces_coloridas.avi"

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(str(saida), fourcc, 15.0, (1280,720))

s = SQUARE_SIZE_M
cube = np.float32([
    [2*s, s, 0],   [3*s, s, 0],   [3*s, 2*s, 0],   [2*s, 2*s, 0],
    [2*s, s, -s],  [3*s, s, -s],  [3*s, 2*s, -s],  [2*s, 2*s, -s]
])

faces = [
    ([0,1,2,3], (0,255,0)),
    ([4,5,6,7], (255,0,0)),
    ([0,1,5,4], (0,255,255)),
    ([1,2,6,5], (0,0,255)),
    ([2,3,7,6], (255,255,0)),
    ([3,0,4,7], (255,0,255)),
]

for i in range(90):
    a = i / 89.0
    frame, _, _ = generate_view(
        8 + 8*np.sin(a*np.pi*2),
        -12 + 12*np.sin(a*np.pi),
        6*np.sin(a*np.pi*2),
        -0.10 + 0.02*np.sin(a*np.pi*2),
        -0.07,
        0.75 + 0.05*np.sin(a*np.pi*2)
    )

    ok, corners = find_refined_corners(frame)

    if ok:
        _, rvec, tvec = cv2.solvePnP(
            object_points(), corners, TRUE_K, TRUE_DIST
        )
        print(f"--- Frame {i} ---")
        print(f"Rotação (rvec):\n{rvec.ravel()}")
        print(f"Translação (tvec):\n{tvec.ravel()}\n")
        
        pts, _ = cv2.projectPoints(
            cube, rvec, tvec, TRUE_K, TRUE_DIST
        )
        p = pts.reshape(-1,2).astype(int)

        overlay = frame.copy()

        for ids, color in faces:
            poly = np.array([p[idx] for idx in ids], np.int32)
            cv2.fillConvexPoly(overlay, poly, color)

        frame = cv2.addWeighted(overlay, 0.30, frame, 0.70, 0)

        for ids, _ in faces:
            poly = np.array([p[idx] for idx in ids], np.int32)
            cv2.polylines(frame, [poly], True, (30,30,30), 2)

    writer.write(frame)

writer.release()
print("Vídeo AR com faces coloridas criado em:", saida)