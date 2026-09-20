import time
import cv2
import numpy as np
from dnn_utils import (
    read_pipeline_frame, DEFAULT_K, DEFAULT_DIST, OUT,
    segment_red_hsv, orb_features, hog_or_haar_detector,
    predict_opencv, topk, draw_top3
)


frame_orig, path = read_pipeline_frame()
t_total_inicio = time.time()

# ---------------------------------------------------------
# (1) Correção de Distorção (Undistort)
# ---------------------------------------------------------
t0 = time.time()
frame = cv2.undistort(frame_orig, DEFAULT_K, DEFAULT_DIST)
t_undistort = (time.time() - t0) * 1000

painel = np.hstack([frame_orig, frame])
cv2.putText(painel, 'original', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
cv2.putText(painel, 'undistort', (frame_orig.shape[1] + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
cv2.imwrite(str(OUT / '1_undistort_painel.jpg'), painel)

final_frame = frame.copy()

# ---------------------------------------------------------
# (2) Segmentação de ROI por HSV
# ---------------------------------------------------------
t0 = time.time()
mask, roi = segment_red_hsv(frame)
t_hsv = (time.time() - t0) * 1000

vis_hsv = frame.copy() 

if roi:
    x, y, w, h = roi
    cv2.rectangle(vis_hsv, (x, y), (x+w, y+h), (0, 255, 255), 3)
    cv2.putText(vis_hsv, 'ROI HSV', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.rectangle(final_frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
    cv2.putText(final_frame, 'ROI HSV', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    crop = frame[y:y+h, x:x+w]
else:
    x, y, w, h = 0, 0, frame.shape[1], frame.shape[0]
    crop = frame

cv2.imwrite(str(OUT / '2_roi_hsv.jpg'), vis_hsv)

# ---------------------------------------------------------
# (3) Extração de features ORB
# ---------------------------------------------------------
t0 = time.time()
kp, des = orb_features(crop)
t_orb = (time.time() - t0) * 1000

vis_orb = cv2.drawKeypoints(crop, kp, None, color=(0, 255, 0), flags=0)
cv2.imwrite(str(OUT / '3_orb_roi.jpg'), vis_orb)

if roi:
    for p in kp:
        p.pt = (p.pt[0] + x, p.pt[1] + y)
cv2.drawKeypoints(final_frame, kp, final_frame, color=(0, 255, 0), flags=0)

# ---------------------------------------------------------
# (4) Detecção com HOG+SVM / Haar Cascade
# ---------------------------------------------------------
t0 = time.time()
boxes = hog_or_haar_detector(frame)
t_detectores = (time.time() - t0) * 1000

vis_hog_haar = frame.copy() 

if boxes:
    for nome, bx, by, bw, bh in boxes:
        cv2.rectangle(vis_hog_haar, (bx, by), (bx+bw, by+bh), (255, 0, 0), 2)
        cv2.putText(vis_hog_haar, nome, (bx, by-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.rectangle(final_frame, (bx, by), (bx+bw, by+bh), (255, 0, 0), 2)
        cv2.putText(final_frame, nome, (bx, by-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
else:
    cv2.putText(vis_hog_haar, 'HOG/Haar: sem deteccao', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 180), 2)
    cv2.putText(final_frame, 'HOG/Haar: sem deteccao', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 180), 2)

cv2.imwrite(str(OUT / '4_detector_hog_haar.jpg'), vis_hog_haar)

# ---------------------------------------------------------
# (5) Classificação DNN da ROI
# ---------------------------------------------------------
t0 = time.time()
prob = predict_opencv(crop)
t_dnn = (time.time() - t0) * 1000

vis_dnn = frame.copy()
vis_dnn = draw_top3(vis_dnn, topk(prob, 3))
if roi:
    cv2.rectangle(vis_dnn, (x, y), (x+w, y+h), (0, 255, 255), 3)
cv2.imwrite(str(OUT / '5_classificar_roi.jpg'), vis_dnn)

final_frame = draw_top3(final_frame, topk(prob, 3))

# ---------------------------------------------------------
# Fechamento e Relatórios
# ---------------------------------------------------------
t_total_fim = time.time()
t_total = (t_total_fim - t_total_inicio) * 1000

caminho_final = str(OUT / '6_pipeline_completo.jpg')
cv2.imwrite(caminho_final, final_frame)

print("=== Relatório de Execução do Pipeline (ms) ===")
print(f"(1) Undistort:         {t_undistort:.2f} ms")
print(f"(2) Segmentação HSV:   {t_hsv:.2f} ms")
print(f"(3) Extração ORB:      {t_orb:.2f} ms")
print(f"(4) Detecção HOG/Haar: {t_detectores:.2f} ms")
print(f"(5) Classificação DNN: {t_dnn:.2f} ms")
print("-" * 46)
print(f"Tempo Total:           {t_total:.2f} ms")
print(f"\nImagens individuais e pipeline completo salvos na pasta 'saidas/'!")