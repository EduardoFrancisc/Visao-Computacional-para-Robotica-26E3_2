import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


def parse_source(value):
    """Converte '0' em câmera 0; demais valores são tratados como caminho/URL."""
    try:
        return int(value)
    except ValueError:
        return value


def open_capture(source):
    cap = cv2.VideoCapture(parse_source(source))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {source}")
    return cap


def resize_keep_aspect(frame, width=None):
    if width is None or frame.shape[1] <= width:
        return frame
    scale = width / frame.shape[1]
    return cv2.resize(frame, None, fx=scale, fy=scale)


def clean_foreground_mask(mask, kernel_size=5, min_area=500):
    """Aplica limiarização e morfologia para reduzir ruído no foreground."""
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def contours_from_mask(mask, min_area=500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            valid.append(c)
    return valid


def annotate_contours(frame, contours, color=(0, 255, 0)):
    annotated = frame.copy()
    for idx, c in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, f"obj {idx}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated


def moving_average(values, maxlen=60):
    dq = deque(maxlen=maxlen)
    for v in values:
        dq.append(v)
    return sum(dq) / max(1, len(dq))


def overlay_text(frame, lines, x=10, y=24, color=(255, 255, 255)):
    out = frame.copy()
    for i, line in enumerate(lines):
        yy = y + i * 24
        cv2.putText(out, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(out, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    return out


def ensure_output_dir():
    Path("output").mkdir(exist_ok=True)
    return Path("output")
