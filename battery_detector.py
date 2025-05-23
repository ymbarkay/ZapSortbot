import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression

model = YOLO("best.pt")

# === Helpers ===
def normalize_lighting(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def classify_color(hsv_pixel):
    h, s, v = map(int, hsv_pixel)

    if v < 40 and s < 40:
        return "empty"

    if 38 <= h <= 50 and v > 120:
        return "gold"

    if (h <= 20 or 160 <= h <= 180) and s >= 60:
        return "red/orange"

    if 45 <= h <= 90 and s >= 45 and v >= 55:
        return "green"

    if 100 <= h <= 130 and s >= 45 and v >= 55:
        return "blue"

    if s < 30 and v > 100 and not (35 <= h <= 55):
        return "white/gray"

    if s < 35 or v < 50:
        return "others"

    return "others"

def get_robust_color(hsv_pixels):
    v_values = hsv_pixels[:, 2]
    v_min, v_max = np.percentile(v_values, [20, 80])
    filtered = hsv_pixels[(v_values >= v_min) & (v_values <= v_max)]
    return np.median(filtered, axis=0) if len(filtered) > 0 else np.mean(hsv_pixels, axis=0)

def infer_rotated_size_from_crop(crop):
    h, w = crop.shape[:2]
    length = max(h, w)
    if 110<= length < 130:
        return "AAA", length
    elif 130 <= length < 150:
        return "AA", length
    else:
        return "unknown", length

def detect_battery_from_frame(frame):
    zoom_ratio = 0.5
    h, w = frame.shape[:2]
    crop_h, crop_w = int(h * zoom_ratio), int(w * zoom_ratio)
    y1 = h // 2 - crop_h // 2
    x1 = w // 2 - crop_w // 2
    frame = frame[y1:y1 + crop_h, x1:x1 + crop_w]

    raw_results = model.predict(source=frame, conf=0.25, verbose=False)[0]
    boxes_tensor = raw_results.boxes.data.unsqueeze(0) if isinstance(raw_results.boxes.data, torch.Tensor) else torch.tensor(raw_results.boxes.data).unsqueeze(0)
    boxes = non_max_suppression(boxes_tensor, conf_thres=0.4, iou_thres=0.5)[0]

    if len(boxes) == 0:
        return None

    box = sorted(boxes, key=lambda x: x[4], reverse=True)[0]
    x1, y1, x2, y2, conf, cls = map(float, box[:6])
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = height / width if width != 0 else 0

    if width > 0.8 * w or height > 0.8 * h or aspect_ratio > 5 or aspect_ratio < 1.2:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None

    crop = normalize_lighting(crop)
    size_label, length = infer_rotated_size_from_crop(crop)
    if size_label is None:
        return None

    hsv = cv2.cvtColor(cv2.GaussianBlur(crop, (11, 11), 0), cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    avg_hsv = get_robust_color(pixels)
    color_label = classify_color(avg_hsv)

    return {
        "size": size_label,
        "length": int(length),
        "color": color_label,
        "confidence": conf,
        "box": (x1, y1, x2, y2)
    }
