import flet as ft
import cv2
import numpy as np
import torch
import threading
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression

model = YOLO("best.pt")

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

def main(page: ft.Page):
    page.title = "Zapsortbot | Test Inference"
    page.scroll = ft.ScrollMode.AUTO
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = ft.colors.BLACK  # Fixed case

    log_box = ft.TextField(
        multiline=True,
        read_only=True,
        expand=True,
        max_lines=40,
        text_style=ft.TextStyle(font_family="Consolas", size=12),
        bgcolor="#0f0f1f",
        color="white"
    )

    def log(msg):
        log_box.value += msg + "\n"
        log_box.update()

    def run_test_inference():
        def _infer():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                log("❌ Could not open webcam.")
                return

            log("📸 Test Inference Started — Press 'q' in the window to quit.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    log("⚠️ Could not grab frame.")
                    continue

                h, w = frame.shape[:2]
                zoom_ratio = 0.5
                crop_h, crop_w = int(h * zoom_ratio), int(w * zoom_ratio)
                y1 = h // 2 - crop_h // 2
                x1 = w // 2 - crop_w // 2
                frame = frame[y1:y1 + crop_h, x1:x1 + crop_w]

                raw_results = model.predict(source=frame, conf=0.25, verbose=False)[0]
                boxes_tensor = raw_results.boxes.data.unsqueeze(0) if isinstance(raw_results.boxes.data, torch.Tensor) else torch.tensor(raw_results.boxes.data).unsqueeze(0)
                boxes = non_max_suppression(boxes_tensor, conf_thres=0.4, iou_thres=0.5)[0]

                if len(boxes) == 0:
                    cv2.imshow("🔋 Test Inference (Zoomed)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                boxes = sorted(boxes, key=lambda x: x[4], reverse=True)[:1]
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = map(float, box[:6])
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    crop = frame[y1:y2, x1:x2]
                    crop = normalize_lighting(crop)

                    size_label, length = infer_rotated_size_from_crop(crop)
                    hsv = cv2.cvtColor(cv2.GaussianBlur(crop, (11, 11), 0), cv2.COLOR_BGR2HSV)
                    pixels = hsv.reshape(-1, 3)
                    avg_hsv = get_robust_color(pixels)
                    color_label = classify_color(avg_hsv)

                    label = f"{size_label}, {color_label} ({conf:.2f}) [len: {int(length)}]"
                    log(f"📏 Detected size: {size_label} | Length: {int(length)}")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                cv2.imshow("🔋 Test Inference (Zoomed)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            log("🛑 Test Inference Ended.")

        threading.Thread(target=_infer, daemon=True).start()

    title = ft.Text("🔍 DropBot | Battery Inference", size=22, weight=ft.FontWeight.BOLD)
    start_btn = ft.ElevatedButton("▶ Run Test Inference", on_click=lambda e: run_test_inference(), bgcolor=ft.colors.BLUE_GREY_700)
    exit_btn = ft.ElevatedButton("❌ Exit", on_click=lambda e: page.window_close(), bgcolor=ft.colors.PURPLE_700)

    page.add(
        ft.Column([
            title,
            log_box,
            ft.Row([start_btn, exit_btn], alignment=ft.MainAxisAlignment.CENTER)
        ], spacing=20, expand=True, alignment=ft.MainAxisAlignment.CENTER)
    )

ft.app(target=main)