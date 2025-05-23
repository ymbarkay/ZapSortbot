import flet as ft
import yaml
import os
import cv2
from datetime import datetime
import threading

CLASS_FILE = "classes.yaml"
IMAGES_PER_CLASS = 10
ZOOM_RATIO = 0.5

paths = {
    "train_img": "images/train",
    "val_img": "images/val",
    "train_lbl": "labels/train",
    "val_lbl": "labels/val"
}

for p in paths.values():
    os.makedirs(p, exist_ok=True)

def load_classes():
    if os.path.exists(CLASS_FILE):
        with open(CLASS_FILE, 'r') as f:
            return yaml.safe_load(f) or {"names": []}
    return {"names": []}

def save_classes(classes):
    with open(CLASS_FILE, 'w') as f:
        yaml.dump(classes, f)

def main(page: ft.Page):
    page.title = "DropBot | Dataset Manager"
    page.scroll = ft.ScrollMode.AUTO
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = ft.colors.BLACK

    class_data = load_classes()
    class_options = class_data["names"]
    class_dropdown = ft.Dropdown(label="Select Class", options=[ft.dropdown.Option(n) for n in class_options], width=280)
    new_class_field = ft.TextField(label="New Class Name", width=280)
    status_text = ft.Text("", size=12, color=ft.colors.GREEN_400)

    def refresh_dropdown():
        class_dropdown.options = [ft.dropdown.Option(n) for n in class_data["names"]]
        page.update()

    def add_class(e):
        name = new_class_field.value.strip()
        if name and name not in class_data["names"]:
            class_data["names"].append(name)
            save_classes(class_data)
            refresh_dropdown()
            class_dropdown.value = name
            new_class_field.value = ""
            status_text.value = f"✅ Added class: {name}"
        else:
            status_text.value = "⚠️ Class already exists or invalid."
        page.update()

    def delete_class(e):
        selected = class_dropdown.value
        if selected in class_data["names"]:
            class_data["names"].remove(selected)
            save_classes(class_data)
            refresh_dropdown()
            class_dropdown.value = None
            status_text.value = f"❌ Deleted class: {selected}"
            page.update()

    def capture_images(e):
        def _capture():
            nonlocal status_text
            selected = class_dropdown.value
            if not selected:
                status_text.value = "⚠️ Please select a class first."
                page.update()
                return

            cap = cv2.VideoCapture(0)
            box = []
            ix, iy = -1, -1
            drawing = False

            def draw_box(event, x, y, flags, param):
                nonlocal box, ix, iy, drawing
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    ix, iy = x, y
                    box = [(ix, iy)]
                elif event == cv2.EVENT_MOUSEMOVE and drawing:
                    box = [(ix, iy), (x, y)]
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                    box.append((x, y))

            cv2.namedWindow("Press 's' to Save Datum")
            cv2.setMouseCallback("Press 's' to Save Datum", draw_box)

            count = 0
            while count < IMAGES_PER_CLASS:
                ret, frame = cap.read()
                if not ret:
                    continue
                h, w = frame.shape[:2]
                ch, cw = int(h * ZOOM_RATIO), int(w * ZOOM_RATIO)
                y1, x1 = h // 2 - ch // 2, w // 2 - cw // 2
                frame = frame[y1:y1 + ch, x1:x1 + cw]

                display = frame.copy()
                if len(box) == 2:
                    cv2.rectangle(display, box[0], box[1], (0, 255, 0), 2)
                cv2.putText(display, f"Captured: {count}/{IMAGES_PER_CLASS}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Press 's' to Save Datum", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s') and len(box) == 2:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_name = f"{selected}_{timestamp}_{count}.jpg"
                    lbl_name = img_name.replace(".jpg", ".txt")
                    is_val = ((count + 1) % 3 == 0)
                    img_path = os.path.join(paths["val_img" if is_val else "train_img"], img_name)
                    lbl_path = os.path.join(paths["val_lbl" if is_val else "train_lbl"], lbl_name)
                    cv2.imwrite(img_path, frame)
                    x1, y1 = box[0]
                    x2, y2 = box[1]
                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])
                    x_c = (x1 + x2) / 2 / frame.shape[1]
                    y_c = (y1 + y2) / 2 / frame.shape[0]
                    w_b = (x2 - x1) / frame.shape[1]
                    h_b = (y2 - y1) / frame.shape[0]
                    with open(lbl_path, 'w') as f:
                        f.write(f"{class_data['names'].index(selected)} {x_c} {y_c} {w_b} {h_b}\n")
                    box.clear()
                    count += 1
                elif key == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            status_text.value = "✅ Capture complete."
            page.update()

        threading.Thread(target=_capture, daemon=True).start()

    content_column = ft.Column([
        class_dropdown,
        new_class_field,
        ft.Row([
            ft.ElevatedButton("➕", on_click=add_class, bgcolor=ft.colors.with_opacity(0.2, ft.colors.GREEN_200)),
            ft.ElevatedButton("❌", on_click=delete_class, bgcolor=ft.colors.with_opacity(0.2, ft.colors.RED_200)),
        ], alignment=ft.MainAxisAlignment.CENTER),
        ft.ElevatedButton("📸 Capture & Annotate", on_click=capture_images, bgcolor=ft.colors.BLUE_600),
        status_text,
        ft.Divider(),
        ft.ElevatedButton("Exit", icon=ft.icons.CLOSE, bgcolor=ft.colors.PURPLE_700, on_click=lambda e: page.window_close())
    ], spacing=20, alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER)

    page.add(ft.Row([
        ft.Text("📂 DropBot Dataset Manager", size=20, weight=ft.FontWeight.BOLD)
    ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=20, expand=True))

    page.add(ft.Container(
        content=content_column,
        margin=ft.margin.only(top=10),
        alignment=ft.alignment.center
    ))

ft.app(target=main)