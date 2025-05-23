import flet as ft
import traceback
import threading
import time
import cv2
import base64
from pyniryo import NiryoRobot, PoseObject
from battery_detector import detect_battery_from_frame
from weight import get_weight_from_esp32
from flet import Colors, Icons

ROBOT_IP = "172.20.10.4"
ESP32_IP = "172.20.10.2"

# === Positions ===
VIEW_POSITION = PoseObject(0.351, 0.077, 0.219, 2.663, 1.049, 2.522)
WEIGHT_DROP = PoseObject(0.122, -0.159, 0.106, -1.028, 1.551, -2.579)
PICK_POSITION = PoseObject(0.378, 0.128, 0.151, -2.679, 1.554, -2.743)
LIFT_POSITION = PoseObject(0.174, 0.009, 0.223, -0.194, 0.897, -0.236)
LIFT_POSITION2 = PoseObject(0.137, -0.072, 0.153, -0.643, 1.230, -1.119)
ALKALINE_DROP = PoseObject(0.351, -0.124, 0.151, -1.326, 1.431, -1.440)
NiMH_DROP = PoseObject(0.238, -0.117, 0.127, -1.161, 1.429, -1.255)
ZINC_DROP = PoseObject(0.284, 0.026, 0.132, -0.630, 1.061, -0.431)
LITHIUM_DROP = PoseObject(0.189, 0.024, 0.120, -0.715, 1.303, -0.750)
UNKNOWN_DROP = PoseObject(0.443, -0.135, 0.166, -0.180, 1.354, -0.315)

def main(page: ft.Page):
    page.title = "DropBot | Robot Classification"
    page.scroll = ft.ScrollMode.AUTO
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = Colors.BLACK

    log_box = ft.TextField(
        multiline=True,
        min_lines=30,
        max_lines=30,
        expand=True,
        text_align=ft.TextAlign.LEFT,
        read_only=True
    )

    webcam_img = ft.Image(src="", width=640, height=480, expand=True)

    def log(msg):
        log_box.value += msg + "\n"
        log_box.update()

    def update_webcam_view(frame):
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            b64 = base64.b64encode(buffer).decode()
            webcam_img.src_base64 = b64
            webcam_img.update()
        except Exception as e:
            log(f"❌ Webcam error: {e}")

    page.add(
        ft.Row([
            ft.Icon(name=Icons.ANDROID, size=28, color=Colors.PINK_200),
            ft.Text("DropBot Robot Classification", size=22, weight=ft.FontWeight.BOLD)
        ], alignment=ft.MainAxisAlignment.CENTER)
    )

    page.add(
        ft.Row([
            webcam_img,
            log_box
        ], alignment=ft.MainAxisAlignment.SPACE_EVENLY, vertical_alignment=ft.CrossAxisAlignment.START)
    )

    robot = None
    try:
        robot = NiryoRobot(ROBOT_IP)
        log("✅ Connected to robot.")
        robot.calibrate_auto()
        robot.update_tool()
        log("🛠 Robot calibrated and tool updated.")
    except Exception as e:
        log("❌ Robot connection or calibration failed:")
        log(traceback.format_exc())
        return

    def zoom_center(frame, zoom_ratio=0.5):
        h, w = frame.shape[:2]
        crop_h, crop_w = int(h * zoom_ratio), int(w * zoom_ratio)
        y1 = h // 2 - crop_h // 2
        x1 = w // 2 - crop_w // 2
        return frame[y1:y1 + crop_h, x1:x1 + crop_w]

    def run_classification():
        try:
            cap = cv2.VideoCapture(0)
            log("🔍 Waiting for battery detection...")
            robot.move_pose(VIEW_POSITION)

            while True:
                ret, frame = cap.read()
                if not ret:
                    log("⚠️ Failed to read frame.")
                    continue

                zoomed_frame = zoom_center(frame)
                update_webcam_view(cv2.resize(zoomed_frame, (640, 480)))
                battery = detect_battery_from_frame(frame)
                if battery:
                    log(f"🔍 Initial detection: {battery['size']}, {battery['color']} | Length: {battery.get('length', '—')}")
                    time.sleep(2.0)
                    ret, frame = cap.read()
                    zoomed_frame = zoom_center(frame)
                    update_webcam_view(cv2.resize(zoomed_frame, (640, 480)))
                    if not ret:
                        log("⚠️ Failed to read frame after delay.")
                        continue

                    battery = detect_battery_from_frame(frame)
                    if not battery:
                        log("⚠️ Battery moved out of frame after delay. Skipping...")
                        continue

                    size = battery['size']
                    color = battery['color']
                    log(f"🔄 Final detection: {size}, {color} | Length: {battery.get('length', '—')}")

                    robot.open_gripper()
                    robot.move_pose(PICK_POSITION)
                    robot.close_gripper()
                    robot.move_pose(LIFT_POSITION)
                    robot.move_pose(WEIGHT_DROP)
                    robot.open_gripper()
                    time.sleep(1.0)

                    weight = None
                    for attempt in range(3):
                        try:
                            weight = get_weight_from_esp32(ESP32_IP)
                            if weight is not None:
                                break
                            log(f"⚠️ Retry weight read {attempt + 1}/3...")
                            time.sleep(0.5)
                        except Exception as e:
                            log(f"⚠️ Error reading weight: {e}")

                    if weight is None:
                        log("❌ Failed to read weight after 3 attempts. Returning to view.")
                        robot.close_gripper()
                        robot.move_pose(VIEW_POSITION)
                        continue

                    log(f"⚖️ Weight = {weight:.2f} g")
                    robot.close_gripper()
                    robot.move_pose(LIFT_POSITION2)

                    classification = "unknown"
                    drop_pose = UNKNOWN_DROP

                    if size == "AA":
                        if color == "green" and 20 <= weight < 24:
                            classification = "unknown"
                            drop_pose = UNKNOWN_DROP
                        elif 20 <= weight < 24:
                            classification = "alkaline"
                            drop_pose = ALKALINE_DROP
                        elif 13 <= weight < 15 or 17<=weight<18:
                            classification = "lithium"
                            drop_pose = LITHIUM_DROP
                        elif color=="blue" and 24 <= weight <= 27:
                            classification = "unknown"
                            drop_pose = UNKNOWN_DROP
                        elif 24 <= weight <= 27:
                            classification = "NiMH"
                            drop_pose = NiMH_DROP
                        elif 14 <= weight < 17 or 10<=weight<13:
                            classification = "zinc"
                            drop_pose = ZINC_DROP
                        else:
                            classification = "unknown"
                            drop_pose = UNKNOWN_DROP
                    elif size == "AAA":
                        if color == "green" and 9 <= weight <= 11:
                            classification = "unknown"
                            drop_pose = UNKNOWN_DROP
                        elif 9 <= weight <= 11:
                            classification = "alkaline"
                            drop_pose = ALKALINE_DROP
                        elif 5 < weight < 9:
                            classification = "zinc"
                            drop_pose = ZINC_DROP
                        elif 3 <= weight < 5:
                            classification = "lithium"
                            drop_pose = LITHIUM_DROP
                        elif color=="blue" and 11 < weight <= 13:
                            classification = "unknown"
                            drop_pose = UNKNOWN_DROP
                        elif 11 < weight <= 13:
                            classification = "NiMH"
                            drop_pose = NiMH_DROP

                    log(f"🔹 Classed as {classification.upper()}")
                    robot.move_pose(drop_pose)
                    robot.open_gripper()
                    time.sleep(0.5)
                    robot.move_pose(VIEW_POSITION)

        except Exception as e:
            log("❌ Unexpected error during classification:")
            log(traceback.format_exc())
            try:
                robot.move_pose(VIEW_POSITION)
            except:
                log("⚠️ Could not return to view position.")

    page.add(
        ft.Row([
            ft.ElevatedButton("▶ Start Classification", icon=Icons.PLAY_ARROW, bgcolor=Colors.BLUE_600,
                              on_click=lambda e: threading.Thread(target=run_classification, daemon=True).start()),
            ft.ElevatedButton("❌ Exit", icon=Icons.CLOSE, bgcolor=Colors.PURPLE_700,
                              on_click=lambda e: page.window_close())
        ], alignment=ft.MainAxisAlignment.CENTER)
    )

ft.app(target=main)
