# robot_classification.py
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


# === Joint waypoints ===
VIEW_JOINTS        = [-0.202, -0.540, -0.066, -0.399, -1.146, -0.663]        
WEIGHT_DROP_JOINTS = [-0.031167, -0.265639, -1.114273, -0.374199, -0.185704, 0.303821]
PICK_POSITION_JOINTS = [-0.355338, -1.014022, 0.556712, -0.041325, -1.172054, -0.584354]
LIFT_POSITION_JOINTS = [-0.219886, 0.138852, -0.791590, 0.190306, -0.958831, -0.372665]
LIFT_POSITION2_JOINTS = [-0.219886, 0.138852, -0.791590, 0.190306, -0.958831, -0.372665]
ALKALINE_DROP_JOINTS = [-1.336981, -0.533784, -0.146223, 0.119743, -0.945025, -1.696490]
NiMH_DROP_JOINTS = [-1.612450, -0.503485, -0.187126, -0.015247, -0.945025, -1.696490]
ZINC_DROP_JOINTS = [-1.082819, -0.586807, 0.006787, 0.201044, -0.945025, -1.696490]
LITHIUM_DROP_JOINTS = [-0.808872, -0.838288, 0.171916, 0.122811, -0.943491, -1.696490]
UNKNOWN_DROP_JOINTS = [-0.694728, -1.103404, 0.679423, 0.245530, -0.943491, -1.696490]

def main(page: ft.Page):
    page.title = "ZapSortBot | Robot Classification"
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
            log(f"Webcam error: {e}")

    page.add(
        ft.Row([
            ft.Text("ZapSortBot Robot Classification", size=22, weight=ft.FontWeight.BOLD)
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
        log("Connected to robot.")
        robot.calibrate_auto()
        robot.update_tool()
        log("Robot calibrated and tool updated.")
    except Exception as e:
        log("Robot connection or calibration failed:")
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

            robot.move_joints(VIEW_JOINTS)
            cap = cv2.VideoCapture(0)
            log("Waiting for battery detection...")
            

            while True:
                ret, frame = cap.read()
                if not ret:
                    log("Failed to read frame.")
                    continue

                zoomed_frame = zoom_center(frame)
                update_webcam_view(cv2.resize(zoomed_frame, (640, 480)))
                battery = detect_battery_from_frame(frame)
                if battery:
                    log(f"Initial detection: {battery['size']}, {battery['color']} | Length: {battery.get('length', '—')}")
                    time.sleep(2.0)
                    ret, frame = cap.read()
                    zoomed_frame = zoom_center(frame)
                    update_webcam_view(cv2.resize(zoomed_frame, (640, 480)))
                    if not ret:
                        log("Failed to read frame after delay.")
                        continue

                    battery = detect_battery_from_frame(frame)
                    if not battery:
                        log("Battery moved out of frame after delay. Skipping...")
                        continue

                    size = battery['size']
                    color = battery['color']
                    log(f"Final detection: {size}, {color} | Length: {battery.get('length', '—')}")

                    robot.open_gripper()
                    robot.move_joints(PICK_POSITION_JOINTS)
                    robot.close_gripper()
                    robot.move_joints(LIFT_POSITION_JOINTS)
                    robot.move_joints(WEIGHT_DROP_JOINTS)
                    robot.open_gripper()
                    time.sleep(1.0)

                    weight = None
                    for attempt in range(3):
                        try:
                            weight = get_weight_from_esp32(ESP32_IP)
                            if weight is not None:
                                break
                            log(f"Retry weight read {attempt + 1}/3...")
                            time.sleep(0.5)
                        except Exception as e:
                            log(f"Error reading weight: {e}")

                    if weight is None:
                        log("Failed to read weight after 3 attempts. Returning to view.")
                        robot.close_gripper()
                        robot.move_joints(VIEW_JOINTS)
                        continue

                    log(f"Weight = {weight:.2f} g")
                    robot.close_gripper()
                    robot.move_joints(LIFT_POSITION2_JOINTS)

                    classification = "unknown"
                    drop_joints = UNKNOWN_DROP_JOINTS

                    if size == "AA":
                        if color == "green" and 20 <= weight < 24:
                            classification = "unknown"
                            drop_joints = UNKNOWN_DROP_JOINTS
                        elif 20 <= weight < 24:
                            classification = "alkaline"
                            drop_joints = ALKALINE_DROP_JOINTS
                        elif 13 <= weight < 15 or 17<=weight<18:
                            classification = "lithium"
                            drop_joints = LITHIUM_DROP_JOINTS
                        elif color=="blue" and 24 <= weight <= 27:
                            classification = "unknown"
                            drop_joints = UNKNOWN_DROP_JOINTS
                        elif 24 <= weight <= 27:
                            classification = "NiMH"
                            drop_joints = NiMH_DROP_JOINTS
                        elif 14 <= weight < 17 or 10<=weight<13:
                            classification = "zinc"
                            drop_joints = ZINC_DROP_JOINTS
                        else:
                            classification = "unknown"
                            drop_joints = UNKNOWN_DROP_JOINTS
                    elif size == "AAA":
                        if color == "green" and 9 <= weight <= 11:
                            classification = "unknown"
                            drop_joints = UNKNOWN_DROP_JOINTS
                        elif 9 <= weight <= 11:
                            classification = "alkaline"
                            drop_joints = ALKALINE_DROP_JOINTS
                        elif 5 < weight < 9:
                            classification = "zinc"
                            drop_joints = ZINC_DROP_JOINTS
                        elif 3 <= weight < 5:
                            classification = "lithium"
                            drop_joints = LITHIUM_DROP_JOINTS
                        elif color=="blue" and 11 < weight <= 13:
                            classification = "unknown"
                            drop_joints = UNKNOWN_DROP_JOINTS
                        elif 11 < weight <= 13:
                            classification = "NiMH"
                            drop_joints = NiMH_DROP_JOINTS

                    log(f"Classed as {classification.upper()}")
                    robot.move_joints(drop_joints)
                    robot.open_gripper()
                    time.sleep(0.5)
                    robot.move_joints(VIEW_JOINTS)

        except Exception as e:
            log("Unexpected error during classification:")
            log(traceback.format_exc())
            try:
                robot.move_joints(VIEW_JOINTS)
            except:
                log("Could not return to view position.")

    page.add(
        ft.Row([
            ft.ElevatedButton("Start Classification", icon=Icons.PLAY_ARROW, bgcolor=Colors.BLUE_600,
                              on_click=lambda e: threading.Thread(target=run_classification, daemon=True).start()),
            ft.ElevatedButton("Exit", icon=Icons.CLOSE, bgcolor=Colors.PURPLE_700,
                              on_click=lambda e: page.window_close())
        ], alignment=ft.MainAxisAlignment.CENTER)
    )

ft.app(target=main)
VIEW_POSITION_JOINTS = [-0.245759, -0.418648, -0.250754, -0.484645, -0.859122, -0.183985]
