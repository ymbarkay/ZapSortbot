import flet as ft
import threading
import cv2
import base64
import time
import traceback
from pyniryo import NiryoRobot
from battery_detector import detect_battery_from_frame

ROBOT_IP = "172.20.10.4"
POSE_FILE = "robot_classification.py"

pose_names = [
    "VIEW_POSITION", "WEIGHT_DROP", "PICK_POSITION", "LIFT_POSITION",
    "LIFT_POSITION2", "ALKALINE_DROP", "NiMH_DROP", "ZINC_DROP",
    "LITHIUM_DROP", "UNKNOWN_DROP"
]

def save_pose_to_file(name, pose):
    try:
        with open(POSE_FILE, 'r', encoding="utf-8") as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{name} = PoseObject"):
                lines[i] = f"{name} = PoseObject({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f}, {pose.roll:.3f}, {pose.pitch:.3f}, {pose.yaw:.3f})\n"
                break
        with open(POSE_FILE, 'w', encoding="utf-8") as file:
            file.writelines(lines)
        return True
    except Exception as e:
        return str(e)

def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode()

def main(page: ft.Page):
    page.title = "DropBot | Pose Editor"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = ft.colors.BLACK

    # === UI Elements ===
    status_text = ft.Text("", size=12, color=ft.colors.GREEN_400)
    position_text = ft.Text("Position: —", size=14, color=ft.colors.BLUE_200)
    detection_text = ft.Text("Live Detection: —", size=14, color=ft.colors.ORANGE_200)
    dropdown = ft.Dropdown(
        label="Select a Pose to Update",
        options=[ft.dropdown.Option(name) for name in pose_names],
        width=300
    )
    image = ft.Image(width=480, height=360, fit=ft.ImageFit.CONTAIN)

    # === Robot Connection ===
    global robot
    robot = None
    try:
        robot = NiryoRobot(ROBOT_IP)
        robot.update_tool()
        robot.calibrate_auto()
        status_text.value = "✅ Connected to robot and tool updated."
        status_text.color = ft.colors.GREEN_400
    except Exception as e:
        status_text.value = "❌ Failed to connect or calibrate robot:\n" + str(e)
        status_text.color = ft.colors.RED_400

    # === Pose Save ===
    def update_pose(e):
        pose_name = dropdown.value
        if robot is None:
            status_text.value = "❌ Robot not connected."
            status_text.color = ft.colors.RED_400
            page.update()
            return
        try:
            pose = robot.get_pose()
            result = save_pose_to_file(pose_name, pose)
            if result is True:
                status_text.value = f"✅ {pose_name} updated."
                status_text.color = ft.colors.GREEN_400
            else:
                status_text.value = f"❌ Failed to save file: {result}"
                status_text.color = ft.colors.RED_400
        except Exception as e:
            status_text.value = f"❌ Error: {e}"
            status_text.color = ft.colors.RED_400
        page.update()

    # === Webcam Stream + Inference ===
    def stream_and_infer():
        cap = cv2.VideoCapture(0)
        zoom_ratio = 0.5

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Apply zoom crop
            h, w = frame.shape[:2]
            crop_h, crop_w = int(h * zoom_ratio), int(w * zoom_ratio)
            y1 = h // 2 - crop_h // 2
            x1 = w // 2 - crop_w // 2
            cropped_frame = frame[y1:y1 + crop_h, x1:x1 + crop_w]

            result = detect_battery_from_frame(frame)
            if result:
                length = result.get("length", "—")
                label = f"{result['size']} | {result['color']} | Length: {length}"
                detection_text.value = f"🔍 Detected: {label}"
            else:
                detection_text.value = "🔍 No battery detected."

            image.src_base64 = encode_frame(cropped_frame)
            page.update()
            time.sleep(0.1)
        cap.release()

    # === Position Tracker ===
    def update_position_loop():
        while True:
            try:
                if robot:
                    pose = robot.get_pose()
                    pos = f"x: {pose.x:.3f}, y: {pose.y:.3f}, z: {pose.z:.3f}, roll: {pose.roll:.3f}, pitch: {pose.pitch:.3f}, yaw: {pose.yaw:.3f}"
                    position_text.value = f"📍 Live Position: {pos}"
                    page.update()
                time.sleep(1)
            except:
                break

    threading.Thread(target=stream_and_infer, daemon=True).start()
    threading.Thread(target=update_position_loop, daemon=True).start()

    # === Layout ===
    page.add(
        ft.Column([
            ft.Row([ft.Text("🤖 DropBot Pose Editor", size=22, weight=ft.FontWeight.BOLD)]),
            dropdown,
            ft.ElevatedButton("📍 Save Current Position", on_click=update_pose, bgcolor=ft.colors.BLUE_600),
            position_text,
            detection_text,
            status_text,
            ft.Divider(),
            image,
            ft.ElevatedButton("Exit", icon=ft.icons.CLOSE, bgcolor=ft.colors.PURPLE_700, on_click=lambda e: page.window_close())
        ], spacing=20, alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
    )

    def on_window_close(e):
        if robot:
            robot.close_connection()
        page.window_destroy()

    page.on_window_close = on_window_close

ft.app(target=main)
