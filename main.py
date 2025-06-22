import flet as ft
import subprocess
import os

def main(page: ft.Page):
    page.title = "Zapsortbot | Control Panel"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = ft.colors.BLACK
    page.scroll = ft.ScrollMode.AUTO

    SCRIPTS = {
        "Robot Classification": ("robot_classification.py", "[AI]"),
        "Manage Dataset": ("annotation.py", "[DB]"),
        "Test Inference": ("test_inference.py", "[CAM]"),
        "Update Robot Positions": ("pose_editor.py", "[BOT]")
    }

    def run_script(path):
        if not os.path.exists(path):
            page.snack_bar = ft.SnackBar(ft.Text(f"Script not found: {path}"), bgcolor=ft.colors.RED_700)
            page.snack_bar.open = True
            page.update()
            return
        subprocess.Popen(["python", path], shell=True)

    # === Title ===
    title = ft.Row([
        ft.Icon(name=ft.icons.BOLT, color=ft.colors.PINK_300, size=20),
        ft.Text(
            "Zapsortbot Control Panel",
            size=18,
            weight=ft.FontWeight.BOLD,
            color=ft.colors.GREY_200,
            font_family="Roboto",
            text_align=ft.TextAlign.LEFT
        )
    ], alignment=ft.MainAxisAlignment.START, spacing=10)
    page.add(title)

    # === Script Cards ===
    card_rows = []
    row = []

    for title, (script, emoji) in SCRIPTS.items():
        def create_card(script_path=script, label=title, icon=emoji):
            return ft.Container(
                content=ft.Column([
                    ft.Text(f"{icon}", size=26),
                    ft.Text(label, size=16, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER)
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                width=220,
                height=120,
                border_radius=20,
                padding=20,
                alignment=ft.alignment.center,
                bgcolor=ft.colors.with_opacity(0.1, ft.colors.ON_SURFACE),
                ink=True,
                on_click=lambda e: run_script(script_path)
            )

        row.append(create_card())
        if len(row) == 2:
            card_rows.append(ft.Row(controls=row, alignment=ft.MainAxisAlignment.CENTER))
            row = []
    if row:
        card_rows.append(ft.Row(controls=row, alignment=ft.MainAxisAlignment.CENTER))

    page.add(ft.Column(card_rows, alignment=ft.MainAxisAlignment.CENTER, spacing=25))

    # === Footer ===
    page.add(ft.Divider())
    page.add(ft.Text("Welcome to Zapsortbot!", size=12, text_align=ft.TextAlign.CENTER))
    page.add(
        ft.ElevatedButton("Exit", icon=ft.icons.CLOSE, bgcolor=ft.colors.PURPLE_700,
                          on_click=lambda e: page.window_close())
    )

ft.app(target=main)
