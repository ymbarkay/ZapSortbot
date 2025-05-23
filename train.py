from ultralytics import YOLO

# Load base YOLOv8 model
model = YOLO("yolov8n.pt")

# Train with recommended augmentations and params
model.train(
    data="classes.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="battery-detect-v2",
    optimizer="SGD",  # more stable for small datasets
    lr0=0.005,
    warmup_epochs=3,
    weight_decay=0.001,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.0,
    fliplr=0.5,
    translate=0.1,
    scale=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    auto_augment="randaugment",
    erasing=0.4,
    cache=False,
    workers=4,
)
