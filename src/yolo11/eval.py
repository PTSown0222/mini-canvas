from ultralytics import YOLO

# Load weights
model = YOLO("models/yolo11n_car_detect_20ep.pt")
metrics = model.val(
    data="src/yolo11/car_dataset.yaml",
    split="val",      
    batch=16,
    imgsz=640,
    device="mps",     
)

print("\n========== EVALUATION METRICS ==========")
print(f"mAP@50-95 : {metrics.box.map:.4f}")      
print(f"mAP@50    : {metrics.box.map50:.4f}")
print(f"mAP@75    : {metrics.box.map75:.4f}")
print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")
print("=========================================")