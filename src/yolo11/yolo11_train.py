import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def get_best_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def parse_opt():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11 for Car Detection")
    
    # Paths
    parser.add_argument("--weights", type=str, default="", help="Initial weights path (leave empty for default)")
    parser.add_argument("--data", type=str, default="", help="Path to car_dataset.yaml (leave empty for default)")
    parser.add_argument("--project", type=str, default="./models", help="Directory to save checkpoints")
    parser.add_argument("--name", type=str, default="yolo11n_car_detect", help="Experiment name")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=20, help="Total training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (pixels)")
    parser.add_argument("--device", type=str, default=get_best_device(),help="CUDA device index, 'mps' (Mac Apple Silicon), or 'cpu'")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument("--freeze", type=int, default=None, help="Number of backbone layers to freeze")
    return parser.parse_args()


def get_default_paths():
    curr_dir = Path(__file__).resolve().parent
    project_root = curr_dir.parent.parent
    model_path = project_root / "models" / "yolo11n.pt"
    data_path = project_root / "src" / "yolo11" / "car_dataset.yaml"
    return str(model_path), str(data_path)

def train(opt):
    default_model, default_data = get_default_paths()
    weights_path = opt.weights if opt.weights else default_model
    data_path = opt.data if opt.data else default_data

    print(f"[Debug] Check weights path: {weights_path}")
    print(f"[Debug] Check dataset path: {data_path}")

    model = YOLO(weights_path)

    # Train
    model.train(
        data=data_path,
        epochs=opt.epochs,
        imgsz=opt.imgsz,
        batch=opt.batch,
        device=opt.device,
        workers=opt.workers,

        # --- CHECKPOINT & EARLY STOPPING ---
        save=True,
        save_period=-1,
        patience=opt.patience,
        project=opt.project,
        name=opt.name,
        exist_ok=True,

        # --- OPTIMIZER & LEARNING RATE ---
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3.0,
        weight_decay=0.0005,

        # --- FINETUNE ---
        pretrained=True,
        freeze=opt.freeze,

        # --- DATA AUGMENTATION ---
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        fliplr=0.5,
        close_mosaic=10,

        # --- VALIDATION & LOGGING ---
        val=True,
        plots=True,
        verbose=True,
    )

if __name__ == "__main__":
    opt = parse_opt()
    train(opt)