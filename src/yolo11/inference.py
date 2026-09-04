import argparse
from pathlib import Path
import torch
import cv2
from ultralytics import YOLO

def get_best_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def parse_opt():
    parser = argparse.ArgumentParser(description="Inference with Fine-tuned YOLO11")
    
    # Model & Source
    parser.add_argument(
        "--weights", 
        type=str, 
        default="models/yolo11n_car_detect_20ep.pt",
        help="Path to trained weights (best.pt)"
    )
    parser.add_argument(
        "--source", 
        type=str, 
        default="data/car_dataset/test/images",
        help="Source path: image file, folder, video file, or '0' for webcam"
    )
    
    # Inference parameters
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default=get_best_device(), help="'mps', 'cpu', or '0'")
    
    # Output
    parser.add_argument("--project", type=str, default="runs/predict", help="Save directory")
    parser.add_argument("--name", type=str, default="car_detect_result", help="Result folder name")
    parser.add_argument("--save", action="store_true", default=True, help="Save annotated images/videos")
    parser.add_argument("--show", action="store_true", help="Display results in a window")

    return parser.parse_args()

def run_inference(opt):
    weights_path = Path(opt.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found at: {weights_path.resolve()}")

    # 1. Load model
    model = YOLO(str(weights_path))
    source = int(opt.source) if opt.source.isnumeric() else opt.source

    print(f"[Info] Running inference using device: {opt.device}")
    print(f"[Info] Loading weights: {weights_path}")
    print(f"[Info] Source: {source}")

    results = model.predict(
        source=source,
        conf=opt.conf,
        iou=opt.iou,
        imgsz=opt.imgsz,
        device=opt.device,
        project=opt.project,
        name=opt.name,
        save=opt.save,
        show=False,
        exist_ok=True,
    )

    max_demo = 5  
    print(f"\n[Demo] Display {min(max_demo, len(results))} first samples...")

    for idx, r in enumerate(results[:max_demo]):
        annotated_frame = r.plot()
        
        window_name = f"Demo Car Detection - Sample {idx + 1}"
        cv2.imshow(window_name, annotated_frame)
        print(f"-> Displaying {idx + 1}. Type any keywords to continue...")
        cv2.waitKey(0)  
        cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    opt = parse_opt()
    run_inference(opt)