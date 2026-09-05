import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights, 
    FastRCNNPredictor
)
from torchvision.transforms import Compose, Resize, ColorJitter, ToTensor, Normalize
from dataclasses import dataclass
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.faster_rcnn.face_detection_dataset import FaceDetectionDataset


@dataclass
class Config:
    save_path: str = "./models/checkpoints"
    num_epochs: int = 1
    image_size: int = 224
    batch_size: int = 4
    lr: float = 0.0001
    weight_decay: float = 0.0005
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def collate_fn(batch):
    all_images = []
    all_labels = []
    for image, label in batch:
        all_images.append(image)
        all_labels.append(label)
    return all_images, all_labels

class MobileNetDetection(nn.Module):
    def __new__(cls, num_classes=2):
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
            trainable_backbone_layers=6,
        )
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_channels=model.roi_heads.box_predictor.cls_score.in_features,
            num_classes=num_classes,  
        )
        return model

def train(model, train_dataloader, val_dataloader, test_dataloader, config):
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )
    best_map = -1e6
    history_loss_train = []
    history_loss_val = []
    history_map = []

    for epoch in range(config.num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = []
        train_progress_bar = tqdm(train_dataloader, colour='cyan', desc=f"Epoch {epoch + 1}/{config.num_epochs} [Train]")
        
        for images, labels in train_progress_bar:
            images = [image.to(config.device) for image in images]
            labels = [{"boxes": label["boxes"].to(config.device), "labels": label["labels"].to(config.device)} for label in labels]
            
            loss_components = model(images, labels)
            losses = sum([loss for loss in loss_components.values()])
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_loss.append(losses.item())
            avg_train_loss = np.mean(train_loss)
            train_progress_bar.set_postfix(loss=f"{avg_train_loss:0.4f}")
        
        history_loss_train.append(avg_train_loss)

        # loss for eval
        model.train()
        with torch.no_grad():
            for images, labels in val_dataloader:
                loss_components = model(images, labels)
                losses = sum([loss for loss in loss_components.values()])
                
                optimizer.zero_grad()
                val_losses.backward()
                optimizer.step()
                val_losses.append(losses.item())
        
        avg_val_loss = np.mean(val_losses)
        history_val_loss.append(avg_val_loss)

        # --- Validation & mAP Phase ---
        model.eval()
        metric = MeanAveragePrecision(iou_type='bbox')
        
        val_progress_bar = tqdm(val_dataloader, colour='magenta', desc=f"Epoch {epoch + 1}/{config.num_epochs} [Val]")
        with torch.no_grad():
            for images, labels in val_progress_bar:
                images = [image.to(config.device) for image in images]
                
                # predict to compute mAP
                predictions = model(images)
                metric.update(predictions, labels)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        
        # plot
        map_metrics = metric.compute()
        current_map = map_metrics['map'].item()
        history_map.append(current_map)

        # get the best checkpoint
        os.makedirs(config.save_path, exist_ok=True)
        if current_map > best_map:
            best_map = current_map
            torch.save(checkpoint, os.path.join(config.save_path, "best.pt"))
        
        # resumbale training for systematic crashes
        torch.save(checkpoint, os.path.join(config.save_path, "last.pt"))
        print(f"Epoch {epoch + 1} Summary -> Train Loss: {history_loss_train[-1]:.4f} | Val Loss: {avg_val_loss:.4f} | mAP: {current_map:.4f}")

    return history_loss_train, history_loss_val, history_map

def plot(history_loss_train, history_loss_val, history_map):
    epochs = range(1, len(history_loss_train) + 1)
    
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_loss_train, label='Train Loss', color='blue', marker='o')
    plt.plot(epochs, history_loss_val, label='Val Loss', color='orange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # mAP
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_map, label='mAP (Validation)', color='green', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.title('Mean Average Precision (mAP)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    #plt.savefig('training_metrics.png')
    #print("Biểu đồ đã được lưu vào file 'training_metrics.png'")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Faster R-CNN model with MobileNet backbone.")
    parser.add_argument("--root", type=str, default="data/face_dataset", help="the root of datasets")
    parser.add_argument("--save_path", type=str, default="models", help="save at path")
    parser.add_argument("--image_size", type=int, default=224, help="Size to which images will be resized.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for detection.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay.")
    parser.add_argument("--num_epochs", type=int, default=30, help="Epochs for training")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cpu, cuda, or mps")
    args = parser.parse_args()
    
    config = Config(
        save_path=args.save_path,
        image_size=args.image_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        device=torch.device(args.device)
    )
    print(f"[Debug] GPU-MPS: {config.device}")
    train_transform = Compose([
        Resize((config.image_size, config.image_size)),
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = Compose([
        Resize((config.image_size, config.image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = Compose([
        Resize((config.image_size, config.image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data = FaceDetectionDataset(root=args.root, mode='train', transform=train_transform)
    valid_data = FaceDetectionDataset(root=args.root, mode="valid", transform=val_transform)
    test_data = FaceDetectionDataset(root=args.root, mode="test", transform=test_transform)

    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=0, collate_fn=collate_fn)
    val_dataloader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=False, drop_last=False, num_workers=0, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, drop_last=False, num_workers=0, collate_fn=collate_fn)

    model = MobileNetDetection(num_classes = config.num_classes).to(config.device)
    
    history_loss_train, history_loss_val, history_map = train(model, train_dataloader, val_dataloader, test_dataloader, config)
    
    plot(history_loss_train, history_loss_val, history_map)