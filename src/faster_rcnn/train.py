"""
Sh:
    bash script/run_faster_rcnn.sh
"""
import argparse
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights, 
    FastRCNNPredictor
)
from dataclasses import dataclass
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.faster_rcnn.face_detection_dataset import FaceDetectionDataset

@dataclass
class Config:
    num_epochs: int = 30
    image_size: int = 224
    batch_size: int = 16
    lr: float = 0.0001
    weight_decay: float = 0.0005
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def collate_fn(batch):
    all_images = []
    all_labels = []
    for image, label in batch:
        all_images.append(image)
        all_labels.append(label)
    return all_images, all_labels

class MobileNetDetection(nn.Module):
    def __init__(self, num_classes = 1):
        super().__init__()
        self.model = model
        
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
            trainable_backbone_layers=6,
        )

        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_channels=model.roi_heads.box_predictor.cls_score.in_features,
            num_classes=2,  # 1 face + 1 background = 2
        )
        
    def forward(self, images, targets = None):
        return self.model(images, targets)

def train():
    train_transform = Compose([
        Resize((config.image_size, config.image_size)),
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])

    val_transform = Compose([
        Resize((config.image_size, config.image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])

    test_transform = Compose([
        Resize((config.image_size, config.image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])

    train_data = FaceDetectionDataset(
        root='data/face_dataset',
        mode='train',
        transform = train_transform,
    )

    valid_data = FaceDetectionDataset(
        root='data/face_dataset',
        mode = "valid",
        transform = val_transform,
    )

    test_data = FaceDetectionDataset(
        root='data/face_dataset',
        mode = "test",
        transform = test_transform,
    )

    train_params = {
        "batch_size": config.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": 2,
        "collate_fn": collate_fn
    }

    val_params = {
        "batch_size": config.batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": 2,
        "collate_fn": collate_fn
    }

    test_params = {
        "batch_size": config.batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": 2,
        "collate_fn": collate_fn
    }

    train_dataloader = DataLoader(
        train_data,
        **train_params,
    )

    val_dataloader = DataLoader(
        valid_data,
        **val_params,
    )

    test_dataloader = DataLoader(
        test_data,
        **test_params,
    )

    optimizer = optim.Adam(
        model.parameters(), 
        lr = config.lr, 
        weight_decay = config.weight_decay
    )

    best_map = -1
    history_train_loss = []
    history_val_loss = []  
    history_map = []
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = []
        train_progress_bar = tqdm(train_dataloader, colour='cyan')
        for iter, (images, labels) in enumerate(train_progress_bar):
            images = [image.to(device) for image in images]
            labels = [
                {
                    "boxes": label["boxes"].to(device),
                    "labels": label["labels"].to(device),
                }
                for label in labels
            ]
            loss_components = 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Faster R-CNN model with MobileNet backbone.")
    parser.add_argument("--image_size", type=int, default=224, help="Size to which images will be resized (default: 224).")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes for detection (default: 4).")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate (default: 0.0001).")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay (default: 0.0005).")
    parser.add_argument("--num_epochs", type=int, default=30, help="epochs for training")
    args = parser.parse_args()
    config = Config(
        image_size=args.image_size,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        num_epochs = args.num_epochs,
    )
    
    model = MobileNetDetection(num_classes = config.num_classes)
    train(model, train_dataloader, val_dataloader, test_dataloader)