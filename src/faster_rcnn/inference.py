import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from src.faster_rcnn.train import MobileNetDetection

def deploy(args):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = MobileNetDetection(num_classes = 2).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()
    # preprocessing like training
    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = image.astype(np.float32) / 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    image = [torch.from_numpy(np.transpose(image, (2, 0, 1))).to(device).float()]
    
    with torch.no_grad():
        # prediction
        predictions = model(image)
    # loop to get metrics in prediction including boexes, scores, and labels
    for box, score, label in zip(predictions[0]["boxes"], predictions[0]["scores"], predictions[0]["labels"]):
        if score > args.conf_threshold:
            xmin, ymin, xmax, ymax = box
            xmin = int(xmin/args.image_size * w)
            ymin = int(ymin/args.image_size * h)
            xmax = int(xmax/args.image_size * w)
            ymax = int(ymax/args.image_size * h)
            cv2.rectangle(
                ori_image, (xmin, ymin), (xmax, ymax), (0,255,0), 1
            )
            cv2.putText(ori_image, f"face {score:.2f}", (xmin, ymin - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 1)
    
    #cv2.imwrite("prediction.jpg", ori_image)
    cv2.imshow("Face Detection Prediction", ori_image)
    print("type any to quit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference a Faster R-CNN model with MobileNet backbone.")
    parser.add_argument("--image_path", type=str, default="data/person.jpg")
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=str, default="models/faster_mobilenet_facedetect", help="load checkpoint")
    parser.add_argument("--image_size", type=int, default=224, help="Size to which images will be resized.")
    parser.add_argument("--device", type=str, default="mps", help="Device to use: cpu, cuda, or mps")
    args = parser.parse_args()
    deploy(args)
