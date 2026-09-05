import cv2
import glob
import math
import time
import argparse
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
import supervision as sv

@dataclass
class Config:
    model_path: str = "models/yolo26s.pt"
    input_video: str = "data/cars_on_highway.mp4"
    output_video: str = "outputs/car_counter_yolo26s.mp4"

class VehicleCounterSystem:
    def __init__(self, config: Config):
        self.config = config
        self.model = YOLO(self.config.model_path)
        self.input_path = self.config.input_video
        self.output_path = self.config.output_video
        
        # Đọc thông tin video metadata
        self.video_info = sv.VideoInfo.from_video_path(self.input_path)
        
        # Tracker & Smoother
        self.tracker = sv.ByteTrack(frame_rate=self.video_info.fps)
        self.smoother = sv.DetectionsSmoother()
        
        # Lọc class phương tiện giao thông
        vehicle_classes = {'car', 'motorcycle', 'bus', 'truck'}
        self.selected_classes = [
            cls_id for cls_id, class_name in self.model.names.items() 
            if class_name in vehicle_classes
        ]
        
        # Thiết lập vùng đếm Polygon
        self.zone_points = self._init_zones()
        self.zones = [sv.PolygonZone(points) for points in self.zone_points]
        
        # Bộ đếm ID duy nhất (O(1) lookup)
        self.total_counts = set()
        self.counts_up = set()    # Zone 1
        self.counts_down = set()  # Zone 0
        
        self._setup_annotators()

    def _init_zones(self):
        """Scale đa giác theo tỷ lệ khung hình video thực tế (hệ số 0.66)"""
        arr1 = np.array([[761, 642], [1073, 642], [1070, 732], [968, 776], [872, 1038], [97, 1049]], dtype=np.int32)
        arr2 = np.array([[1105, 639], [1402, 645], [1920, 959], [1920, 1080], [930, 1073], [991, 811], [1105, 755]], dtype=np.int32)
        return [np.floor(arr * 0.66).astype(np.int32) for arr in [arr1, arr2]]

    def _setup_annotators(self):
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=self.video_info.resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=self.video_info.resolution_wh)
        
        self.box_annotator = sv.RoundBoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale, 
            text_thickness=thickness, 
            text_position=sv.Position.TOP_CENTER, 
            color_lookup=sv.ColorLookup.TRACK
        )
        self.colors = sv.ColorPalette.from_hex(['#ef260e', '#07f921'])
        self.zone_annotators = [
            sv.PolygonZoneAnnotator(zone=zone, color=self.colors.by_idx(idx), thickness=2)
            for idx, zone in enumerate(self.zones)
        ]

    def _draw_overlay(self, frame, points, color, alpha=0.25):
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _process_detections(self, frame, detections):
        # 1. Lọc xe cộ
        detections = detections[np.isin(detections.class_id, self.selected_classes)]

        # 2. Vẽ nền mờ cho 2 vùng đa giác
        zone_colors_bgr = [(88, 117, 234), (11, 244, 113)]
        for points, color in zip(self.zone_points, zone_colors_bgr):
            self._draw_overlay(frame, points, color=color, alpha=0.25)

        # 3. Vẽ toàn bộ bounding box và label của các xe đang track
        if len(detections) > 0 and detections.tracker_id is not None:
            labels = [
                f"{self.model.names[c]} #{t}" 
                for c, t in zip(detections.class_id, detections.tracker_id)
            ]
            frame = self.box_annotator.annotate(scene=frame, detections=detections)
            frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)

            # 4. Logic đếm: Lấy điểm đáy bánh xe (BOTTOM_CENTER)
            anchors = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            for track_id, bottom_center in zip(detections.tracker_id, anchors):
                cx, cy = map(int, bottom_center)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), cv2.FILLED)
                
                # Zone 0: Xe chạy chiều xuống
                if cv2.pointPolygonTest(self.zone_points[0], (cx, cy), False) >= 0:
                    self.total_counts.add(track_id)
                    self.counts_down.add(track_id)
                
                # Zone 1: Xe chạy chiều lên
                if cv2.pointPolygonTest(self.zone_points[1], (cx, cy), False) >= 0:
                    self.total_counts.add(track_id)
                    self.counts_up.add(track_id)

        # 5. Vẽ viền đa giác và số đếm HUD
        for zone_annotator in self.zone_annotators:
            zone_annotator.annotate(scene=frame)

        self._draw_counters(frame)
        return frame

    def _draw_counters(self, frame):
        counter_labels = [
            f"TOTAL: {len(self.total_counts)}", 
            f"UP   : {len(self.counts_up)}", 
            f"DOWN : {len(self.counts_down)}"
        ]
        count_colors = [(0, 0, 0), (6, 104, 2), (0, 0, 255)]
        
        # Bảng HUD góc trái trên
        cv2.rectangle(frame, (0, 0), (280, 140), (255, 255, 255), cv2.FILLED)
        for i, (label, color) in enumerate(zip(counter_labels, count_colors)):
            cv2.putText(frame, label, (15, 40 + i * 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    def run(self):
        cap = cv2.VideoCapture(self.input_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.output_path, fourcc, self.video_info.fps, 
            (self.video_info.width, self.video_info.height)
        )

        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {self.input_path}")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. Phát hiện đối tượng (Detection)
                results = self.model(frame, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                
                # 2. Định danh tracking & Làm mượt rung lắc
                detections = self.tracker.update_with_detections(detections)
                detections = self.smoother.update_with_detections(detections)

                # 3. Đếm & Vẽ
                frame = self._process_detections(frame, detections)

                out.write(frame)
                cv2.imshow("Vehicle Counter System", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Counting System using YOLO & Supervision")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="models/yolo26s.pt", 
        help="Path to YOLO weights file (.pt)"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/cars_on_highway.mp4", 
        help="Path to input video file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="outputs/car_counter_yolo26s.mp4", 
        help="Path to output saved video file"
    )
    
    args = parser.parse_args()
    cfg = Config(
        model_path=args.model,
        input_video=args.input,
        output_video=args.output
    )
    app = VehicleCounterSystem(cfg)
    app.run()