import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

class SoccerTracker:
    def __init__(self):
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_iou_distance=0.7)
        
    def process_frame(self, frame):
        results = self.yolo_model(frame)
        detections = results.xyxy[0]
        player_detections = [det[:4].cpu().numpy() for det in detections if int(det[5]) == 0]
        deepsort_detections = [list(det[:4]) + [det[4].item()] for det in player_detections]
        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
        return tracks
