export default [
    `\
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)`
];