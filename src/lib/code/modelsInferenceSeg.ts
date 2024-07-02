export default [
    `\
from inference import get_model

model = get_model(model_id="yolov8n-seg-640")
results = model.infer(image)[0]
detections = sv.Detections.from_inference(results)`
];