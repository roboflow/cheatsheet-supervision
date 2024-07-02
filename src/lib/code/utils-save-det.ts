export default [
    `pip install supervision inference`,

    `pip install "supervision[assets]"`,
    
    `\
import supervision as sv
from inference import get_model

model = get_model(model_id="yolov8n-640")
results = model.infer("dog.jpeg")[0]
detections = sv.Detections.from_inference(results)`,

    `\
annotated_image = sv.BoundingBoxAnnotator().annotate(
    scene=image.copy(), detections=detections
)
annotated_image = sv.LabelAnnotator().annotate(
    scene=annotated_image, detections=detections
)
sv.plot_image(annotated_image)`,
];