export default [
    `\
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

image = Image.open("dog.jpeg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

target_size = torch.tensor([[image.size[1], image.size[0]]])
results = processor.post_process_object_detection(
    outputs=outputs, target_sizes=target_size)[0]

detections = sv.Detections.from_transformers(
    transformers_results=results,
    id2label=model.config.id2label)`
];