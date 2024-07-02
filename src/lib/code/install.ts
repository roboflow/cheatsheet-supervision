export default [
    `pip install supervision -q`,
    `import supervision as sv`,

    `\
pip install inference -q

# Alternatively, to run on a GPU
pip install inference-gpu -q`,
`from inference import get_model`,
    
`pip install ultralytics -q`,
`from ultralytics import YOLO`,

`import transformers`,
`from transformers import DetrImageProcessor, DetrForObjectDetection`
];