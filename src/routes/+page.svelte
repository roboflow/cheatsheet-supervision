<script lang="ts">
  import Page from "$lib/Page.svelte";
  import SplashPage from "$lib/SplashPage.svelte";
  import CodeBlock from "$lib/CodeBlock.svelte";
  import Section from "$lib/Section.svelte";
</script>

<svelte:head>
  <title>Cheatsheet • Supervision</title>
  <meta
    name="description"
    content="A cheatsheet for Roboflow Supervision, covering commonly used functions and features: model loading, annotation, object detection, segmentation, and keypoint detection."
  />
  <meta
    name="keywords"
    content="Roboflow, Supervision, computer vision, cheatsheet, SvelteKit, annotation, detection, segmentation, keypoints"
  />
  <meta name="author" content="Linas Kondrackis" />
</svelte:head>

<div class="justify-center w-full flex flex-col items-center gap-12 pb-4 px-6">
  <br />
  <SplashPage>
    <div slot="col1" class="p-6 flex flex-col gap-4">
      <!-- Describes what supervision is, and the typical process -->
      <Section header="Basic Principles">
        <div class="text-sm px-2 pb-2">
          <p>
            Supervision simplifies the process of working with vision models. It offers connectors
            to popular model libraries, a plethora of visualizers (annotators), powerful
            post-processing features and an easy learning curve.
          </p>

          <ul
            class="w-full flex flex-col sm:flex-row mt-2 font-semibold !list-none gap-3 sm:flex-wrap mt-4 sm:justify-center sm:-ml-4"
          >
            <li class="flex gap-2">
              <span class="text-[#8622FF] sm:hidden">»</span><span>Load image or video</span>
            </li>
            <li class="flex gap-2">
              <span class="text-[#8622FF]">»</span>
              <span>Load the model</span>
            </li>
            <li class="flex gap-2">
              <span class="text-[#8315F9]">»</span>
              <span>Run the model</span>
            </li>
            <li class="flex gap-2">
              <span class="text-[#8315F9]">»</span>
              <span>Annotate</span>
            </li>
          </ul>
        </div>
      </Section>

      <div class="mt-2" />
      <Section header="Quickstart">
        <CodeBlock bash code={`pip install supervision inference -q`} />
        <CodeBlock
          bash
          code={`
wget https://media.roboflow.com/notebooks/examples/dog.jpeg`}
        />
        <CodeBlock
          code={`
import cv2
import supervision as sv
from inference import get_model`}
        />
        <CodeBlock
          code={`
image = cv2.imread("dog.jpeg")
model = get_model(model_id="yolov8n-640")
results = model.infer(image)[0]
detections = sv.Detections.from_inference(results)`}
        />
        <CodeBlock
          code={`
annotated_image = sv.BoundingBoxAnnotator().annotate(
    scene=image.copy(), detections=detections
)
annotated_image = sv.LabelAnnotator().annotate(
    scene=annotated_image, detections=detections
)
sv.plot_image(annotated_image)
`}
        />
      </Section>
    </div>
    <div slot="col2">
      <!-- Describes the core concepts you'll encounter when working with it -->
      <Section header="Core Concepts">
        <ul class="list-disc list-inside ml-2 flex flex-col gap-1">
          <li>
            <strong>sv.Detections</strong>: Common class for model both object detection and
            segmentation. Contains fields: <strong>xyxy</strong>, <strong>mask</strong>,
            <strong>class_id</strong>, <strong>tracker_id</strong>, <strong>data</strong>.
          </li>
          <li>
            <strong>import supervision as sv</strong>: All useful functions available in global
            scope
          </li>
          <li>
            <strong>A selection of models</strong>: Load
            <a
              href="https://inference.roboflow.com/quickstart/aliases/"
              class="underline text-[#8315F9]"
              target="_blank">popular</a
            >,
            <a
              href="https://inference.roboflow.com/quickstart/explore_models/"
              class="underline text-[#8315F9]"
              target="_blank">fine-tuned</a
            >, or
            <a
              href="https://inference.roboflow.com/quickstart/load_from_universe/"
              class="underline text-[#8315F9]"
              target="_blank">Universe</a
            > models.
          </li>
          <li>
            <strong>sv.Detections.from_X</strong>: Load from one of
            <a
              href="https://supervision.roboflow.com/latest/detection/core/"
              class="underline text-[#8315F9]"
              target="_blank">12 sources.</a
            >
          </li>
          <li>
            <strong>Annotators</strong>: Draw the detections with one of
            <a
              href="https://supervision.roboflow.com/latest/detection/annotators/"
              class="underline text-[#8315F9]"
              target="_blank">20 annotators.</a
            >
          </li>
          <li>
            <strong>More features</strong>: This sheet contains &lt; 50% of supervision's features.
            Find others
            <a
              href="https://supervision.roboflow.com/latest/"
              class="underline text-[#8315F9]"
              target="_blank">here!</a
            >
          </li>
        </ul>
      </Section>

      <div class="mt-4" />
      <Section header="Read images & Videos">
        <CodeBlock
          preface="Load a single image"
          code={`
import cv2
image = cv2.imread("dog.jpeg")`}
        />
        <CodeBlock
          preface="Iterate over video frames"
          code={`
for frame in sv.get_video_frames_generator(source_path=<VIDEO_PATH>):
    print(frame.shape)`}
        />
        <CodeBlock
          preface="Run a function over every frame, save output"
          code={`
import numpy as np

def callback(scene: np.ndarray, index: int) -> np.ndarray:
    print(f"Processing frame {index}")
    return scene;

sv.process_video(
    source_path=<SOURCE_VIDEO_PATH>,
    target_path="out.mp4",
    callback=callback)`}
        />
      </Section>
    </div>
  </SplashPage>

  <Page header="Object Detection & Segmentation">
    <div slot="col1" class="p-6 flex flex-col gap-4">
      <CodeBlock code={`import cv2\nimport supervision as sv\n\nimage = cv2.imread("dog.jpeg")`} />
      <Section header="Frequent choices: Inference, Ultralytics & Transfomers">
        <CodeBlock
          code={`
from inference import get_model

model = get_model(model_id="yolov8n-640")
results = model.infer(image)[0]
detections = sv.Detections.from_inference(results)`}
        />
        <CodeBlock
          code={`
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)`}
        />
        <CodeBlock
          code={`import torch
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
    id2label=model.config.id2label)`}
        />
      </Section>
      <div class="flex flex-row justify-center text-lg font-bold">
        <a
          href="https://supervision.roboflow.com/latest/detection/core/"
          target="_blank"
          class="underline text-blue-500">+9 more connectors</a
        >&nbsp;⚡
      </div>
    </div>
    <div slot="col2" class="p-6 flex flex-col gap-4">
      <Section header="Annotate Detection">
        <CodeBlock
          code={`bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = bounding_box_annotator.annotate(
    scene=image.copy(), detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)`}
        />
        <div class="flex flex-row justify-center text-lg font-bold">
          <a
            href="https://supervision.roboflow.com/latest/detection/annotators/"
            target="_blank"
            class="underline text-blue-500">+18 more annotators</a
          >&nbsp;🎨
        </div>
      </Section>
      <Section header="Segmentation">
        <div class="flex flex-row justify-center text-sm font-bold">
          <span>
            For <code>inference</code> and <code>ultralytics</code>, you only need to change the
            model ID:
          </span>
        </div>
        <CodeBlock
          code={`from inference import get_model

model = get_model(model_id="yolov8n-seg-640")
results = model.infer(image)[0]
detections = sv.Detections.from_inference(results)`}
        />
        <CodeBlock
          code={`from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)`}
        />
      </Section>
      <Section header="Annotate Segmentation">
        <CodeBlock
          code={`mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = mask_annotator.annotate(
    scene=image.copy(), detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)`}
        />
      </Section>
    </div>
  </Page>

  <Page header="Keypoints">
    <div slot="col1" class="p-6 flex flex-col gap-4">
      <CodeBlock code={`import cv2\nimport supervision as sv\n\nimage = cv2.imread("dog.jpeg")`} />
      <Section header="Inference">
        <CodeBlock
          code={`
from inference import get_model

model = get_model(model_id="yolov8s-pose-640")

results = model.infer(image)[0]
key_points = sv.KeyPoints.from_inference(results)
`}
        />
      </Section>
      <Section header="Ultralytics">
        <CodeBlock
          code={`
from ultralytics import YOLO

model = YOLO("yolov8s-pose.pt")

results = model(image)[0]
key_points = sv.KeyPoints.from_ultralytics(results)`}
        />
      </Section>

      <Section header="Yolo NAS">
        <CodeBlock
          code={`
import torch
import super_gradients

device = "cuda" if torch.cuda.is_available() else "cpu"
model = super_gradients.training.models.get(
    "yolo_nas_pose_s", pretrained_weights="coco_pose").to(device)

results = model.predict(image, conf=0.1)
key_points = sv.KeyPoints.from_yolo_nas(results)`}
        />
      </Section>
    </div>
    <div slot="col2" class="p-6 flex flex-col gap-4">
      <CodeBlock
        preface={`⚠️ Available in pre-release: pip install git+https://github.com/roboflow/supervision.git@develop `}
        code={`
import mediapipe as mp

image = cv2.imread("dog.jpeg")
image_height, image_width, _ = image.shape
mediapipe_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="pose_landmarker_heavy.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_poses=2)

PoseLandmarker = mp.tasks.vision.PoseLandmarker
with PoseLandmarker.create_from_options(options) as landmarker:
    pose_landmarker_result = landmarker.detect(mediapipe_image)

key_points = sv.KeyPoints.from_mediapipe(
    pose_landmarker_result, (image_width, image_height))`}
      />

      <Section header="Annotate KeyPoints">
        <CodeBlock
          code={`vertex_annotator = sv.VertexAnnotator(radius=10)
edge_annotator = sv.EdgeAnnotator(thickness=5)

annotated_frame = edge_annotator.annotate(
    scene=image.copy(),
    key_points=key_points
)
annotated_frame = vertex_annotator.annotate(
    scene=annotated_frame,
    key_points=key_points)`}
        />
        <div class="flex flex-row justify-center text-lg font-bold">
          <a
            href="https://supervision.roboflow.com/latest/detection/annotators/"
            target="_blank"
            class="underline text-blue-500">+1 more annotator</a
          >&nbsp;🎨
        </div>
      </Section>
    </div>
  </Page>

  <Page header="What can supervision do?">
    <div slot="col1" class="p-6 flex flex-col gap-4">
      <CodeBlock
        code={`
import cv2
import supervision as sv
from inference import get_model

`}
      />

      <Section header="Track Object Movement">
        <CodeBlock
          code={`
video_info = sv.VideoInfo.from_video_path(video_path=<VIDEO_PATH>)
frames_generator = sv.get_video_frames_generator(source_path=<VIDEO_PATH>)

model = get_model("yolov8s-640")
tracker = sv.ByteTrack(frame_rate=video_info.fps)
smoother = sv.DetectionsSmoother()

trace_annotator = sv.TraceAnnotator()

with sv.VideoSink(target_path="out.mp4", video_info=video_info) as sink:
    for frame in frames_generator:
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)

        annotated_frame = trace_annotator.annotate(
            frame.copy(), detections)

        sink.write_frame(frame=frame)`}
        />
      </Section>
      <Section header="Count objects crossing a LineZone">
        <CodeBlock
          code={`
frames_generator = sv.get_video_frames_generator(source_path=<VIDEO_PATH>)
model = get_model("yolov8s-640")
tracker = sv.ByteTrack()

start, end = sv.Point(x=0, y=500), sv.Point(x=200, y=1000)
line_zone = sv.LineZone(start=start, end=end)

for frame in frames_generator:
    results = model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)
    detections = tracker.update_with_detections(detections)
    crossed_in, crossed_out = line_zone.trigger(detections)
print(line_zone.in_count, line_zone.out_count)`}
        />
      </Section>
    </div>
    <div slot="col2" class="p-6 flex flex-col gap-4">
      <Section header="Detect Small Objects">
        <CodeBlock
          preface={"InferenceSlicer breaks the image into small parts and runs the model on each one"}
          code={`
import cv2
import supervision as sv
from inference import get_model

image = cv2.imread("dog.jpeg")
model = get_model("yolov8s-640")

def callback(image_slice: np.ndarray) -> sv.Detections:
    results = model.infer(image_slice)[0]
    return sv.Detections.from_inference(results)

slicer = sv.InferenceSlicer(
    callback=callback,
    overlap_filter_strategy=sv.OverlapFilter.NON_MAX_SUPPRESSION,
)

detections = slicer(image)`}
        />
      </Section>
      <Section header="Count objects inside PolygonZone">
        <CodeBlock
          code={`
frames_generator = sv.get_video_frames_generator(source_path=<VIDEO_PATH>)
model = get_model("yolov8s-640")
tracker = sv.ByteTrack()

polygon = np.array([[100, 200], [200, 100], [300, 200], [200, 300]])
polygon_zone = sv.PolygonZone(polygon=polygon)

for frame in frames_generator:
    results = model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)
    detections = tracker.update_with_detections(detections)
    is_detections_in_zone = polygon_zone.trigger(detections)
    print(polygon_zone.current_count)`}
        />
      </Section>
    </div>
  </Page>

  <Page header="What can supervision do? (continued)">
    <div slot="col1" class="p-6 flex flex-col gap-4">
      <Section header="Save Detections to CSV">
        <CodeBlock
          code={`
frames_generator = sv.get_video_frames_generator(<VIDEO_PATH>)
model = get_model("yolov8s-640")

csv_sink = sv.CSVSink("out.csv")
with csv_sink as sink:
    for frame in frames_generator:
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        sink.append(
            detections, custom_data={"<YOUR_LABEL>":"<YOUR_DATA>"})`}
        />
      </Section>

      <Section header="Save Detections to JSON">
        <CodeBlock
          code={`
frames_generator = sv.get_video_frames_generator(<VIDEO_PATH>)
model = get_model("yolov8s-640")

json_sink = sv.JSONSink("out.json")
with json_sink as sink:
    for frame in frames_generator:
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        sink.append(
            detections, custom_data={"<YOUR_LABEL>":"<YOUR_DATA>"})`}
        />
      </Section>
    </div>
    <div slot="col2" class="p-6 flex flex-col gap-4">
      <Section header="Run a fine-tuned LMM">
        <CodeBlock code={`pip install peft -q`} bash />
        <CodeBlock
          code={`
from inference.models.paligemma.paligemma import PaliGemma
from PIL import Image
import supervision as sv

image = Image.open("dog.jpeg")
prompt = "Detect the dog."

pg = PaliGemma(model_id= api_key="<ROBOFLOW_API_KEY>")
results = pg.predict(image, prompt)

detections = sv.Detections.from_lmm(
    sv.LMM.PALIGEMMA,
    results,
    resolution_wh=(1000, 1000),
    classes=["cat", "dog"]
)
`}
        />
      </Section>
      <Section header="Compute Metrics">
        <CodeBlock
          code={`
dataset = sv.DetectionDataset.from_yolo("<PATH_TO_DATASET>")

model = get_model("yolov8s-640")
def callback(image: np.ndarray) -> sv.Detections:
    results = model.infer(image)[0]
    return sv.Detections.from_inference(results)

confusion_matrix = sv.ConfusionMatrix.benchmark(
    dataset=dataset, callback=callback
)
print(confusion_matrix.matrix)
                `}
        />
      </Section>
    </div>
  </Page>

  <Page header="Utilities">
    <div slot="col1" class="p-6 flex flex-col gap-4">
      <Section header="sv.Detections Operations">
        <CodeBlock
          preface={"Empty detections. Returned by every model when nothing is detected."}
          code={`
empty_detections = sv.Detections.empty()
if empty_detections.is_empty():
    print("Nothing was detected!")`}
        />
        <CodeBlock preface={"Count detected objects"} code={`len(detections)`} />
        <CodeBlock
          preface={"Loop over detection results"}
          code={`
for xyxy, mask, confidence, class_id, tracker_id, data in detections:
    print(xyxy, mask, confidence, class_id, tracker_id, data)`}
        />
        <CodeBlock
          preface={"Filter detections by class"}
          code={`
detections = sv.Detections.from_inference(results)
detections = detections[detections.class_id == 0]`}
        />
        <CodeBlock
          preface={"Filter by class name"}
          code={`
detections = sv.Detections.from_inference(results)
detections = detections[detections.data["class_name"] == "cat"]`}
        />
        <CodeBlock
          preface={"Merge multiple sv.Detections"}
          code={`
detections1 = sv.Detections.from_inference(results1)
detections2 = sv.Detections.from_inference(results2)
merged_detections = sv.Detections.merge([detections1, detections2])`}
        />
      </Section>
      <Section header="Video Assets">
        <CodeBlock
          bash
          preface={"supervision provides a handful of videos for testing"}
          code={`
pip install "supervision[assets]" -q`}
        />
        <CodeBlock
          code={`
from supervision.assets import download_assets, VideoAssets

download_assets(VideoAssets.VEHICLES)
print(VideoAssets.VEHICLES.value)`}
        />
      </Section>
    </div>
    <div slot="col2" class="p-6 flex flex-col gap-4">
      <Section header={"Image Utilities"}>
        <CodeBlock
          preface={"Crop image"}
          code={`cropped_image = sv.crop_image(image=image, xyxy=[200, 400, 600, 800])`}
        />
        <CodeBlock
          preface={"Scale image"}
          code={`scaled_image = sv.scale_image(image=image, scale_factor=0.5)`}
        />
        <CodeBlock
          preface={"Resize image"}
          code={`
resized_image = sv.resize_image(
    image=image, resolution_wh=(1000, 1000), keep_aspect_ratio=True)`}
        />
        <CodeBlock
          preface={"Letterbox image (resize + pad)"}
          code={`
letterboxed_image = sv.letterbox_image(
    image=image, resolution_wh=(1000, 1000))`}
        />
        <CodeBlock
          preface={"Overlay image"}
          code={`
overlay = np.zeros((400, 400, 3), dtype=np.uint8)
resulting_image = sv.overlay_image(
    image=image, overlay=overlay, anchor=(200, 400)`}
        />
      </Section>

      <Section header="for Google Colab">
        <CodeBlock
          preface={"Install custom branch of supervision"}
          bash
          code={`
pip install git+https://github.com/YourName/supervision.git@your-branch`}
        />
        <CodeBlock
          preface={"Display image in Colab by converting to PIL"}
          bash
          code={`
sv.cv2_to_pillow(frame)`}
        />
        <CodeBlock
          preface={"Display image in Colab by plotting with matplotlib"}
          bash
          code={`
%matplotlib inline\nsv.plot_image(frame)`}
        />
      </Section>
    </div>
  </Page>
</div>
<footer class="w-full max-w-[1123px] mx-auto text-xs py-2 text-gray-400">
  &copy; 2024 Roboflow, Inc. All rights reserved.
</footer>

<style>
  ul {
    @apply text-xs list-disc;
  }
  code {
    color: #8315f9;
  }
</style>
