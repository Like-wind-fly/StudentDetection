import os
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np

# --- Configuration ---
MODEL_PATH = "D:\\code\\yolo11\stage3\\train\\weights\\best.pt"  # Replace with your model path
VIDEO_PATH = "D:\code\dataset\people\考试站立\考试站立\展厅左前角\展厅左前角_20240904103130_20240904103420.mp4"      # Replace with your video path
OUTPUT_PATH = "output_video.mp4"          # Replace with your output video path
CROP_DIR = "cropped_objects"               # Directory to save cropped images


# --- Inference Knowledge Base (Simple Example) ---
inference_knowledge = {
    "person": "Detected a person.",
}

# --- Functions ---

def crop_with_aspect_ratio(image, box, aspect_ratio=(1, 2)):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    new_width, new_height = width, int(width / aspect_ratio[0] * aspect_ratio[1])
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    new_x1 = int(max(0, center_x - new_width / 2))
    new_y1 = int(max(0, center_y - new_height / 2))
    new_x2 = int(min(image.shape[1], center_x + new_width / 2))
    new_y2 = int(min(image.shape[0], center_y + new_height / 2))
    crop_obj = image[new_y1:new_y2, new_x1:new_x2]
    return crop_obj

def perform_inference(cls_id, model_names):
    """Performs simple inference based on class ID."""
    cls_name = model_names[cls_id]
    return inference_knowledge.get(cls_name, "Unknown object detected.")

# --- Main Program ---

# Load YOLO model
model = YOLO(MODEL_PATH)
names = model.names

# Open video capture
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Error opening video file: {VIDEO_PATH}")

# Get video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Adjust codec if needed
video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

# Create directory for cropped images
os.makedirs(CROP_DIR, exist_ok=True)

idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, show=False, show_labels=True, show_conf=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    annotator = Annotator(frame, line_width=2, example=names)

    for box, cls_id in zip(boxes, clss):
        idx += 1
        inference_result = perform_inference(int(cls_id), names)
        # annotator.box_label(box, color=colors(int(cls_id), True), label=f"{names[int(cls_id)]}: {inference_result}")

        crop_obj = crop_with_aspect_ratio(frame, box)
        if crop_obj.size > 0:
            cv2.imwrite(os.path.join(CROP_DIR, f"{idx}.png"), crop_obj)

    cv2.imshow("YOLO Detection", annotator.result())
    video_writer.write(annotator.result())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Video processing complete. Output saved to: {OUTPUT_PATH}")