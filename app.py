import streamlit as st
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import argparse
 

# Argument Parser
parser = argparse.ArgumentParser(description="Object Detection & Background Removal")
parser.add_argument("--image_path", type=str, help="Path to the input image")
args = parser.parse_args()

# Load Models
@st.cache_resource
def load_models():
    yolov8_model = YOLO("yolov8n.pt")
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)
    return yolov8_model, predictor

yolov8_model, predictor = load_models()

# Streamlit UI
st.title("🖼️ Object Detection & Background Removal with YOLOv8 + SAM")
st.write("Upload an image to detect, segment, and remove background.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGBA")
    image_np = np.array(image)  # Convert to numpy array
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect & Remove Background"):
        image_path = "original.png"
        image.save(image_path)

        # Run YOLOv8 for object detection
        results = yolov8_model.predict(source=image_path, conf=0.25)

        # Extract bounding box of the first detected object
        bbox = None
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                bbox = boxes.xyxy[0].tolist()  # Convert tensor to list
                break

        if bbox:
            st.write("✅ Object Detected! Running segmentation...")

            # Convert image to RGB for SAM
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            predictor.set_image(image_rgb)

            # Convert bbox to correct format for SAM
            input_box = np.array(bbox)

            # Run SAM for segmentation
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False
            )

            # Remove background
            mask = masks[0].astype(np.uint8) * 255
            alpha_channel = np.where(mask == 255, 255, 0).astype(np.uint8)
            image_np[..., 3] = alpha_channel  # Apply mask to alpha channel

            result_img = Image.fromarray(image_np)
            st.image(result_img, caption="Background Removed", use_column_width=True)

            # Save result
            buf = BytesIO()
            result_img.save(buf, format="PNG")
            st.download_button("📥 Download Image", buf.getvalue(), "output.png", "image/png")
        else:
            st.write("⚠️ No object detected. Try another image.")
