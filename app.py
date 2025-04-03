import streamlit as st
import requests
import io
from PIL import Image
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI server URL - Note the endpoint URL needs to be complete
FASTAPI_URL = "http://127.0.0.1:8000/"

st.title("🖼️ Object Detection & Background Removal with YOLOv8 + SAM")
st.write("Upload an image to detect, segment, and remove background.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image_path = "original.png"
    image = Image.open(uploaded_file).convert("RGB")  # Change to RGB to match FastAPI
    st.image(image, caption="Uploaded Image", use_container_width=True)  # ✅ Updated here

    if st.button("🔍 Detect & Remove Background"):
        try:
            with st.spinner("Processing..."):
                # Prepare the image for upload
                image_bytes = io.BytesIO()
                image.save(image_bytes, format="PNG")
                image_bytes.seek(0)
                
                # Log request details
                logger.info(f"Sending request to {FASTAPI_URL}detect")
                
                # Send the request to the FastAPI endpoint
                files = {"file": ("image.png", image_bytes, "image/png")}
                response = requests.post(FASTAPI_URL + "detect", files=files)
                
                # Log response
                logger.info(f"Response status: {response.status_code}")
                if response.status_code != 200:
                    logger.error(f"Error response: {response.text}")
                
                # Process the response
                if response.status_code == 200:
                    result_data = response.json()
                    result_image = Image.open(io.BytesIO(bytes.fromhex(result_data["image"])))
                    st.image(result_image, caption="Background Removed", use_container_width=True)  # ✅ Updated here
                    st.success("✅ Background removed successfully!")
                else:
                    error_msg = "Error processing image"
                    if response.status_code == 400:
                        try:
                            error_data = response.json()
                            if "error" in error_data:
                                error_msg = error_data["error"]
                        except:
                            pass
                    st.error(f"⚠️ {error_msg}")
                    
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            logger.error(f"Exception: {str(e)}", exc_info=True)
