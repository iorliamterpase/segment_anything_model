import argparse
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import torch
from PIL import Image
from io import BytesIO
import io
import logging
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Argument parser setup
parser = argparse.ArgumentParser(description="Start FastAPI app with configurable model settings.")
parser.add_argument("--yolo_model_path", type=str, default="yolov8n.pt", help="Path to YOLOv8 model")
parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth", help="Path to SAM checkpoint")
parser.add_argument("--model_type", type=str, default="vit_h", help="Type of SAM model (e.g., vit_h, vit_b)")
parser.add_argument("--conf_threshold", type=float, default=0.25, help="YOLOv8 confidence threshold")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the FastAPI server on")
parser.add_argument("--port", type=int, default=8000, help="Port to run the FastAPI server on")

# Parse args but keep the remainder for uvicorn
args, _ = parser.parse_known_args()

# Initialize FastAPI app
app = FastAPI()

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI SAM & YOLOv8 Service!"}

# Load Models
def load_models():
    try:
        logger.info(f"Loading YOLOv8 model from {args.yolo_model_path}...")
        yolov8_model = YOLO(args.yolo_model_path)
        
        logger.info(f"Loading SAM model ({args.model_type}) from {args.sam_checkpoint}...")
        sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
        predictor = SamPredictor(sam)
        
        logger.info("Models loaded successfully")
        return yolov8_model, predictor
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

try:
    yolov8_model, predictor = load_models()
except Exception as e:
    logger.error(f"Failed to initialize models: {str(e)}")
    

@app.post("/detect")
async def detect_and_segment(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}, content-type: {file.content_type}")
        
        # Read and convert image
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")
        
        image = Image.open(io.BytesIO(contents)).convert("RGB")  # Changed to RGB
        image_np = np.array(image)
        logger.info(f"Image shape: {image_np.shape}")

        # Run YOLOv8 for object detection
        logger.info(f"Running YOLOv8 detection with confidence threshold {args.conf_threshold}...")
        results = yolov8_model.predict(source=image_np, conf=args.conf_threshold)
        
        # Log detection results
        all_boxes = []
        for result in results:
            boxes = result.boxes
            logger.info(f"Detected {len(boxes)} objects")
            if len(boxes) > 0:
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].tolist()
                    conf = boxes.conf[i].item()
                    cls = int(boxes.cls[i].item())
                    logger.info(f"Box {i}: {box}, confidence: {conf:.2f}, class: {cls}")
                    all_boxes.append((box, conf, cls))
        
        # Extract bounding box of the first detected object
        bbox = None
        if all_boxes:
            # Sort by confidence and take highest
            all_boxes.sort(key=lambda x: x[1], reverse=True)
            bbox = all_boxes[0][0]
            logger.info(f"Selected bbox with highest confidence: {bbox}")
        
        if not bbox:
            logger.warning("No objects detected in the image")
            return JSONResponse(
                content={"error": "No object detected. Try a different image or check model configuration."}, 
                status_code=400
            )

        # Convert image for SAM
        logger.info("Running SAM segmentation...")
        image_rgb = image_np.copy()
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
        logger.info(f"Mask shape: {masks[0].shape}")

        # Prepare RGBA image for transparency
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:  # RGB image
            rgba_image = np.dstack((image_np, np.ones(image_np.shape[:2], dtype=np.uint8) * 255))
        else:  # Already RGBA
            rgba_image = image_np

        # Apply mask to alpha channel
        mask = masks[0].astype(np.uint8) * 255
        rgba_image[..., 3] = np.where(mask == 255, 255, 0).astype(np.uint8)

        logger.info("Creating final output image...")
        result_img = Image.fromarray(rgba_image)
        buf = BytesIO()
        result_img.save(buf, format="PNG")
        buf.seek(0)  # Rewind the buffer
        logger.info("Processing completed successfully")
        
        return JSONResponse(
            content={"message": "Background removed successfully", "image": buf.getvalue().hex()}
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"error": f"Failed to process image: {str(e)}"}, 
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting FastAPI server at {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)