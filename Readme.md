# **Clone this repo and cd into it**


# **Create a Virtual Environment.**

python -m venv venv

venv\scripts\activate

python -m pip install --upgrade pip


# **Install requirements.txt**


pip install \
'git+https://github.com/facebookresearch/segment-anything.git'
pip install -q YOLO

# **DOWNLOAD MODELS HERE.**

'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'

pip install git+https://github.com/facebookresearch/segment-anything.git

 yolov8n.pt

#** run server.py which is the FastAPI script locally.**

pip install fastapi uvicorn

uvicorn server:app --host 0.0.0.0 --port 8000

#**When you run the scripts open the URL below on your terminal different from the terminal you ran your streamlit app.py or you can call the API on your streamlit app for testing.

Open your browser and go to:
 http://127.0.0.1:8000

#**Run inference on images**

python app.py --image_path original.png

#** run server.py which is the FastAPI script locally.**

pip install fastapi uvicorn

uvicorn server:app --host 0.0.0.0 --port 8000

#**When you run the scripts open the URL below on your terminal different from the terminal you ran your streamlit app.py

