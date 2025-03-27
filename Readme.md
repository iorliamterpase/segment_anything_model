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



#**Run inference on images**

python app.py --image_path original.png
