# model_config.py
from pathlib import Path
import json

MODEL_PATH = Path("insect-detect/models/yolov5n_320_openvino_2022.1_4shave.blob")
CONFIG_PATH = Path("insect-detect/models/json/yolov5_v7_320.json")

def load_model_config():
    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = json.load(f)
    return MODEL_PATH, config
