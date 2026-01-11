from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = ROOT_DIR / "images" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "images" / "processed"

MODEL_NAME = "TahaDouaji/detr-doc-table-detection"  # the model we want to use from Hugging Face
DETECTION_THRESHOLD = 0.8        # threshold for displaying detected objects