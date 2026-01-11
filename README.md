# DataleonPythonTest

---

This project provides a Python class for detecting tables in document images (invoices, bank statements) using a pre-trained HuggingFace DETR-based object detection model. It supports table prediction and visualization with bounding boxes, and includes pytest tests for various scenarios.

## Project Structure

---

```bash
DataleonPythonTest/
├── images		# Used/Saved images for tests or demo
│ ├── processed
│ └── raw
├── detector/ 	# Core TableDetector class
│ ├── __init__.py
│ └── tableDetector.py
├── tests/ 		# Pytest test cases
│ ├── test_errors.py
│ ├── test_prediction.py
│ └── test_visualization.py
├── config.py		# File conainting parameters
├── demo.py		# Demo script with visualization
├── requirements.txt # Project dependencies
└── README.md
```

## Installation

---

1. Clone this repository:
```bash
git clone RunRanRua/DataleonPythonTest
cd DataleonPythonTest
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

---

### How to initialize

`TableDetector` class accepts 3 parameters:

```python
model_name:str				# the model from huggingFace
detection_threshold:float = 0.8		# value between 0 and 1 that defines the threshold for displaying detected objects
device:str | None = None	# device, optional
```

### Methods

There are 2 main methods in `TableDetector` class:

```python
def predict(self, image_path: str) -> dict:
    """
    Predict tables in the given document image.

    Args:
        image_path (str): Path to the input image file. The image should be a valid document image (e.g., invoice or bank statement).

    Returns:
        dict: A dictionary containing detection results:
            - "boxes": list of bounding boxes [x_min, y_min, x_max, y_max]
            - "labels": list of detected object labels (integer)
            - "scores": list of confidence scores (float)
    """

        
def visualize(
    self, 
    image_path: str, 
    results: dict, 
    output_path: str = None, 
    show: bool = True
) -> Image.Image:
    """
    Draw detected table bounding boxes on the image.

    Args:
        image_path (str): Path to the input image file.
        results (dict): Detection results from `predict()` containing "boxes", "labels", and "scores".
        output_path (str, optional): Path to save the visualized image. If None, the image is not saved.
        show (bool, optional): Whether to display the image after drawing boxes. Default is True.

    Returns:
        PIL.Image.Image: Image object with bounding boxes drawn.
    """

```

### Example

```python
from detector import TableDetector

# Parameters
modelName = "TahaDouaji/detr-doc-table-detection"  # the model you use
detectionThreshold = 0.8  # value between 0 and 1 that defines the confidence threshold
imagePath = "YOUR_IMAGE_PATH/sample_img.png"
outputPath = "SAVED_IMAGE_PATH/sample_img_saved.png" # path where you save the image after find out tables

# 1. Initialize the detector
detector = TableDetector(
    model_name = modelName,
    detection_threshold = detectionThreshold
)

# 2. Predict
results = detector.predict(image_path = imagePath)
    
# 3. Visualize
detector.visualize(
	image_path = imagePath,
	results = results,
	output_path = outputPath,
	show=True
)
```

You may also check `demo.py` to see how to use