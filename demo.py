import config
from detector import TableDetector


if __name__ == "__main__":

    # Parameters you may need, they can be modified as you want
    modelName = config.MODEL_NAME                    # "TahaDouaji/detr-doc-table-detection"  the model you use
    detectionThreshold = config.DETECTION_THRESHOLD  # value between 0 and 1 that defines the confidence threshold
    imagePath = config.RAW_DATA_DIR / "bank_doc_bordered.png"  # your image path
    outputPath = config.PROCESSED_DATA_DIR / "bank_doc_bordered_detected.png" # path where you want to save the image with detected tables

    
    # 1. Initialize the detector
    detector = TableDetector(
        model_name = modelName,
        detection_threshold = detectionThreshold
    )

    # 2. Predict
    results = detector.predict(image_path = imagePath)
    print(" ==== Detection Results ==== ")
    print(results)
    
    # 3. Visualize
    # -  You can save the image by giving the output_path argument
    detector.visualize(
        image_path = imagePath,
        results = results,
        output_path = outputPath,
        show=True
    )