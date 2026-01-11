from detector import TableDetector

if __name__ == "__main__":

    # Parameters
    modelName = "TahaDouaji/detr-doc-table-detection" # the model you use
    detectionThreshold = 0.8        # value between 0 and 1 that defines the confidence threshold
    imagePath = "images/raw/bank_doc_bordered.png"  # your image path
    outputPath = "images/processed/bank_doc_bordered_detected.png" # path to store the image with detected tables

    
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