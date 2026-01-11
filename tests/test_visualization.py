import pytest
from detector import TableDetector
import config
from PIL import Image

detector = TableDetector(model_name=config.MODEL_NAME)

def test_visualize_returns_image(tmp_path):
    """Test that visualize method returns an Image and saves output file."""
    sample_img = config.RAW_DATA_DIR / "invoice_table_bordered.png"
    output_file = config.PROCESSED_DATA_DIR / "invoice_table_bordered_detected.png"
    
    results = detector.predict(sample_img)
    img = detector.visualize(str(sample_img), results, output_path=str(output_file), show=False)
    assert isinstance(img, Image.Image)  
    assert output_file.exists()