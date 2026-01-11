import pytest
from detector import TableDetector
import config

detector = TableDetector(model_name=config.MODEL_NAME)

def test_file_not_found():
    """Test handling of file not found error."""
    with pytest.raises(FileNotFoundError) as excinfo:
        detector.predict(config.RAW_DATA_DIR / "NotFound.png")
    assert "Image file not found" in str(excinfo.value)

def test_invalid_file():
    """Test handling of invalid OS error."""
    with pytest.raises(ValueError) as excinfo:
        detector.predict(config.RAW_DATA_DIR / "not_an_image.txt")
    assert "Cannot open image file" in str(excinfo.value)


def test_empty_table_image():
    """Test prediction on an image with no tables."""
    result = detector.predict(config.RAW_DATA_DIR / "empty.png")
    assert isinstance(result, dict)
    assert len(result["boxes"]) == 0