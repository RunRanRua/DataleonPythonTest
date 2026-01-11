import pytest
from detector import TableDetector
import config

detector = TableDetector(model_name=config.MODEL_NAME)

def test_invoice_prediction_bordered():
    """Test prediction on an invoice image with bordered tables."""
    results = detector.predict(config.RAW_DATA_DIR / "invoice_table_bordered.png")
    assert isinstance(results, dict)
    assert "boxes" in results
    assert "labels" in results
    assert "scores" in results
    assert len(results["boxes"]) > 0 

def test_invoice_prediction_borderless():
    """Test prediction on an invoice image with borderless tables."""
    results = detector.predict(config.RAW_DATA_DIR / "invoice_table_borderless.png")
    assert isinstance(results, dict)
    assert "boxes" in results
    assert "labels" in results
    assert "scores" in results
    assert len(results["boxes"]) > 0

def test_bank_doc_bordered():
    """Test prediction on a bank document image with bordered tables."""
    results = detector.predict(config.RAW_DATA_DIR / "bank_doc_bordered.png")
    assert isinstance(results, dict)
    assert "boxes" in results
    assert "labels" in results
    assert "scores" in results
    assert len(results["boxes"]) > 0