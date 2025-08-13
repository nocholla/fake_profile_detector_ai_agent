import pytest
import pandas as pd
import sys
from pathlib import Path
from src.run import FakeProfileDetector, load_data, preprocess_data
import yaml

# Ensure src directory is in sys.path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

@pytest.fixture
def config():
    with open(BASE_DIR / 'config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_load_data(config):
    df = load_data(config)
    assert 'label' in df.columns
    assert 'userId' in df.columns

def test_preprocess_data(config):
    df = load_data(config)
    X, y, _, _, _ = preprocess_data(df, config)
    assert X.shape[0] == y.shape[0]

def test_detector_train(config):
    detector = FakeProfileDetector()
    detector.train()
    assert detector.rf_model is not None
    assert detector.iforest is not None
    assert detector.anomaly_net is not None
    # xgb_model may be None if only one class is present