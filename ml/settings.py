import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ML_MODEL = os.path.join(BASE_DIR, 'ml')
DATA_DIR = os.path.join(ML_MODEL, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')

TRAIN_RAW_DIR = os.path.join(RAW_DATA_DIR, 'train')
TEST_RAW_DIR = os.path.join(RAW_DATA_DIR, 'test')

APP_DIR = os.path.join(BASE_DIR, 'app')

MODEL_DIR = os.path.join(ML_MODEL, 'saved_models')
