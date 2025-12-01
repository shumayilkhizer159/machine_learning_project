
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# --- COPIED CLASSES ---

class NFLDataLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
    def preprocess_input_data(self, df):
        return df

class FeatureEngineering:
    def add_temporal_features(self, df): return df
    def add_physics_features(self, df): return df

class XGBoostTrajectoryModel:
    def __init__(self):
        self.models_x = {}
        self.models_y = {}
    def load_models(self, save_dir):
        import pickle
        save_dir = Path(save_dir)
        print(f"Attempting to load metadata from {save_dir / 'metadata.pkl'}")
        with open(save_dir / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        print("Metadata loaded.")
        # Mock loading models to avoid needing actual xgb files for this quick test
        # or actually load them if they exist
        self.max_future_frames = metadata['max_future_frames']
        print(f"Max future frames: {self.max_future_frames}")

# --- TEST LOGIC ---

CONFIG = {
    'data_dir': 'c:/Machine_Learning_Project/nfl_project_complete/nfl_project/data',
    'models_dir': 'c:/Machine_Learning_Project/nfl_project_complete/nfl_project/models' 
}

print("=== STARTING LOCAL LOADING TEST ===")

# 1. Scaler Search
print('Searching for scaler.pkl...')
possible_scalers = glob.glob(f"{CONFIG['models_dir']}/**/scaler.pkl", recursive=True)
if possible_scalers:
    scaler_path = possible_scalers[0]
    print(f'✅ Found scaler at: {scaler_path}')
    scaler = joblib.load(scaler_path)
    print("Scaler loaded.")
else:
    print('❌ Scaler NOT found.')
    # In the real script, this disables the model.

# 2. Model Search
print('Searching for metadata.pkl...')
possible_models = glob.glob(f"{CONFIG['models_dir']}/**/metadata.pkl", recursive=True)

if possible_models:
    model_dir = os.path.dirname(possible_models[0])
    print(f'✅ Found model directory at: {model_dir}')
    
    model = XGBoostTrajectoryModel()
    try:
        model.load_models(model_dir)
        print('✅ Models loaded successfully')
    except Exception as e:
        print(f'❌ Error loading models: {e}')
        # This mimics the CRASH we want to see if it fails
        raise e
else:
    print('❌ Model params NOT found.')
    # This mimics the CRASH we want to see
    raise FileNotFoundError("Model params not found!")

print("=== TEST COMPLETED SUCCESSFULLY ===")
