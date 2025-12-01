import json
import os
from pathlib import Path

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def create_notebook():
    print("Generating Kaggle Submission Notebook...")
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    src_dir = base_dir / 'src'
    models_dir = base_dir / 'models'
    output_dir = base_dir / 'notebooks'
    output_dir.mkdir(exist_ok=True)
    
    # Read source files
    data_loader_code = read_file(src_dir / 'data_loader.py')
    model_xgboost_code = read_file(src_dir / 'model_xgboost.py')
    
    # Combine all code into a single source list
    combined_source = []
    
    # 1. System Diagnostic
    combined_source.extend([
        "import os\n",
        "print(\"=== SYSTEM DIAGNOSTIC ===\")\n",
        "print(\"Listing ALL files in /kaggle/input:\")\n",
        "if os.path.exists('/kaggle/input'):\n",
        "    found_any = False\n",
        "    for root, dirs, files in os.walk('/kaggle/input'):\n",
        "        level = root.replace('/kaggle/input', '').count(os.sep)\n",
        "        indent = ' ' * 4 * (level)\n",
        "        print(f\"{indent}{os.path.basename(root)}/\")\n",
        "        subindent = ' ' * 4 * (level + 1)\n",
        "        for f in files:\n",
        "            found_any = True\n",
        "            print(f\"{subindent}{f}\")\n",
        "    if not found_any:\n",
        "        print(\"  (Directory is empty)\")\n",
        "else:\n",
        "    print(\"/kaggle/input does not exist (Are you running locally?)\")\n",
        "print(\"=========================\")\n",
        "\n"
    ])

    # 2. Imports and Setup
    combined_source.extend([
        "import numpy as np\n",
        "import pandas as pd\n",
        "import xgboost as xgb\n",
        "import joblib\n",
        "import json\n",
        "import sys\n",
        "import torch\n",
        "from pathlib import Path\n",
        "from tqdm.notebook import tqdm\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# Configuration\n",
        "CONFIG = {\n",
        "    # 1. COMPETITION DATA (Where test.csv lives)\n",
        "    # DO NOT CHANGE this if you added the 'NFL Big Data Bowl 2026' dataset.\n",
        "    'data_dir': '/kaggle/input/nfl-big-data-bowl-2026-prediction',\n",
        "\n",
        "    # 2. YOUR MODELS (Where you uploaded your 'models' folder)\n",
        "    # CHANGE THIS to match the path of your uploaded dataset.\n",
        "    # It usually looks like: '/kaggle/input/YOUR-DATASET-NAME/models/xgboost'\n",
        "    'models_dir': '/kaggle/input/my-data-set-2/models/xgboost',\n",
        "\n",
        "    'device': 'cuda' if torch.cuda.is_available() else 'cpu'\n",
        "}\n",
        "\n",
        "# Auto-detect models_dir if not found\n",
        "if not Path(CONFIG['models_dir']).exists():\n",
        "    print(f\"⚠️ Default models_dir '{CONFIG['models_dir']}' not found. Searching...\")\n",
        "    # Search for a known model file to locate the directory\n",
        "    possible_models = list(Path('/kaggle/input').glob('**/xgboost_params.json'))\n",
        "    if possible_models:\n",
        "        # Found the params file, the parent is likely the models dir\n",
        "        # Check if it is inside 'xgboost' folder or if it IS the root\n",
        "        # We expect structure: .../models/xgboost/xgboost_params.json\n",
        "        found_dir = possible_models[0].parent\n",
        "        CONFIG['models_dir'] = str(found_dir)\n",
        "        print(f\"✅ Auto-detected models_dir at: {CONFIG['models_dir']}\")\n",
        "    else:\n",
        "        print(\"❌ Could not auto-detect models directory. Please check 'models_dir' path.\")\n",
        "\n",
        "# Auto-detect competition data if not found\n",
        "if not Path(CONFIG['data_dir']).exists():\n",
        "    print(f\"⚠️ Default data_dir '{CONFIG['data_dir']}' not found. Searching...\")\n",
        "    found_data = False\n",
        "    if Path('/kaggle/input').exists():\n",
        "        for d in Path('/kaggle/input').iterdir():\n",
        "            if d.is_dir() and (d / 'test.csv').exists():\n",
        "                CONFIG['data_dir'] = str(d)\n",
        "                print(f\"✅ Auto-detected competition data at: {CONFIG['data_dir']}\")\n",
        "                found_data = True\n",
        "                break\n",
        "    if not found_data:\n",
        "        print(\"❌ Could not auto-detect competition data. Please check 'Add Data'.\")\n",
        "\n",
        "# Feature columns (Must match training)\n",
        "FEATURE_COLS = [\n",
        "    'x', 'y', 's', 'a', 'dir', 'o',\n",
        "    'ball_land_x', 'ball_land_y', 'dist_to_ball_land',\n",
        "    'v_x', 'v_y',\n",
        "    'player_position_encoded', 'player_side_encoded', 'player_role_encoded',\n",
        "    'player_height_inches', 'player_weight', 'player_age',\n",
        "    'play_direction_binary', 'absolute_yardline_number'\n",
        "]\n",
        "\n",
        "\n",
        "# Verify paths immediately\n",
        "print('Checking configuration paths...')\n",
        "for key, path in CONFIG.items():\n",
        "    if key.endswith('_dir'):\n",
        "        if Path(path).exists():\n",
        "            print(f'✅ {key} found: {path}')\n",
        "        else:\n",
        "            print(f'❌ {key} NOT found: {path}')\n",
        "\n",
        "print('Cell 1 (Imports and Config) executed successfully')\n",
        "\n"
    ])
    
    # 3. Data Loader Class
    combined_source.extend(data_loader_code.splitlines(keepends=True))
    combined_source.append("\nprint('Cell 2 (Data Loader) executed successfully')\n\n")
    
    # 4. XGBoost Model Class
    combined_source.extend(model_xgboost_code.splitlines(keepends=True))
    combined_source.append("\nprint('Cell 3 (XGBoost Model) executed successfully')\n\n")

    inference_code = [
        "import polars as pl",
        "import kaggle_evaluation.nfl_inference_server",
        "import os",
        "import glob",
        "",
        "# --- Global Setup ---",
        "print('Initializing Global Resources...')",
        "",
        "# 1. Robust Scaler Search",
        "scaler = None",
        "print('Searching for scaler.pkl...')",
        "possible_scalers = glob.glob('/kaggle/input/**/scaler.pkl', recursive=True)",
        "if possible_scalers:",
        "    scaler_path = possible_scalers[0]",
        "    print(f'✅ Found scaler at: {scaler_path}')",
        "    scaler = joblib.load(scaler_path)",
        "else:",
        "    print('❌ Scaler NOT found. Model predictions will be garbage.')",
        "",
        "# 2. Load Models",
        "model = None",
        "print('Searching for metadata.pkl...')",
        "possible_models = glob.glob('/kaggle/input/**/metadata.pkl', recursive=True)",
        "if possible_models:",
        "    model_dir = os.path.dirname(possible_models[0])",
        "    print(f'✅ Found model directory at: {model_dir}')",
        "    loader = NFLDataLoader(CONFIG['data_dir'])",
        "    fe = FeatureEngineering()",
        "    model = XGBoostTrajectoryModel()",
        "    model.load_models(model_dir)",
        "    print('✅ Models loaded successfully')",
        "else:",
        "    print('❌ Model metadata NOT found in /kaggle/input. Model predictions will be garbage.')",
        "",
        "def predict(test: pl.DataFrame, test_input: pl.DataFrame):",
        "    # Initialize with Center Field (Better than 0,0)",
        "    n_rows = len(test)",
        "    x_preds = [60.0] * n_rows",
        "    y_preds = [26.65] * n_rows",
        "    ",
        "    try:",
        "        # Convert to Pandas",
        "        test_pd = test.to_pandas()",
        "        test_input_pd = test_input.to_pandas()",
        "        ",
        "        # DEBUG: Print columns to diagnose 'frame_id' error",
        "        if not hasattr(predict, 'debug_printed'):",
        "            print(f'DEBUG: Test Columns: {test_pd.columns.tolist()}')",
        "            print(f'DEBUG: Input Columns: {test_input_pd.columns.tolist()}')",
        "            predict.debug_printed = True",
        "",
        "        # --- 1. Physics Calculation (Vectorized) ---",
        "        # Get last known state for each player",
        "        last_input = test_input_pd.sort_values('frame_id').groupby(['game_id', 'play_id', 'nfl_id']).last().reset_index()",
        "        ",
        "        # Calculate velocities",
        "        s_filled = last_input['s'].fillna(0.0)",
        "        dir_filled = last_input['dir'].fillna(0.0)",
        "        last_input['v_x_calc'] = s_filled * np.sin(np.radians(dir_filled))",
        "        last_input['v_y_calc'] = s_filled * np.cos(np.radians(dir_filled))",
        "        ",
        "        # Prepare last state for merge",
        "        last_input_merge = last_input[['game_id', 'play_id', 'nfl_id', 'x', 'y', 'frame_id', 'v_x_calc', 'v_y_calc']].rename(",
        "            columns={'x': 'x_last', 'y': 'y_last', 'frame_id': 'frame_id_last'}",
        "        )",
        "        ",
        "        # Merge last state into test dataframe",
        "        test_pd['original_index'] = test_pd.index",
        "        test_merged = test_pd.merge(",
        "            last_input_merge,",
        "            on=['game_id', 'play_id', 'nfl_id'],",
        "            how='left'",
        "        )",
        "        ",
        "        # Calculate dt",
        "        sort_cols = ['game_id', 'play_id', 'nfl_id']",
        "        if 'frame_id' in test_merged.columns:",
        "            sort_cols.append('frame_id')",
        "        test_merged = test_merged.sort_values(sort_cols)",
        "        test_merged['frame_offset'] = test_merged.groupby(['game_id', 'play_id', 'nfl_id']).cumcount() + 1",
        "        test_merged['dt'] = test_merged['frame_offset'] * 0.1",
        "            ",
        "        # Physics Predictions",
        "        test_merged['x_cv'] = test_merged['x_last'] + test_merged['v_x_calc'] * test_merged['dt']",
        "        test_merged['y_cv'] = test_merged['y_last'] + test_merged['v_y_calc'] * test_merged['dt']",
        "        ",
        "        # Fill NaNs with Center Field",
        "        test_merged['x_cv'] = test_merged['x_cv'].fillna(60.0)",
        "        test_merged['y_cv'] = test_merged['y_cv'].fillna(26.65)",
        "        ",
        "        # Default final predictions to Physics",
        "        test_merged['x_final'] = test_merged['x_cv']",
        "        test_merged['y_final'] = test_merged['y_cv']",
        "        ",
        "        # --- 2. XGBoost Prediction ---",
        "        if model is not None and scaler is not None:",
        "            # Preprocessing",
        "            test_input_pd = loader.preprocess_input_data(test_input_pd)",
        "            test_input_pd = fe.add_physics_features(test_input_pd)",
        "            test_input_pd = fe.add_temporal_features(test_input_pd)",
        "            ",
        "            # Scaling",
        "            for col in FEATURE_COLS:",
        "                if col not in test_input_pd.columns:",
        "                    test_input_pd[col] = 0.0",
        "            test_input_pd[FEATURE_COLS] = scaler.transform(test_input_pd[FEATURE_COLS])",
        "",
        "            # Predict",
        "            last_input_preprocessed = test_input_pd.sort_values('frame_id').groupby(['game_id', 'play_id', 'nfl_id']).last().reset_index()",
        "            players_to_predict = last_input_preprocessed[last_input_preprocessed['player_to_predict'] == True]",
        "            ",
        "            if len(players_to_predict) > 0:",
        "                max_frame = int(test_merged['frame_offset'].max())",
        "                batch_preds = model.predict_batch(last_input_preprocessed, FEATURE_COLS, max_frame)",
        "                ",
        "                # Convert to DataFrame",
        "                pred_data = []",
        "                for key, preds in batch_preds.items():",
        "                    for i, (px, py) in enumerate(preds):",
        "                        pred_data.append(list(key) + [i + 1, px, py])",
        "                ",
        "                if pred_data:",
        "                    pred_df = pd.DataFrame(pred_data, columns=['game_id', 'play_id', 'nfl_id', 'frame_offset', 'x_model', 'y_model'])",
        "                    ",
        "                    # Merge model preds",
        "                    test_merged = test_merged.merge(",
        "                        pred_df,",
        "                        on=['game_id', 'play_id', 'nfl_id', 'frame_offset'],",
        "                        how='left'",
        "                    )",
        "",
        "                    # --- 3. SANITY CHECK & ENSEMBLE ---",
        "                    # Calculate distance between Model and Physics",
        "                    test_merged['dist_diff'] = np.sqrt(",
        "                        (test_merged['x_model'] - test_merged['x_cv'])**2 + ",
        "                        (test_merged['y_model'] - test_merged['y_cv'])**2",
        "                    )",
        "                    ",
        "                    # Logic: Aggressive Model Trust (Break the Physics Baseline)",
        "                    # Threshold: 15.0 yards (Allow turns)",
        "                    # Weights: 0.6 Model / 0.4 Physics",
        "                    ",
        "                    mask_use_ensemble = (test_merged['x_model'].notna()) & (test_merged['dist_diff'] < 15.0)",
        "                    ",
        "                    # Apply Ensemble (0.6 * Model + 0.4 * Physics)",
        "                    test_merged.loc[mask_use_ensemble, 'x_final'] = (",
        "                        0.6 * test_merged.loc[mask_use_ensemble, 'x_model'] + ",
        "                        0.4 * test_merged.loc[mask_use_ensemble, 'x_cv']",
        "                    )",
        "                    test_merged.loc[mask_use_ensemble, 'y_final'] = (",
        "                        0.6 * test_merged.loc[mask_use_ensemble, 'y_model'] + ",
        "                        0.4 * test_merged.loc[mask_use_ensemble, 'y_cv']",
        "                    )",
        "                    # Else: Keep Physics (already in x_final)",
        "                    ",
        "                    # --- 4. TEMPORAL SMOOTHING ---",
        "                    # Smooth the final trajectory to remove jitter",
        "                    # Window 3",
        "                    test_merged['x_final'] = test_merged.groupby(['game_id', 'play_id', 'nfl_id'])['x_final'].transform(",
        "                        lambda x: x.rolling(3, min_periods=1).mean()",
        "                    )",
        "                    test_merged['y_final'] = test_merged.groupby(['game_id', 'play_id', 'nfl_id'])['y_final'].transform(",
        "                        lambda x: x.rolling(3, min_periods=1).mean()",
        "                    )",
        "",
        "        test_merged = test_merged.sort_values('original_index')",
        "        x_preds = test_merged['x_final'].values",
        "        y_preds = test_merged['y_final'].values",
        "",
        "    except Exception as e:",
        "        print(f'CRITICAL ERROR in predict function: {e}')",
        "        # Fallback is already initialized to Center Field",
        "    return pl.DataFrame({'x': x_preds, 'y': y_preds})",
        "",
        "# --- Server Startup ---",
        "inference_server = kaggle_evaluation.nfl_inference_server.NFLInferenceServer(predict)",
        "",
        "if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):",
        "    print('Starting Inference Server...')",
        "    inference_server.serve()",
        "else:",
        "    print('Running Local Gateway...')",
        "    inference_server.run_local_gateway((CONFIG['data_dir'],))"
    ]
    
    combined_source.extend([line + "\n" for line in inference_code])

    # Create single cell
    cells = [{
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": combined_source
    }]
    
    # Notebook JSON structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    output_path = output_dir / 'inference_notebook.ipynb'
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)
        
    print(f"Notebook created at {output_path}")

if __name__ == "__main__":
    create_notebook()
