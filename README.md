# NFL Big Data Bowl 2026: Player Trajectory Prediction

## Project Overview
This project predicts the future trajectory (x, y coordinates) of NFL players after a pass is thrown, using a hybrid approach that combines **Physics-based Kinematics** and **Machine Learning (XGBoost)**.

## Key Features
*   **Hybrid Ensemble**: Physics baseline (Constant Velocity) + XGBoost model.
*   **Autoregressive Prediction**: Sequential future frame prediction.
*   **Safety Net**: Fallback to physics for unrealistic ML predictions.

## Project Structure
*   `src/`: Source code for data loading, training, and inference.
*   `notebooks/`: Generated Jupyter notebooks.
*   `models/`: Trained model artifacts (not included due to size).
*   `data/`: competition data (not included).

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/shumayilkhizer159/machine_learning_project.git
cd machine_learning_project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```



### 3. Run Inference
To generate the submission notebook for Kaggle:
```bash
python src/create_submission_notebook.py
```
This produces `notebooks/inference_notebook.ipynb`.

### 4. Training and Execution
For detailed instructions on how to train models and run the pipeline step-by-step, please see [RUNNING_GUIDE.md](RUNNING_GUIDE.md).
