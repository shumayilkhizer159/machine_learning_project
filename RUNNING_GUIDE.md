# Project Execution Guide

This guide explains how to run the project scripts in the correct order. All scripts are located in the `src/` directory.

## 1. Exploratory Data Analysis (Optional)
**Script:** `src/run_1_eda.py`
*   **Purpose:** Generates initial statistics and visualizations of the dataset.
*   **Output:** Figures saved to `figures/` and notes to `competition_notes.md`.
*   **Command:**
    ```bash
    python src/run_1_eda.py
    ```

## 2. Train Models
**Script:** `src/run_2_train_models.py`
*   **Purpose:** Trains the XGBoost and Deep Learning models on the training data.
*   **Details:** This script handles data loading, preprocessing, and training on GPU (if available).
*   **Output:** Saved models in `models/` and training logs in `outputs/`.
*   **Command:**
    ```bash
    python src/run_2_train_models.py
    ```

## 3. Finalize and Optimize
**Script:** `src/run_3_finalize_project.py`
*   **Purpose:** Monitors training progress (if running in parallel) and performs ensemble optimization to find the best weights.
*   **Output:** Optimized ensemble weights and final validation metrics.
*   **Command:**
    ```bash
    python src/run_3_finalize_project.py
    ```

## 4. Create Submission
**Script:** `src/run_4_create_submission.py`
*   **Purpose:** Packages the code and models into a single Jupyter Notebook (`inference_notebook.ipynb`) for Kaggle submission.
*   **Output:** `notebooks/inference_notebook.ipynb`
*   **Command:**
    ```bash
    python src/run_4_create_submission.py
    ```

## Helper Files
*   `src/data_loader.py`: Shared logic for loading and cleaning data.
*   `src/model_*.py`: Model definitions (XGBoost, Ensemble, etc.).
*   `src/legacy/`: Old or unused scripts.
