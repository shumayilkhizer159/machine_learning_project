# GitHub Upload Guide

## 1. Important Files & Folders
You do NOT need to upload everything. Data files are huge and should be ignored.

### ✅ Upload These (The "Code"):
*   **`src/`**: This is the brain of your project.
    *   `data_loader.py`: Handles loading and cleaning data.
    *   `model_xgboost.py`: The AI model definition and training script.
    *   `create_submission_notebook.py`: The script that generates the final Kaggle notebook.
*   **`notebooks/`**:
    *   `inference_notebook.ipynb`: Your final product.
*   **`requirements.txt`**: Lists the libraries needed (pandas, xgboost, etc.).
*   **`README.md`**: The manual for your project.
*   **`Presentation Material/`**: Your defense guide.

### ❌ Do NOT Upload These (The "Data" & "Junk"):
*   `data/`: **NEVER** upload this. It's too big and belongs to Kaggle.
*   `models/`: Usually too big for GitHub (limit is 100MB). If your models are small, you can, but it's better to provide a script to *retrain* them.
*   `__pycache__/`: Python junk files.
*   `.ipynb_checkpoints/`: Notebook backup files.

---

## 2. How to Upload
1.  **Create a New Repository** on GitHub (e.g., "NFL-Trajectory-Prediction").
2.  **Initialize Git** locally:
    ```bash
    git init
    ```
3.  **Create a `.gitignore` file** (to automatically ignore junk):
    Create a file named `.gitignore` and add these lines:
    ```text
    data/
    models/
    __pycache__/
    *.pyc
    .ipynb_checkpoints/
    ```
4.  **Add and Commit**:
    ```bash
    git add .
    git commit -m "Initial commit of NFL Prediction Project"
    ```
5.  **Push to GitHub**:
    ```bash
    git remote add origin https://github.com/YOUR_USERNAME/NFL-Trajectory-Prediction.git
    git push -u origin master
    ```

---

## 3. How to Run (Instructions for README)
Add this to your `README.md`:

### Installation
```bash
pip install -r requirements.txt
```

### Training the Model
To train the XGBoost model from scratch:
```bash
python src/model_xgboost.py
```
*Note: Ensure you have the NFL Big Data Bowl 2026 dataset in the `data/` folder.*

### Generating Submission
To create the inference notebook for Kaggle:
```bash
python src/create_submission_notebook.py
```
