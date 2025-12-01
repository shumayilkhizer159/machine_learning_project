# Local GPU Training Guide - NFL Big Data Bowl 2026

This guide provides detailed, step-by-step instructions for setting up the environment and running the complete model training pipeline on your local machine with GPU support.

---

## 1. Prerequisites

Before you begin, ensure you have the following installed:

- **NVIDIA GPU**: A modern NVIDIA GPU with at least 8GB of VRAM is recommended for training the deep learning models.
- **NVIDIA Drivers**: The latest NVIDIA drivers for your GPU.
- **CUDA Toolkit**: CUDA 11.8 or 12.x. You can download it from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).
- **Python**: Python 3.10 or 3.11.
- **Git**: For cloning the project repository.

---

## 2. Environment Setup

Follow these steps to create an isolated environment and install all necessary dependencies.

**Step 1: Clone the Project Repository**

```bash
git clone <repository_url>
cd nfl-big-data-bowl-2026
```

**Step 2: Create a Python Virtual Environment**

This ensures that the project's dependencies do not conflict with other Python projects on your system.

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows, the activation command is:

```bash
venv\Scripts\activate
```

**Step 3: Install Python Dependencies**

We have provided a `requirements.txt` file that lists all the required packages. Install them using pip.

```bash
pip install -r requirements.txt
```

This will install all necessary libraries, including `torch` (with GPU support), `xgboost`, `pandas`, and `scikit-learn`.

**Step 4: Set Up Kaggle API Credentials**

To download the competition data, you need to authenticate with Kaggle.

1.  Go to your Kaggle account settings: [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
2.  Scroll down to the "API" section and click **"Create New Token"**.
3.  This will download a `kaggle.json` file.
4.  Place this file in the `~/.kaggle/` directory (for Linux/macOS) or `C:\Users\<Your-Username>\.kaggle\` (for Windows).

```bash
# On Linux/macOS
mkdir -p ~/.kaggle
mv path/to/your/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**Step 5: Download the Competition Data**

Run the following command from the project's root directory to download and extract the data:

```bash
kaggle competitions download -c nfl-big-data-bowl-2026-prediction -p data/
unzip data/nfl-big-data-bowl-2026-prediction.zip -d data/
rm data/nfl-big-data-bowl-2026-prediction.zip
```

After this step, your `data/` directory should contain the `train/` folder with all the weekly CSV files.

---

## 3. Running the Training Pipeline

The main script for training all models is `src/train_gpu.py`. This script will:

1.  Load and preprocess all 18 weeks of training data.
2.  Perform feature engineering.
3.  Train the Baseline, XGBoost, LSTM, GRU, and Transformer models.
4.  Save the trained models and performance metrics.

**To start the training process, run the following command:**

```bash
python src/train_gpu.py
```

**Expected Output:**

You will see detailed logs in your terminal as the script progresses:
- Data loading and preprocessing status.
- Training progress for each model, including epoch-by-epoch loss and RMSE.
- Final RMSE scores for each model.

**Important Notes:**
- **Training Time**: Training all models, especially the deep learning ones, can take a significant amount of time, even with a powerful GPU. Expect it to run for several hours.
- **Monitoring**: You can monitor your GPU usage using the `nvidia-smi` command in a separate terminal.
- **Checkpoints**: The script is configured to save the best version of each deep learning model to the `models/` directory whenever a new best validation score is achieved. This means if the training is interrupted, you won't lose all your progress.

---

## 4. Customizing the Training

You can customize the training process by editing the `CONFIG` dictionary at the top of `src/train_gpu.py`.

**Key Configuration Options:**

- `weeks_to_train`: To run a quick test, you can reduce the number of weeks to train on, for example: `weeks_to_train: [1, 2, 3]`.
- `batch_size`: If you run into GPU memory errors (`CUDA out of memory`), try reducing the `batch_size` (e.g., from 64 to 32 or 16).
- `epochs`: You can adjust the number of training epochs for each model to train for a shorter or longer duration.
- `device`: If you want to force training on the CPU, you can change this to `'cpu'`, but it will be extremely slow.

---

## 5. Using the Trained Models

Once the training is complete, the best models will be saved in the `models/` directory.

- **XGBoost**: Saved as a set of JSON files in `models/xgboost/`.
- **LSTM, GRU, Transformer**: Saved as `.pth` checkpoint files in their respective directories (`models/lstm/`, etc.).

The `ModelEnsemble` class in `src/model_ensemble.py` is designed to load these trained models and combine their predictions. You can use this class for inference or to build your final submission.

**Example of using the ensemble:**

```python
from src.model_ensemble import ModelEnsemble

# Initialize and load models
ensemble = ModelEnsemble()
ensemble.load_models(model_types=["lstm", "gru"])

# Set weights (e.g., based on validation performance)
ensemble.set_weights({"lstm": 0.6, "gru": 0.4})

# Make predictions on new data
# (You would need to load and preprocess the new data first)
# predictions = ensemble.predict_play(new_play_data, feature_cols)
```

This guide provides a complete roadmap for training and evaluating the models on your own hardware. Good luck!
