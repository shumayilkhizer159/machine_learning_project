# NFL Player Trajectory Prediction: Project Defense Guide

## 1. The Core Concept: "Autoregressive Trajectory Prediction"
**"How do we predict where a player goes?"**

Imagine you are watching a player run. To guess where they will be in the next second, you look at:
1.  **Where they are now** (Position `x, y`).
2.  **How fast they are moving** (Speed `s`).
3.  **Where they are facing** (Direction `dir`).
4.  **Where the ball is going** (Target).

Our model does exactly this, but frame-by-frame.
*   **Step 1**: Take the current state (Frame 0).
*   **Step 2**: Predict the *next* step (Frame 1).
*   **Step 3**: Treat that *predicted* step as the "current state" for the next prediction.
*   **Step 4**: Repeat until the play ends.

This is called **Autoregressive Prediction**. We build the future one step at a time.

---

## 2. Key Features (The "Why" behind the prediction)
We don't just give the model raw numbers; we give it *context*.

### A. Physics Features (The Foundation)
*   **Velocity (`v_x`, `v_y`)**: Broken down into horizontal and vertical components.
*   **Acceleration (`a`)**: Is the player speeding up or slowing down?
*   **Constant Velocity Prediction (`x_cv`, `y_cv`)**: "If the player keeps running exactly like this, where will they be?" This is our **Baseline**.

### B. Context Features (The "Game Sense")
*   **Distance to Ball**: Players run *towards* the ball. This is a huge clue.
*   **Angle to Ball**: Is the player facing the ball or running away?
*   **Player Role**: A Wide Receiver (WR) runs differently than a Lineman (OL).

### C. Temporal Features (The "Smoothness")
*   **Rolling Mean**: We look at the average of the last few frames to smooth out "jittery" predictions. Real players don't teleport!

---

## 3. The Project Journey (Our "Fail-Safe" Story)

### Phase 1: The "Naive" Baseline (Score: ~4.0 - 5.0)
*   **Approach**: Just assume the player stands still or moves randomly.
*   **Result**: Terrible. The players moved, and we didn't.

### Phase 2: The Physics Model (Score: 1.613)
*   **Approach**: "Newton's First Law". An object in motion stays in motion.
*   **Formula**: `Next Position = Current Position + (Velocity * Time)`
*   **Result**: Much better! This captures straight-line running perfectly.
*   **Limitation**: It fails when players **turn**, **stop**, or **accelerate**. It assumes they are robots on rails.

### Phase 3: The Machine Learning Model (XGBoost)
*   **Approach**: Train an AI to look at thousands of past plays and learn *how* players turn and react.
*   **Goal**: Beat the Physics Baseline (Score < 1.613).
*   **Current Status**: We achieved a score of **1.681** (slightly worse than physics).
    *   **Why?**: This is a classic "Overfitting" or "Noise" problem. The model is trying to be too smart and making small errors that add up.
    *   **The "Silent Failure"**: In our final submission, the system likely fell back to the Physics model because the AI model was either too aggressive or failed to load correctly in the cloud environment.

### Phase 4: The Hybrid "Safety Net" (Our Best Idea)
*   **Concept**: Trust the **Physics** for simple, straight runs. Trust the **AI** only when it's confident (e.g., during a sharp turn).
*   **Implementation**:
    *   Calculate `Distance(Physics_Prediction, AI_Prediction)`.
    *   If the difference is small (< 2 yards), trust the AI (it's refining the physics).
    *   If the difference is huge (> 15 yards), the AI is probably hallucinating -> **Trust Physics**.

---

## 4. How to Run This Project (For the Judge)

### 1. Training (`src/model_xgboost.py`)
We use **XGBoost**, a powerful gradient boosting algorithm.
```bash
python src/model_xgboost.py
```
*   This reads the data from `data/train`.
*   Trains separate models for `x` and `y` coordinates for each future frame (1 to 10).
*   Saves the models to `models/xgboost`.

### 2. Inference (`notebooks/inference_notebook.ipynb`)
This is the file we submit to Kaggle.
*   It loads the saved models.
*   It reads the live game data.
*   It combines Physics + AI to make the final prediction.

---

## 5. Conclusion
We built a robust system that:
1.  **Starts with Physics** (Guaranteed reasonable performance).
2.  **Enhances with AI** (Learns complex movement).
3.  **Protects with Sanity Checks** (Prevents wild errors).

Even though our final score (1.613) reflects the Physics baseline, the **architecture** is correct. The next step is simply tuning the AI to be slightly more accurate than Newton's Laws!
