# NFL Big Data Bowl 2026: Trajectory Prediction Report

**Team:** [Your Team Name]
**Date:** December 1, 2025

---

## 1. Introduction & Challenge Overview
The goal of this project was to predict the future trajectory (x, y coordinates) of NFL players after a pass is thrown. This is a time-series forecasting problem where the input is the state of players (position, velocity, direction) up to the moment of the pass, and the output is their location for the next 10 frames (1 second).

**The Challenge:** NFL players do not move linearly. They accelerate, decelerate, cut, and react to the ball and other players. A simple physics model fails to capture these complex behaviors, while a pure machine learning model often struggles with "noise" and overfitting, leading to erratic predictions.

---

## 2. Data Exploration & Strategy

### 2.1 Data Analysis
We utilized the NFL Big Data Bowl tracking data, which provides player locations at 10Hz.
*   **Key Features**: `x`, `y`, `s` (speed), `dir` (direction), `o` (orientation).
*   **Observation**: Player movement is highly correlated with their current velocity vector. However, significant deviations occur when players are "targeting" the ball or reacting to defenders.

### 2.2 Strategy Selection
We evaluated two primary strategies:
1.  **Physics-Based (Kinematics)**: Assuming constant velocity or constant acceleration.
    *   *Pros*: Extremely robust, never makes "impossible" errors.
    *   *Cons*: Cannot predict turns or stops.
2.  **Machine Learning (XGBoost)**: Training a gradient boosting model on historical trajectories.
    *   *Pros*: Can learn non-linear patterns (curves, cuts).
    *   *Cons*: Prone to overfitting and "teleportation" errors if not constrained.

**Our Choice: The Hybrid Ensemble**
We chose a **Hybrid Approach**. We use the Physics model as a "Safety Net" and the XGBoost model as a "Refinement".
*   **Base Prediction**: Calculate where the player would be if they ran in a straight line.
*   **ML Adjustment**: Use XGBoost to predict the *deviation* from this straight line based on context (ball location, role).
*   **Sanity Check**: If the ML model predicts a location > 15 yards away from the physics baseline (impossible in 1 second), we discard it and fallback to physics.

---

## 3. Methodology

### 3.1 Feature Engineering
We engineered features to give the model "Game Sense":
1.  **Physics**: Decomposed velocity (`v_x`, `v_y`) and acceleration.
2.  **Context**:
    *   `dist_to_ball`: Euclidean distance to the football.
    *   `angle_to_ball`: Is the player looking at the ball?
    *   `player_role`: Encoded categorical features (WR, CB, etc.).
3.  **Temporal**: Rolling means (window=3) to smooth out sensor noise.

### 3.2 Model Architecture
We trained separate **XGBoost Regressors** for `x` and `y` coordinates for each future frame.
*   **Input**: State at Frame $t=0$.
*   **Output**: Position at Frame $t+k$ (where $k \in [1, 10]$).
*   **Training Data**: 2023 NFL Season (Weeks 1-8).

### 3.3 Handling Incomplete Data
*   **Missing Values**: We filled missing velocity/direction with `0.0` (assuming stationary).
*   **Robust Loading**: We implemented a `glob`-based search to ensure our model files (`metadata.pkl`) are found regardless of the directory structure in the evaluation environment.

---

## 4. Results & Evaluation

### 4.1 Metrics
The competition uses **RMSE (Root Mean Squared Error)**.
*   Lower is better.
*   Measures the average distance between predicted and actual location.

### 4.2 Performance
*   **Baseline (Physics Only)**: **1.613**
*   **XGBoost Model (Pure)**: **> 4.0** (Suffered from overfitting/noise).
*   **Hybrid Model (Current)**: **1.681**

### 4.3 Analysis
Our Hybrid model scored slightly worse (1.681) than the pure Physics baseline (1.613).
*   **Why?** The XGBoost model, while capturing some turns, introduced more noise than signal for straight-line runs. The "Sanity Check" threshold of 15 yards was likely too loose, allowing some bad ML predictions to slip through.
*   **The "Silent Failure"**: During development, we discovered that if the model fails to load (e.g., wrong path), the system must fallback to physics. We spent significant time debugging this to ensure we *always* produce a valid submission.

---

## 5. Critical Reflection (What could be better?)

### 5.1 What went wrong?
1.  **Over-reliance on "Smart" Features**: We assumed the model would easily learn that players run towards the ball. In reality, this introduced noise because players often run *routes* first, then look for the ball.
2.  **Categorical Encoding Mismatch**: We suspected a mismatch between how "Positions" (WR, CB) were encoded in training vs. inference, potentially confusing the model.

### 5.2 Future Improvements
1.  **Better Target Encoding**: Instead of raw coordinates, predict the *acceleration vector*. This bounds the problem (players can't accelerate infinitely).
2.  **Transformer Models**: Use a Sequence-to-Sequence Transformer instead of XGBoost. Transformers are better at understanding the "history" of movement (the past 10 frames) to predict the future.
3.  **Tighter Integration**: Use the Physics model as a *feature* inside the XGBoost model, rather than just an ensemble partner.

---

## 6. Conclusion
We successfully built an end-to-end pipeline that processes raw tracking data, engineers complex features, and generates predictions. While our ML model did not beat the robust Physics baseline, our **System Architecture** (Hybrid + Fallback) is the correct approach for real-world deployment where reliability is paramount. We prioritized **Robustness** over risky optimization.
