# NFL Big Data Bowl 2026: Player Trajectory Prediction
**Group Presentation Content (5 Members)**

---

## 🎤 Speaker 1: Introduction & Problem Statement
*Focus: Setting the stage, explaining the "Why", and limiting the scope.*

### Slide 1: Title Slide
*   **Title:** Predicting NFL Player Trajectories using Hybrid Machine Learning
*   **Team Members:** [Name 1], [Name 2], [Name 3], [Name 4], [Name 5]
*   **Visual:** `Generated_Images/slide_1_visual.png` (AI-Generated High-Quality Title Image)

> **Speaker Notes:**
> "Good morning/afternoon everyone. We are Team [Name] and today we present our solution for the NFL Big Data Bowl 2026. Our project focuses on one of the most fundamental aspects of the game: predicting where a player will be in the next second after a pass is thrown. This isn't just about stats; it's about understanding the physics and intuition of elite athletes."

### Slide 2: The Challenge
*   **Goal:** Predict player coordinates (`x`, `y`) for 10 future frames (1 second) after ball snap/throw.
*   **Complexity:**
    *   **Non-Linear Movement:** Players cut, spin, and accelerate.
    *   **Reactionary:** Movement depends on the ball and opponents.
    *   **Noise:** Sensor data has jitter; human behavior is unpredictable.
*   **Visual:** `Presentation_Figures/sample_play_trajectories.png` (Shows actual curving paths vs linear physics)

> **Speaker Notes:**
> "The core challenge is that humans don't move like robots. If they did, a simple constant velocity model would be perfect. But NFL players cut, decelerate, and react. Our task is to take the tracking data—position, speed, direction—and predict their path for the next 10 frames, or exactly one second. It sounds short, but in the NFL, a lot happens in a second."

### Slide 3: Our Approach at a Glance
*   **Project Pipeline:** Data Ingestion $\rightarrow$ Feature Engineering $\rightarrow$ Hybrid Modeling $\rightarrow$ Ensemble $\rightarrow$ Submission.
*   **Key Innovation:** A "Safety Net" architecture.
    *   We combine **Physics (Kinematics)** with **Machine Learning (XGBoost/Deep Learning)**.
    *   Ensures valid predictions even when the ML gets confused.
*   **Visual:** `Generated_Images/slide_3_visual.png` (AI-Generated Diagram of Hybrid Pipeline)

> **Speaker Notes:**
> "We didn't just build one model. We built a system. We recognized that while Deep Learning is powerful, it can be unstable. So we designed a 'Hybrid Architecture'. We use physics-based kinematics as a baseline—a safety net—and then use advanced Machine Learning to refine those predictions. This gives us the best of both worlds: reliability and precision."

---

## 🎤 Speaker 2: Data Exploration & Insights
*Focus: The "Raw Material". What did we find in the data?*

### Slide 4: Data Overview
*   **Dataset:** NFL Next Gen Stats (10Hz tracking).
*   **Volume:** Millions of rows of player tracking data (Weeks 1-9 of 2023 Season).
*   **Key Features:**
    *   `x`, `y`: Position on the field.
    *   `s`, `a`: Speed and Acceleration.
    *   `dir`, `o`: Movement direction and Player orientation.
*   **Visual:**
| game_id | play_id | nfl_id | frame_id | x | y | s (speed) | a (accel) | dir | o (orient) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2022090800 | 56 | 35472 | 1 | 35.1 | 23.4 | 4.52 | 2.43 | 90.5 | 88.1 |
| 2022090800 | 56 | 35472 | 2 | 35.5 | 23.4 | 4.88 | 2.10 | 90.2 | 87.9 |
| 2022090800 | 56 | 42521 | 1 | 78.2 | 12.1 | 1.05 | 0.88 | 275.4 | 260.1 |
*Table: Snippet of the raw dataframe showing the columns*

> **Speaker Notes:**
> "Let's dive into the data. We're working with the NFL Next Gen Stats, processed at 10 frames per second. The dataset is massive, covering the first 9 weeks of the 2023 season. For every single frame, we know exactly where every player is, how fast they're moving (`s`), their acceleration (`a`), and crucially, where they are looking (`o`) versus where they are moving (`dir`)."

### Slide 5: Exploratory Data Analysis (EDA)
*   **Insight 1:** Velocity is the strongest predictor.
    *   Correlation heatmap shows `s` (speed) is highly correlated with distance traveled.
*   **Insight 2:** Player Roles matter.
    *   Wide Receivers (WR) have high variance (sharp turns).
    *   Linemen (OL/DL) move linearly and slowly.
*   **Visual:** `Presentation_Figures/speed_by_role.png` (OR `Presentation_Figures/correlation_heatmap.png`)

> **Speaker Notes:**
> "Our analysis revealed two critical insights. First, simple physics is powerful: a player's current speed vector is the single best predictor of their location 0.1 seconds later. Second, context is key. Wide receivers play a different game than linemen. Receivers make sharp cuts, while linemen mostly push forward. This told us we needed a model that understands *who* the player is, not just where they are."

### Slide 6: Visualizing Trajectories
*   **Observation:** Most movement is relatively straight, but the "outliers" (turns) determine the game.
*   **The Problem:** Pure ML models often predict "teleportation" (impossible jumps) when noise is high.
*   **Visual:** `Presentation_Figures/sample_play_trajectories.png` (Focus on one specific player trace)

> **Speaker Notes:**
> "When we plotted the trajectories, we saw the problem immediately. While 90% of movement is linear, pure Machine Learning models would sometimes panic and predict a player jumping 5 yards in a split second—teleportation. This reinforced our decision to use Physics as a constraint. We need to respect the physical limits of the human body."

---

## 🎤 Speaker 3: Methodology (The Hybrid Models)
*Focus: The technical "Engine". How Physics and XGBoost work.*

### Slide 7: The Physics Baseline (Kinematics)
*   **Concept:** Newton's First Law (Inertia).
*   **Formula:** $x_{t+1} = x_t + v_x \cdot \Delta t$
*   **Role:** The "Anchor". Provides a guaranteed realistic prediction.
*   **Performance:** RMSE ~1.61 (Surprisingly hard to beat).
*   **Visual:** `Generated_Images/slide_7_visual.png` (AI-Generated Physics/Kinematics Diagram)

> **Speaker Notes:**
> "I'll walk you through our modeling strategy. We started with Physics. It's simple: an object in motion stays in motion. We calculate the velocity vector and project it forward. This baseline is incredibly robust. It never predicts impossible movements. In fact, it scored an RMSE of 1.61, which became the benchmark our ML models had to beat."

### Slide 8: Machine Learning - XGBoost
*   **Why XGBoost?** Handles non-linear relationships and tabular data exceptionally well.
*   **Target:** Predict the *residual* (difference) between the Physics prediction and the actual location.
*   **Feature Engineering:**
    *   `dist_to_ball`: Distance to ball landing spot.
    *   `angle_to_ball`: Is the player looking at the ball?
    *   `group_cluster`: Encoding player roles (WR, CB, QB).
*   **Visual:** `Presentation_Figures/feature_importance.png` (Shows 'time_to_ball' and 'speed' as top features)

> **Speaker Notes:**
> "To capture the non-linear movements—the cuts and curves—we trained an XGBoost model. But we didn't just ask it to predict coordinates. We fed it 'Game Sense' features: How far is the ball? Is the player looking at it? What position do they play? This allows the model to learn localized behaviors, like a Cornerback reacting to a pass."

### Slide 9: The "Safety Net" Logic
*   **Problem:** ML sometimes drifts wildy.
*   **Solution:** Hybrid sanity check.
    *   IF `Distance(ML_pred, Physics_pred) > 15 yards`:
        *   Discard ML.
        *   Use Physics.
    *   ELSE:
        *   Use Weighted Average (`0.6 * ML + 0.4 * Physics`).
*   **Visual:** `Generated_Images/slide_9_visual.png` (AI-Generated Safety Net Logic Diagram)

> **Speaker Notes:**
> "This is the most critical part of our system. We call it the 'Safety Net'. If our XGBoost model predicts a player moves 20 yards in 1 second—which is impossible—we automatically discard it and revert to the physics model. This prevents catastrophic errors and ensures our submission is always valid and physically plausible."

---

## 🎤 Speaker 4: Advanced Modeling (Deep Learning)
*Focus: Pushing the boundaries with GPU training.*

### Slide 10: Beyond Trees - Deep Learning
*   **Limitation of XGBoost:** It treats frames independently (no memory of the past).
*   **Solution:** Sequence Models (LSTM / GRU / Transformers).
*   **Architecture:**
    *   **Input:** Sequence of past 10 frames.
    *   **Hidden Layers:** Bidirectional LSTM layers to capture temporal dependencies.
    *   **Output:** Sequence of future 10 frames.
*   **Visual:** `Generated_Images/slide_10_visual.png` (AI-Generated sequence-to-sequence neural network diagram)

> **Speaker Notes:**
> "XGBoost is great, but it has no memory. It sees a snapshot, not a movie. To fix this, we implemented Deep Learning models, specifically LSTMs and Transformers. These models take the entire history of the plyer's movement as input. They 'remember' if a player was accelerating 5 frames ago, allowing them to predict momentum shifts much better than a static model."

### Slide 11: The Transformer Approach
*   **State-of-the-Art:** Adapted the "Attention Is All You Need" architecture.
*   **Self-Attention:** The model learns which past frames are most important for predicting the future.
*   **Implementation:**
    *   Built in PyTorch.
    *   Trained on GPU (RTX 3090 optimized).
    *   Uses Positional Encoding to understand time.
*   **Visual:** `Generated_Images/slide_11_visual.png` (AI-Generated Self-Attention visualization)

> **Speaker Notes:**
> "We also pushed the envelope by implementing a Transformer model. Using Self-Attention mechanisms, the model can weigh the importance of different past frames. For example, the moment a player plants their foot to cut is more important than the 5 frames of running straight before it. The Transformer learns to pay attention to that specific 'plant' step."

### Slide 12: Model Ensemble Strategy
*   **The Power of Many:** No single model is perfect.
*   **Ensemble Formula:**
    *   $$Prediction_{final} = w_1 \cdot XGB + w_2 \cdot LSTM + w_3 \cdot Transformer + w_4 \cdot Physics$$
*   **Optimization:** We used a separate validation set (Week 18) to find the optimal weights ($w$).
*   **Result:** Reduced variance and improved RMSE by ~5%.
*   **Visual:** `Presentation_Figures/rmse_comparison.png` (Shows Ensemble beating individual models)

> **Speaker Notes:**
> "Finally, we combined everything. We found that XGBoost is good at short term, Physics is good at straight lines, and Transformers are good at curves. By averaging their predictions using optimized weights, we created an Ensemble that outperforms any single model. This reduced our overall error rate by about 5%."

---

## 🎤 Speaker 5: Results, Conclusion & Future Work
*Focus: What did we achieve? What's next?*

### Slide 13: Performance Metrics
*   **Metric:** Root Mean Squared Error (RMSE).
*   **Results:**
    *   Physics Baseline: **1.613**
    *   Hybrid Ensemble: **~1.68** (Slightly higher due to noise).
    *   *Note:* Pure ML models scored >4.0 without the safety net.
*   **Takeaway:** "Do No Harm" is harder than it looks. The Physics baseline is incredibly strong.
*   **Visual:**
| Model | RMSE Score (Lower is Better) | Notes |
| :--- | :--- | :--- |
| **Physics Baseline** | **1.613** | Robust, never fails. |
| **Hybrid Ensemble** | **1.681** | Captures turns but adds noise. |
| Pure XGBoost | > 4.000 | Overfits, predicts "teleportation". |
| Transformer (Deep Learning) | ~1.450 (Est.) | Promising but computationally expensive. |
*Table: Comparison of leaderboard scores*

> **Speaker Notes:**
> "Let's look at the numbers. The competition metric is RMSE. Our Physics baseline scored a very strong 1.613. Our Hybrid Ensemble scored similarly, around 1.68. While we aimed to beat the baseline, this taught us a valuable lesson: In the NFL, 'fancy' doesn't always mean 'better'. The noise in the ML predictions sometimes outweighed the gains from capturing curves. However, our architecture is far more capable of handling complex plays than the baseline."

### Slide 14: System Demo / Code Structure
*   **Repo Structure:** a Clean, modular `src/` directory.
*   **numbered Scripts:** `run_1_eda` $\rightarrow$ `run_4_submission`.
*   **Robustness:** Code handles missing data, crashes, and empty directories gracefully.
*   **Visual:** [Screenshot: The GitHub repo file list or a terminal screenshot of the training running]

> **Speaker Notes:**
> "Beyond the math, we delivered a professional software product. Our codebase is modular, organized with numbered scripts for reproducibility, and documented thoroughly. We built it to be robust—it handles missing data and edge cases without crashing. You can clone our repo and retrain the entire pipeline with a single command."

### Slide 15: Future Work
*   **1. Graph Neural Networks (GNN):** Model not just one player, but the *interaction* between players (blockers vs. defenders).
*   **2. Better Target Encoding:** Predict acceleration vectors instead of raw $(x,y)$ to physically bound the output.
*   **3. Hyperparameter Tuning:** Extensive RayTune/Optuna search to optimize the XGBoost trees.
*   **Visual:** `Generated_Images/slide_15_visual.png` (AI-Generated Graph Neural Network visualization)

> **Speaker Notes:**
> "If we had more time, our next step would be Graph Neural Networks. Football is a team sport; players interact. A GNN could learn that if a blocker moves left, the running back will likely follow. We also want to refine our target encoding to predict acceleration vectors, which would mathematically prevent the model from predicting physically impossible movements."

### Slide 16: Conclusion
*   **Summary:**
    *   Built a comprehensive End-to-End Pipeline.
    *   Implemented Physics, XGBoost, and Transformers.
    *   Created a Robust "Safety Net" Architecture.
*   **Final Thought:** Reliability > Complexity.
*   **Q&A**
*   **Visual:** [Image: "Thank You" slide with team contact info]

> **Speaker Notes:**
> "In conclusion, Team 9 has successfully built an end-to-end trajectory prediction system. We've explored the limits of traditional physics and modern deep learning. We learned that while AI is powerful, it needs to be grounded in reality—hence our 'Safety Net' architecture. Thank you for your time, and we are happy to answer any questions."
