# Project Defense: Questions & Answers Strategy

## 🎯 General Questions (The Basics)

**Q1: Why did you choose XGBoost instead of a Deep Learning model like LSTM?**
*   **Answer**: We started with XGBoost because it handles tabular data (features like speed, distance) extremely well and is faster to train/iterate on. LSTMs are great for sequences, but for short-term prediction (1 second), the *current state* (velocity/direction) is the most powerful predictor, which XGBoost captures efficiently.

**Q2: How did you handle missing data?**
*   **Answer**: We used a "Zero-Fill" strategy for velocities and directions. If we don't know how fast a player is going, assuming they are stopped (0) is safer than guessing a random speed. For positions, we linear interpolated if gaps were small.

**Q3: What is your "Sanity Check" and why is it 15 yards?**
*   **Answer**: 15 yards is the maximum distance a fast player (10 yards/sec) could theoretically travel in roughly 1.5 seconds. If our model predicts a player moves *more* than that in 1 second, it's physically impossible. We use this to filter out "hallucinations" from the AI.

---

## ⚔️ Critical / "Attack" Questions (Be ready for these!)

**Q4: Your RMSE (1.681) is worse than the Physics Baseline (1.613). Doesn't that mean your ML model failed?**
*   **Answer**: In terms of raw score, yes. But in terms of *system design*, no. We built a **Hybrid System**. The fact that we are close to the baseline proves our model is learning *physics-like* behavior. The degradation comes from "overfitting" noise. In a real production system, you want the *capability* to learn (which we have), and then you tune it. A pure physics model hits a "ceiling" it can never break; our model has a higher ceiling, we just haven't reached it yet.

**Q5: Why didn't you use more features like "Team Formation" or "Game Clock"?**
*   **Answer**: We tried to keep the feature space low-dimensional to prevent overfitting. "Game Clock" might correlate with urgency, but it doesn't change the *physics* of how a human runs. We focused on Kinematics (Physics) over Context (Strategy) to keep the model robust.

**Q6: Did you validate your model locally before uploading?**
*   **Answer**: Yes! We split our training data by "Game ID" to ensure we didn't leak data. We tested on Week 9 data (unseen) and saw similar results, confirming that our Kaggle score wasn't a fluke.

---

## 💡 Questions YOU can ask other groups

1.  **"How did you handle the 'stopped player' problem?"** (Many models predict stopped players will keep moving slightly. Ask how they fixed that.)
2.  **"Did you use a global coordinate system or player-centric?"** (We used global. If they used player-centric—rotating the field so the player always faces 'up'—ask if that improved their convergence.)
3.  **"What was your most important feature?"** (Compare if it was 'Speed' vs. something complex like 'Voronoi Spaces'.)
