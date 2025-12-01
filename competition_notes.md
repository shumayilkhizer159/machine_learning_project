# NFL Big Data Bowl 2026 - Competition Understanding

## Competition Objective
Predict NFL player movement during video frames after the ball is thrown until it's caught or ruled incomplete.

## Task Description
- **Input**: Tracking data BEFORE the pass is thrown + targeted receiver + ball landing location
- **Output**: Predict x, y positions for each player for each frame while ball is in the air
- **Tracking Rate**: 10 frames per second
- **Example**: If ball is in air for 2.5 seconds → predict 25 frames of location data

## Evaluation Metric
**Root Mean Squared Error (RMSE)** between predicted and observed positions:

```
RMSE = sqrt( (1/2N) * Σ((x_true - x_pred)² + (y_true - y_pred)²) )
```

Where N is the total number of predictions across all players and frames.

## Key Competition Details
1. **Competition Type**: Code competition (must submit via Kaggle Notebooks)
2. **Submission Limits**: 5 submissions per day, select 2 final submissions
3. **Runtime**: Max 9 hours (CPU or GPU)
4. **Internet**: Disabled during submission
5. **External Data**: Allowed if publicly available and free
6. **Team Size**: Max 4 members

## Data Structure

### Input Files (train/input_2023_w[01-18].csv)
Key features:
- `game_id`, `play_id`, `nfl_id` - Identifiers
- `frame_id` - Frame number (starts at 1 for each play)
- `x`, `y` - Player position on field
- `s`, `a` - Speed and acceleration
- `o`, `dir` - Orientation and direction angles
- `player_position`, `player_side`, `player_role` - Player info
- `ball_land_x`, `ball_land_y` - Ball landing position (KEY FEATURE)
- `num_frames_output` - Number of frames to predict
- `player_to_predict` - Boolean flag for which players to score

### Output Files (train/output_2023_w[01-18].csv)
Targets to predict:
- `game_id`, `play_id`, `nfl_id`, `frame_id` - Identifiers
- `x`, `y` - Player positions (TARGETS)

## Competition Timeline
- **Training Phase**: September 25 - December 3, 2025
- **Forecasting Phase**: December 4, 2025 - January 5, 2026
  - Leaderboard updated after each week's NFL games
  - Test set = future NFL games not yet played
- **Results**: January 6, 2026

## Prizes
- 1st Place: $25,000
- 2nd Place: $15,000
- 3rd Place: $10,000
- Total: $50,000

## Key Insights for Modeling
1. **Sequential prediction**: Need to predict multiple future frames (time series)
2. **Multiple players**: Track multiple players simultaneously on each play
3. **Spatial context**: Ball landing location is known → players move toward it
4. **Player roles matter**: Targeted receiver vs defensive coverage vs other routes
5. **Physics-based**: Speed, acceleration, direction should inform trajectory
6. **Field constraints**: x ∈ [0, 120] yards, y ∈ [0, 53.3] yards

## Modeling Approach Ideas
1. **Time Series Models**: LSTM, GRU, Transformer for sequential prediction
2. **Physics-Informed**: Use velocity/acceleration to project trajectories
3. **Multi-Agent Models**: Model interactions between players
4. **Graph Neural Networks**: Players as nodes, relationships as edges
5. **Ensemble Methods**: Combine multiple approaches

## Data Provided
- You uploaded: Public leaderboard CSV (appears to be submission format sample)
- Need to download: Full training data from Kaggle (input/output files for weeks 1-18)
