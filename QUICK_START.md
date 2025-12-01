# Quick Start Guide - NFL Big Data Bowl 2026

## 🚀 Get Started in 3 Steps

### Step 1: Setup Environment (5 minutes)
```bash
# Clone/navigate to project
cd nfl_project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Data (10 minutes)
```bash
# Setup Kaggle credentials
mkdir -p ~/.kaggle
# Place your kaggle.json in ~/.kaggle/

# Download competition data
kaggle competitions download -c nfl-big-data-bowl-2026-prediction -p data/
unzip data/nfl-big-data-bowl-2026-prediction.zip -d data/
```

### Step 3: Choose Your Path

#### Option A: Quick Kaggle Submission (Immediate)
1. Open `notebooks/kaggle_submission.py`
2. Copy all code
3. Create new Kaggle notebook
4. Paste and run
5. Submit to leaderboard

**Expected Score**: RMSE ~1.5-2.2 (Top 30-40%)

#### Option B: Train Advanced Models (15-20 hours with GPU)
```bash
# Run full training pipeline
python src/train_gpu.py
```

**Expected Score**: RMSE ~0.9-1.4 (Top 10-20%)

## 📁 Key Files

| File | Purpose |
|------|---------|
| `README.md` | Complete project documentation |
| `PROJECT_SUMMARY.md` | Quick overview of everything |
| `docs/local_training_guide.md` | Detailed GPU training instructions |
| `docs/q_and_a_complete.md` | 23 Q&A for academic defense |
| `notebooks/kaggle_submission.py` | Ready-to-submit Kaggle code |
| `src/train_gpu.py` | Main training script for all models |

## 🎯 For Academic Defense

**Read these in order:**
1. `PROJECT_SUMMARY.md` - Get the big picture
2. `README.md` - Understand the methodology
3. `docs/q_and_a_complete.md` - Prepare for questions
4. Review model code in `src/` - Know the implementations

## 🔧 Troubleshooting

**GPU Out of Memory?**
- Edit `src/train_gpu.py`
- Change `batch_size: 64` to `batch_size: 32` or `16`

**Training Too Slow?**
- Edit `src/train_gpu.py`
- Change `weeks_to_train: list(range(1, 19))` to `weeks_to_train: [1, 2, 3]`

**Need Help?**
- Check `docs/local_training_guide.md` for detailed instructions
- Review error messages carefully
- Ensure all dependencies are installed

## 📊 What Models Are Included?

1. **Physics Baseline** - Fast, interpretable (no training needed)
2. **XGBoost** - Gradient boosting (2-4 hours training)
3. **LSTM** - Advanced RNN with attention (4-6 hours)
4. **GRU** - Faster RNN alternative (3-5 hours)
5. **Transformer** - State-of-the-art (5-8 hours)
6. **Ensemble** - Combines all models for best results

## ✅ Checklist for Success

- [ ] Environment setup complete
- [ ] Data downloaded
- [ ] Ran EDA: `python src/eda_analysis.py`
- [ ] Made Kaggle submission (baseline)
- [ ] Started GPU training (if available)
- [ ] Read Q&A document
- [ ] Prepared academic defense
- [ ] Documented results

---

**Good luck! 🏈🏆**
