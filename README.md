# Waze User Churn Prediction
## MSIN0097 Predictive Analytics — Individual Coursework

### Problem
Binary classification: predict whether a Waze user will churn or be retained, using monthly behavioural data. The target variable is the `label` column (1 = churned, 0 = retained).

### Repo Structure
```
waze-churn/
├── MSIN0097_Predictive_Analytics_Notebook.ipynb  # Main analysis notebook (all 6 steps)
├── src/
│   └── data_loader.py          # load_raw_data() and load_and_prepare() pipeline
├── data/
│   └── waze_dataset.csv        # Place downloaded dataset here (see Data Access below)
├── figures/                    # Auto-generated plots saved during notebook run
├── models/
│   ├── lr_final.pkl            # Saved tuned Logistic Regression model
│   └── preprocessor.pkl        # Saved preprocessing pipeline
├── AI agent Log.xlsx           # Full agent interaction log and decision register
├── requirements.txt
└── README.md
```

### Data Access
The dataset is publicly available on Kaggle:
1. Go to: https://www.kaggle.com/datasets/anaghapaul/waze-user-churn-data
2. Download `waze_dataset.csv`
3. Place it at `./data/waze_dataset.csv`

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Instructions
```bash
# Launch Jupyter
jupyter notebook MSIN0097_Predictive_Analytics_Notebook.ipynb

# Run all cells top-to-bottom (Kernel → Restart & Run All)
# Figures will be saved to ./figures/
# Final models will be saved to ./models/
```

### Agent Tooling
This project used **Claude (claude.ai)** as an agentic collaborator. See `AI agent Log.xlsx` for the full interaction log and decision register (accepted / modified / rejected).

### Key Results

| Model | ROC-AUC | PR-AUC | F1 (churn) |
|---|---|---|---|
| Majority class baseline | 0.500 | — | 0.000 |
| MLP | 0.659 | 0.297 | 0.270 |
| XGBoost (baseline CV) | 0.688 | 0.318 | 0.333 |
| Random Forest | 0.727 | 0.352 | 0.124 |
| Logistic Regression (baseline CV) | 0.758 | 0.392 | 0.441 |
| XGBoost (tuned) | 0.731 | 0.383 | 0.424 |
| **Logistic Regression (tuned)** | **0.743** | **0.377** | **0.425** |

**Final model:** Tuned Logistic Regression (C=0.01, penalty=l2, solver=liblinear)
