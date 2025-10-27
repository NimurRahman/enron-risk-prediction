# 🏆 Enron Employee Risk Prediction Dashboard

> 97% accurate ML model predicting high-risk employees using email network analysis

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)

## 📊 Project Overview

- **Accuracy:** 97% (0.970 PR-AUC)
- **Dataset:** 261K predictions across 21K employees
- **Models:** XGBoost, Random Forest, Logistic Regression
- **Explainability:** SHAP values for all predictions

## 🚀 Features

### 9 Interactive Dashboard Tabs:
1. 📈 Activity Trends
2. ⚠️ Risk Over Time
3. 🎯 Top Risky Entities
4. 🔍 Feature Importance
5. 📋 Predictions Table
6. 🎲 Risk Simulator (What-If Analysis)
7. 🔗 Clustering Analysis
8. 📊 Insights & Reports
9. 🤖 AI Agent (Natural Language)

## 🛠️ Installation
```bash
# Clone repository
git clone https://github.com/NimurRahman/enron-risk-prediction.git
cd enron-risk-prediction

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run streamlit_sna_dashboard.py
```

## 📦 Project Structure
```
enron_modeling/
├── data/              # Input data
├── outputs/           # Predictions & results
├── models/            # Trained models
├── dashboard/         # Streamlit app
│   ├── tabs/          # 9 dashboard tabs
│   └── components/    # UI components
├── scripts/           # Training pipeline
└── shap_analysis/     # SHAP visualizations
```

## 🎓 Training Pipeline
```bash
cd scripts
python 01_eda.py
python 02_prepare_modeling_data.py
# ... (run scripts 03-11 in order)
```

## 📊 Model Performance

| Model | PR-AUC | ROC-AUC |
|-------|--------|---------|
| XGBoost | 0.970 | 0.997 |
| Random Forest | 0.951 | 0.992 |
| Logistic Regression | 0.935 | 0.984 |

## 👤 Author

**Nimur Rahman**
- GitHub: [@NimurRahman](https://github.com/NimurRahman)

## 📝 License

MIT License

---

⭐ Star this repo if you find it useful!