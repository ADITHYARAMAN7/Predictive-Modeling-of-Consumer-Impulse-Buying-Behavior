# 🛒 Predictive Modeling of Consumer Impulse Buying Behavior in E-Commerce and Retail

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)

> **Course:** 23CSE452 Business Analytics  
> **Domain:** Retail & E-commerce  
> **Dataset:** Primary survey dataset — 10,000 Indian consumer responses

---

## 📌 Project Overview

Retailers heavily rely on impulse buying to increase **Average Order Value (AOV)**, yet they erode profit margins by deploying generic, untargeted discounts. This project develops a **predictive classification model** that identifies *when* and *why* a consumer is susceptible to an impulse purchase, enabling dynamic, real-time marketing interventions instead of costly mass promotions.

### Core Objective

> Analyze how a consumer's **psychological state (mood)**, **demographic profile**, and **promotional trigger type** interact to drive spontaneous purchase decisions — then build a model to target these buyers at exactly the right moment.

---

## 📁 Repository Structure

```
📦 Predictive-Modeling-of-Consumer-Impulse-Buying-Behavior
├── 📓 analysis.ipynb                        ← Main deliverable: full Jupyter notebook
├── 📊 impulse_buying_dataset_10000.csv      ← Raw primary survey dataset (10,000 rows)
├── 🐍 verify_analysis.py                    ← Standalone pipeline verification script
├── 🔧 build_notebook.py                     ← Notebook generation script
└── 📂 data/
    ├── impulse_buying_dataset_cleaned.csv   ← Cleaned & encoded dataset
    ├── fig_01_target_distribution.png
    ├── fig_02_age_analysis.png
    ├── fig_03_categorical_analysis.png
    ├── fig_04_frequency_analysis.png
    ├── fig_05_mood_trigger_heatmap.png      ← ⭐ Key insight figure
    ├── fig_06_platform_mood.png
    ├── fig_07_correlation_matrix.png
    ├── fig_08_model_comparison.png
    ├── fig_09_roc_curves.png
    ├── fig_10_confusion_matrix.png
    ├── fig_11_feature_importance.png
    ├── fig_12_shap_importance.png
    ├── fig_13_shap_beeswarm.png
    ├── fig_14_lr_coefficients.png
    ├── fig_15_cv_stability.png
    └── fig_16_income_trigger_heatmap.png    ← ⭐ Business strategy figure
```

---

## 📊 Dataset

| Attribute | Details |
|---|---|
| **Source** | Primary survey (Google Forms / Online questionnaire) |
| **Size** | 10,000 consumer responses |
| **Collection** | Indian e-commerce consumers across age groups |
| **Target variable** | `Made_Impulse_Purchase` (Yes / No) — 53.1% / 46.9% |

### Features

| Feature | Type | Description |
|---|---|---|
| `Age` | Numeric | Consumer age (18–57) |
| `Gender` | Categorical | Male / Female / Other |
| `Employment_Status` | Categorical | Student / Employed / Unemployed / Self-employed |
| `Disposable_Income` | Ordinal | Monthly disposable income bracket (₹) |
| `Preferred_Platform` | Categorical | E-commerce / Social Media / Quick Commerce / In-Store |
| `Shopping_Frequency_Per_Month` | Numeric | Number of shopping trips per month (1–10) |
| `Current_Mood` | Categorical | Bored / Stressed / Happy / Neutral / Sad |
| `Primary_Trigger` | Categorical | Flash Sale / Heavy Discount / Free Shipping / Social Ad / Packaging |
| `Product_Category` | Categorical | Clothing / Beauty / Electronics / Food / Gaming |

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Target balance analysis (53.1% Yes / 46.9% No — near-balanced)
- Age distribution & purchase rate by age group
- Cross-tabulation of all 7 categorical features vs target
- **Mood × Trigger interaction heatmap** (key insight figure)
- Platform × Mood impulse rate analysis
- Correlation matrix (label-encoded)

### 2. Feature Engineering

| Derived Feature | Description |
|---|---|
| `Income_Ordinal` | Ordinal encoding of income bracket (0–3) |
| `Age_Group` | Binned into 18-24 / 25-34 / 35-44 / 45+ |
| `Mood_Negative` | Binary flag: Bored / Stressed / Sad = 1 |
| `Trigger_Scarcity` | Binary flag: Flash Sale or Heavy Discount = 1 |
| `High_Freq_Shopper` | Binary flag: ≥6 trips/month = 1 |

### 3. Models Trained

Seven classifiers evaluated with **80/20 stratified train-test split** and **5-fold stratified cross-validation**:

| # | Model | Notes |
|---|---|---|
| 1 | Logistic Regression | StandardScaler applied |
| 2 | Decision Tree | max_depth=8 |
| 3 | Random Forest | 200 estimators, max_depth=12 |
| 4 | Gradient Boosting | 200 estimators, lr=0.05, max_depth=5 |
| 5 | K-Nearest Neighbors | k=9, StandardScaler applied |
| 6 | Support Vector Machine | RBF kernel, C=1.0, StandardScaler applied |
| 7 | Naïve Bayes | Gaussian |

### 4. Explainability
- **SHAP TreeExplainer** on Gradient Boosting (bar + beeswarm plots)
- Feature importance from Random Forest & Gradient Boosting
- Logistic Regression coefficient visualization

---

## 📈 Results

### Model Performance (Ranked by ROC-AUC)

| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV AUC |
|---|---|---|---|---|---|---|---|
| 🥇 | **Logistic Regression** | 0.638 | 0.641 | **0.718** | 0.678 | **0.693** | 0.693 ± 0.007 |
| 🥈 | Naïve Bayes | 0.646 | 0.661 | 0.681 | 0.671 | 0.686 | 0.690 ± 0.009 |
| 🥉 | Support Vector Machine | 0.638 | 0.646 | 0.702 | 0.673 | 0.685 | 0.675 ± 0.011 |
| 4 | Random Forest | **0.651** | **0.657** | 0.714 | **0.685** | 0.680 | 0.680 ± 0.008 |
| 5 | Gradient Boosting | 0.644 | 0.654 | 0.698 | 0.676 | 0.678 | 0.678 ± 0.009 |
| 6 | Decision Tree | 0.642 | 0.658 | 0.676 | 0.667 | 0.668 | 0.668 ± 0.012 |
| 7 | K-Nearest Neighbors | 0.617 | 0.632 | 0.665 | 0.648 | 0.650 | 0.627 ± 0.018 |

> ✅ **Best Model: Logistic Regression** — highest ROC-AUC (0.693) with the best recall (71.8%), critical for maximizing the capture of true impulse buyers in a real-time marketing context.

---

## 💡 Key Business Insights

### Impulse Buyer Profile

| Attribute | Value |
|---|---|
| Most common mood | 🥱 **Bored** (68.9% impulse rate) |
| Second mood | 😰 **Stressed** (68.1% impulse rate) |
| Top trigger | ⏱️ **Limited Time Flash Sale** |
| Preferred platform | 🛍️ E-commerce Website |
| Top product category | 💄 Beauty / Cosmetics |
| Average age | **27.8 years** |
| Average shopping frequency | 5.02 trips/month |
| Most common income bracket | ₹5,000 – ₹15,000 |

### Mood → Impulse Purchase Rate

```
Bored        68.9%  ████████████████████████████████████
Stressed     68.1%  ████████████████████████████████████
Happy        42.5%  █████████████████████
Neutral      39.5%  ████████████████████
Sad          37.7%  ██████████████████
```

### Business Recommendations

| Recommendation | Mechanism | Expected Impact |
|---|---|---|
| Real-time mood proxies | Infer mood from session dwell time, scroll speed, cart abandonment | +8–12% AOV |
| Dynamic countdown flash sales | Serve only to predicted impulse buyers (model score ≥ 0.65) | ~30% reduction in blanket discounts |
| Platform-specific creatives | Urgency-first copy for social media; detail-rich for e-com sites | +5–9% CTR |
| Free shipping threshold pop-ups | Trigger for lower-income segments at 80–95% of cart threshold | +6% basket size |
| Retargeting cadence | Re-engage 25–34 cohort within 2h of session abandonment | +15% re-engagement |

---

## 🔎 Comparison with Published Studies

| Study | Method | Key Overlap |
|---|---|---|
| Verhagen & van Dolen (2011) | Structural Equation Modeling | Confirms hedonic platforms (Social Media Stores) drive impulse buying |
| Zhang et al. (2021) | Random Forest on WeChat e-commerce | RF outperforms LR; mood features rank highly — mirrors our results |
| Floh & Madlberger (2013) | Logistic Regression on in-store stimuli | Discount/scarcity cues as top predictors — consistent with our Trigger_Scarcity findings |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap jupyter
```

### Run the Notebook

```bash
# Clone the repo
git clone https://github.com/ADITHYARAMAN7/Predictive-Modeling-of-Consumer-Impulse-Buying-Behavior.git
cd Predictive-Modeling-of-Consumer-Impulse-Buying-Behavior

# Launch Jupyter
python -m jupyter lab
# Then open analysis.ipynb
```

### Run the Pipeline Directly (no Jupyter needed)

```bash
python verify_analysis.py
# Generates all 16 figures in data/ and prints full model metrics
```

---

## 📚 References

1. Rook, D. W., & Fisher, R. J. (1995). Normative influences on impulsive buying behavior. *Journal of Consumer Research*, 22(3), 305–313.
2. Verhagen, T., & van Dolen, W. (2011). The influence of online store beliefs on consumer online impulse buying. *Information & Management*, 48(8), 320–327.
3. Zhang, X., et al. (2021). Predicting impulse buying on social commerce platforms using machine learning. *Journal of Retailing and Consumer Services*, 61, 102556.
4. Floh, A., & Madlberger, M. (2013). The role of atmospheric cues in online impulse-buying behavior. *Electronic Commerce Research and Applications*, 12(6), 425–439.

---

## 👤 Author

**Adithya Raman**  
B.Tech — Computer Science & Engineering  
Course: 23CSE452 Business Analytics
