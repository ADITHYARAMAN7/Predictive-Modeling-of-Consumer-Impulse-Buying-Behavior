"""
Verification script: runs the complete analysis pipeline and saves all figures.
Execute with: python verify_analysis.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script mode
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, ConfusionMatrixDisplay
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'figure.dpi': 120, 'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False})
PALETTE = ['#4361EE', '#F72585', '#3A0CA3', '#7209B7', '#4CC9F0', '#4895EF', '#560BAD']
sns.set_palette(PALETTE)
os.makedirs('data', exist_ok=True)

print("=" * 60)
print("  BA Case Study - Impulse Buying Prediction")
print("  Verification Run")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n[1] Loading data...")
df_raw = pd.read_csv('impulse_buying_dataset_10000.csv')
df = df_raw.copy()
print(f"    Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"    Columns: {list(df.columns)}")
vc = df['Made_Impulse_Purchase'].value_counts()
print(f"    Target — Yes: {vc.get('Yes',0):,} ({vc.get('Yes',0)/len(df)*100:.1f}%) | No: {vc.get('No',0):,} ({vc.get('No',0)/len(df)*100:.1f}%)")

# ── Fig 01: Target Distribution ───────────────────────────────────────────────
print("\n[2] Generating EDA figures...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
colors = ['#F72585', '#4361EE']
bars = axes[0].bar(vc.index, vc.values, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, vc.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=11, fontweight='bold')
axes[0].set_title('Impulse Purchase Frequency', fontsize=13, fontweight='bold', pad=15)
axes[0].set_ylabel('Count')
axes[0].set_ylim(0, max(vc.values) * 1.18)
wedges, texts, autotexts = axes[1].pie(
    vc.values, labels=vc.index, colors=colors, autopct='%1.1f%%',
    startangle=140, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
for t in autotexts:
    t.set_fontweight('bold'); t.set_fontsize(12)
axes[1].set_title('Class Distribution', fontsize=13, fontweight='bold', pad=15)
plt.suptitle('Target Variable: Made_Impulse_Purchase', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('data/fig_01_target_distribution.png', bbox_inches='tight')
plt.close()
print("    fig_01 saved")

# ── Fig 02: Age Analysis ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for label, color in zip(['Yes', 'No'], ['#F72585', '#4361EE']):
    subset = df[df['Made_Impulse_Purchase'] == label]['Age']
    axes[0].hist(subset, bins=20, alpha=0.65, label=label, color=color, edgecolor='white')
axes[0].set_title('Age Distribution by Impulse Purchase', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Age'); axes[0].set_ylabel('Count')
axes[0].legend(title='Impulse Purchase')
age_rate = df.groupby('Age')['Made_Impulse_Purchase'].apply(lambda x: (x=='Yes').mean() * 100)
axes[1].plot(age_rate.index, age_rate.values, color='#4361EE', lw=2.5)
axes[1].fill_between(age_rate.index, age_rate.values, alpha=0.15, color='#4361EE')
axes[1].axhline(50, color='#F72585', ls='--', lw=1.5, label='50% baseline')
axes[1].set_title('Impulse Purchase Rate by Age', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Age'); axes[1].set_ylabel('Purchase Rate (%)')
axes[1].legend()
plt.tight_layout()
plt.savefig('data/fig_02_age_analysis.png', bbox_inches='tight')
plt.close()
print("    fig_02 saved")

# ── Fig 03: Categorical Analysis ──────────────────────────────────────────────
cat_cols = ['Gender', 'Employment_Status', 'Current_Mood', 'Primary_Trigger',
            'Product_Category', 'Preferred_Platform', 'Disposable_Income']
fig, axes = plt.subplots(4, 2, figsize=(16, 22))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    ct = pd.crosstab(df[col], df['Made_Impulse_Purchase'], normalize='index') * 100
    ct[['Yes', 'No']].plot(kind='barh', stacked=True, ax=axes[i],
                            color=['#F72585', '#4361EE'], edgecolor='white', linewidth=0.8)
    axes[i].set_title(f'{col.replace("_", " ")} vs Impulse Purchase', fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Proportion (%)'); axes[i].set_ylabel('')
    axes[i].legend(title='Impulse Buy', loc='lower right')
    axes[i].xaxis.set_major_formatter(mticker.PercentFormatter())
for j in range(len(cat_cols), len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Categorical Feature Breakdown by Impulse Purchase', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('data/fig_03_categorical_analysis.png', bbox_inches='tight')
plt.close()
print("    fig_03 saved")

# ── Fig 04: Shopping Frequency ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
freq_rate = df.groupby('Shopping_Frequency_Per_Month')['Made_Impulse_Purchase'].apply(
    lambda x: (x=='Yes').mean() * 100)
axes[0].bar(freq_rate.index, freq_rate.values, color='#7209B7', edgecolor='white', linewidth=0.8)
axes[0].axhline(50, color='#F72585', ls='--', lw=1.5, label='50% baseline')
axes[0].set_title('Impulse Purchase Rate by Shopping Frequency', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Trips per Month'); axes[0].set_ylabel('Purchase Rate (%)'); axes[0].legend()
sns.boxplot(data=df, x='Made_Impulse_Purchase', y='Shopping_Frequency_Per_Month',
            palette=['#4361EE', '#F72585'], ax=axes[1])
axes[1].set_title('Shopping Frequency Distribution by Outcome', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Made Impulse Purchase'); axes[1].set_ylabel('Trips per Month')
plt.tight_layout()
plt.savefig('data/fig_04_frequency_analysis.png', bbox_inches='tight')
plt.close()
print("    fig_04 saved")

# ── Fig 05: Mood x Trigger Heatmap ───────────────────────────────────────────
pivot = df.groupby(['Current_Mood', 'Primary_Trigger'])['Made_Impulse_Purchase'].apply(
    lambda x: (x == 'Yes').mean() * 100).unstack()
fig, ax = plt.subplots(figsize=(11, 5))
cmap_ht = LinearSegmentedColormap.from_list('custom', ['#4361EE', '#F72585'], N=256)
sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap_ht, linewidths=0.5,
            linecolor='white', ax=ax, annot_kws={'size': 10, 'weight': 'bold'},
            cbar_kws={'label': 'Impulse Purchase Rate (%)'})
ax.set_title('Impulse Purchase Rate: Mood x Promotional Trigger', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Primary Trigger', fontsize=11); ax.set_ylabel('Current Mood', fontsize=11)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig('data/fig_05_mood_trigger_heatmap.png', bbox_inches='tight')
plt.close()
print("    fig_05 saved")

# ── Fig 06: Platform x Mood ───────────────────────────────────────────────────
pivot2 = df.groupby(['Preferred_Platform', 'Current_Mood'])['Made_Impulse_Purchase'].apply(
    lambda x: (x == 'Yes').mean() * 100).unstack()
fig, ax = plt.subplots(figsize=(10, 4))
pivot2.plot(kind='bar', ax=ax, color=PALETTE[:5], edgecolor='white', linewidth=0.8, width=0.75)
ax.set_title('Impulse Purchase Rate by Platform & Mood', fontsize=13, fontweight='bold')
ax.set_xlabel('Preferred Platform'); ax.set_ylabel('Purchase Rate (%)')
ax.legend(title='Mood', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('data/fig_06_platform_mood.png', bbox_inches='tight')
plt.close()
print("    fig_06 saved")

# ── Fig 07: Correlation Matrix ────────────────────────────────────────────────
df_enc_corr = df.copy()
for col in df_enc_corr.select_dtypes('object').columns:
    df_enc_corr[col] = LabelEncoder().fit_transform(df_enc_corr[col])
fig, ax = plt.subplots(figsize=(9, 7))
corr = df_enc_corr.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap_div = LinearSegmentedColormap.from_list('div', ['#4361EE', '#ffffff', '#F72585'], N=256)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=cmap_div, center=0,
            linewidths=0.5, linecolor='#eeeeee', ax=ax, annot_kws={'size': 9})
ax.set_title('Correlation Matrix (Label-Encoded Features)', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('data/fig_07_correlation_matrix.png', bbox_inches='tight')
plt.close()
print("    fig_07 saved")

# ── Feature Engineering ───────────────────────────────────────────────────────
print("\n[3] Feature engineering...")
income_order = ['\u20b92,000 - \u20b95,000', '\u20b95,000 - \u20b915,000', 'Above \u20b915,000']
# Robust lookup
income_map = {}
for val in df['Disposable_Income'].unique():
    val_str = str(val)
    if 'Under' in val_str or '2,000' in val_str and '5,000' not in val_str:
        income_map[val] = 0
    elif '2,000' in val_str and '5,000' in val_str:
        income_map[val] = 1
    elif '5,000' in val_str and '15,000' in val_str:
        income_map[val] = 2
    elif 'Above' in val_str:
        income_map[val] = 3
df['Income_Ordinal'] = df['Disposable_Income'].map(income_map)
df['Age_Group'] = pd.cut(df['Age'], bins=[17, 24, 34, 44, 60], labels=['18-24', '25-34', '35-44', '45+'])
df['Mood_Negative'] = df['Current_Mood'].isin(['Stressed', 'Bored', 'Sad']).astype(int)
df['Trigger_Scarcity'] = df['Primary_Trigger'].isin(['Limited Time Flash Sale', 'Heavy Discount']).astype(int)
df['High_Freq_Shopper'] = (df['Shopping_Frequency_Per_Month'] >= 6).astype(int)
print(f"    Income_Ordinal values: {sorted(df['Income_Ordinal'].unique())}")
print(f"    Age_Group values: {sorted(df['Age_Group'].astype(str).unique())}")

features_to_encode = ['Gender', 'Employment_Status', 'Preferred_Platform',
                       'Current_Mood', 'Primary_Trigger', 'Product_Category', 'Age_Group']
df_model = pd.get_dummies(df, columns=features_to_encode, drop_first=False)
df_model['Target'] = (df_model['Made_Impulse_Purchase'] == 'Yes').astype(int)
df_model.drop(columns=['Made_Impulse_Purchase', 'Age', 'Disposable_Income'], inplace=True)
print(f"    Model shape: {df_model.shape} | Features: {df_model.shape[1]-1}")

X = df_model.drop(columns=['Target'])
y = df_model['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"    Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ── Model Training ────────────────────────────────────────────────────────────
print("\n[4] Training classifiers...")
classifiers = {
    'Logistic Regression'    : LogisticRegression(max_iter=500, random_state=42),
    'Decision Tree'          : DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest'          : RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    'Gradient Boosting'      : GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    'K-Nearest Neighbors'    : KNeighborsClassifier(n_neighbors=9),
    'Support Vector Machine' : SVC(probability=True, random_state=42, C=1.0),
    'Naive Bayes'            : GaussianNB(),
}
SCALE_NEEDED = {'Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Machine'}
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in classifiers.items():
    Xtr = X_train_sc if name in SCALE_NEEDED else X_train
    Xte = X_test_sc  if name in SCALE_NEEDED else X_test
    clf.fit(Xtr, y_train)
    y_pred   = clf.predict(Xte)
    y_proba  = clf.predict_proba(Xte)[:, 1]
    cv_sc    = cross_val_score(clf, Xtr, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    results[name] = {
        'Accuracy'    : accuracy_score(y_test, y_pred),
        'Precision'   : precision_score(y_test, y_pred),
        'Recall'      : recall_score(y_test, y_pred),
        'F1-Score'    : f1_score(y_test, y_pred),
        'ROC-AUC'     : roc_auc_score(y_test, y_proba),
        'CV AUC Mean' : cv_sc.mean(),
        'CV AUC Std'  : cv_sc.std(),
    }
    print(f"    {name:<25} | Acc={results[name]['Accuracy']:.3f} | AUC={results[name]['ROC-AUC']:.3f} | CV={results[name]['CV AUC Mean']:.3f}±{results[name]['CV AUC Std']:.3f}")

results_df = pd.DataFrame(results).T.sort_values('ROC-AUC', ascending=False)

print("\n  === RANKED MODEL PERFORMANCE ===")
print(results_df[['Accuracy','Precision','Recall','F1-Score','ROC-AUC','CV AUC Mean']].round(4).to_string())

# ── Fig 08: Model Comparison ──────────────────────────────────────────────────
print("\n[5] Generating model evaluation figures...")
metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(results_df))
width = 0.16
colors_m = ['#4361EE', '#F72585', '#7209B7', '#3A0CA3', '#4CC9F0']
for i, (metric, color) in enumerate(zip(metrics_list, colors_m)):
    ax.bar(x + i * width, results_df[metric], width=width - 0.01,
           label=metric, color=color, edgecolor='white', linewidth=0.7)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(results_df.index, rotation=18, ha='right', fontsize=9)
ax.set_ylabel('Score'); ax.set_ylim(0, 1.12)
ax.set_title('Classifier Performance Comparison', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper right', ncol=5, fontsize=8)
plt.tight_layout()
plt.savefig('data/fig_08_model_comparison.png', bbox_inches='tight')
plt.close()
print("    fig_08 saved")

# ── Fig 09: ROC Curves ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
for (name, clf), color in zip(classifiers.items(), PALETTE):
    Xte = X_test_sc if name in SCALE_NEEDED else X_test
    y_proba = clf.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, lw=2, color=color, label=f'{name} (AUC={auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
ax.fill_between([0, 1], [0, 1], alpha=0.04, color='gray')
ax.set_xlabel('False Positive Rate', fontsize=12); ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - All Classifiers', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=8.5)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig('data/fig_09_roc_curves.png', bbox_inches='tight')
plt.close()
print("    fig_09 saved")

# ── Fig 10: Confusion Matrix ──────────────────────────────────────────────────
best_name   = results_df.index[0]
best_clf    = classifiers[best_name]
Xte_best    = X_test_sc if best_name in SCALE_NEEDED else X_test
y_pred_best = best_clf.predict(Xte_best)
print(f"\n  Best Model: {best_name}")
print(classification_report(y_test, y_pred_best, target_names=['No Impulse', 'Impulse Buy']))
fig, ax = plt.subplots(figsize=(5, 4))
cm   = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Impulse', 'Impulse Buy'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix - {best_name}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('data/fig_10_confusion_matrix.png', bbox_inches='tight')
plt.close()
print("    fig_10 saved")

# ── Fig 11: Feature Importance ────────────────────────────────────────────────
print("\n[6] Feature importance & explainability...")
rf_clf = classifiers['Random Forest']
gb_clf = classifiers['Gradient Boosting']

def plot_feature_importance(clf, X, title, ax, top_n=18):
    imp = pd.Series(clf.feature_importances_, index=X.columns)
    imp = imp.nlargest(top_n).sort_values()
    colors_bar = plt.cm.plasma(np.linspace(0.3, 0.9, len(imp)))
    ax.barh(imp.index, imp.values, color=colors_bar, edgecolor='white')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Importance')

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
plot_feature_importance(rf_clf, X_train, 'Random Forest - Top 18 Features', axes[0])
plot_feature_importance(gb_clf, X_train, 'Gradient Boosting - Top 18 Features', axes[1])
plt.suptitle('Feature Importance Comparison', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('data/fig_11_feature_importance.png', bbox_inches='tight')
plt.close()
print("    fig_11 saved")

# ── Fig 12/13: SHAP ───────────────────────────────────────────────────────────
if SHAP_AVAILABLE:
    explainer = shap.TreeExplainer(gb_clf)
    shap_vals = explainer.shap_values(X_test)
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_vals, X_test, plot_type='bar', max_display=15, show=False, color='#4361EE')
    plt.title('SHAP Feature Importance - Gradient Boosting', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/fig_12_shap_importance.png', bbox_inches='tight')
    plt.close()
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_vals, X_test, max_display=15, show=False)
    plt.title('SHAP Beeswarm - Feature Impact Direction', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/fig_13_shap_beeswarm.png', bbox_inches='tight')
    plt.close()
    print("    fig_12/13 (SHAP) saved")
else:
    print("    SHAP not available — skipping figs 12/13")

# ── Fig 14: LR Coefficients ───────────────────────────────────────────────────
lr_clf  = classifiers['Logistic Regression']
coef_df = pd.Series(lr_clf.coef_[0], index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(10, 8))
colors_coef = ['#F72585' if v > 0 else '#4361EE' for v in coef_df.values]
ax.barh(coef_df.index, coef_df.values, color=colors_coef, edgecolor='white', linewidth=0.6)
ax.axvline(0, color='black', lw=1)
ax.set_title('Logistic Regression Coefficients\n(Pink = higher impulse prob | Blue = lower)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig('data/fig_14_lr_coefficients.png', bbox_inches='tight')
plt.close()
print("    fig_14 saved")

# ── Fig 15: CV Stability ──────────────────────────────────────────────────────
print("\n[7] Cross-validation stability...")
cv_data = {}
for name, clf in classifiers.items():
    Xtr    = X_train_sc if name in SCALE_NEEDED else X_train
    scores = cross_val_score(clf, Xtr, y_train, cv=cv, scoring='f1', n_jobs=-1)
    cv_data[name] = scores
cv_df = pd.DataFrame(cv_data)
fig, ax = plt.subplots(figsize=(12, 5))
bp = ax.boxplot([cv_df[c].values for c in cv_df.columns],
                patch_artist=True, medianprops={'color': 'white', 'lw': 2.5})
for patch, color in zip(bp['boxes'], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.85)
ax.set_xticklabels(cv_df.columns, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('F1-Score (5-Fold CV)')
ax.set_title('Cross-Validation Stability - All Classifiers', fontsize=13, fontweight='bold')
ax.axhline(cv_df.mean().mean(), color='gray', ls='--', lw=1.2, label='Overall mean')
ax.legend()
plt.tight_layout()
plt.savefig('data/fig_15_cv_stability.png', bbox_inches='tight')
plt.close()
print("    fig_15 saved")

# ── Fig 16: Income x Trigger Heatmap ─────────────────────────────────────────
print("\n[8] Business intelligence...")
income_label_map = {}
for val in df['Disposable_Income'].unique():
    v = str(val)
    if 'Under' in v:
        income_label_map[val] = 'Under Rs2K'
    elif '2,000' in v and '5,000' in v:
        income_label_map[val] = 'Rs2K-5K'
    elif '5,000' in v and '15,000' in v:
        income_label_map[val] = 'Rs5K-15K'
    elif 'Above' in v:
        income_label_map[val] = 'Above Rs15K'
income_labels_ord = ['Under Rs2K', 'Rs2K-5K', 'Rs5K-15K', 'Above Rs15K']
df['Income_Label'] = pd.Categorical(
    df['Disposable_Income'].map(income_label_map),
    categories=income_labels_ord, ordered=True)
pivot3 = df.groupby(['Income_Label', 'Primary_Trigger'])['Made_Impulse_Purchase'].apply(
    lambda x: (x == 'Yes').mean() * 100).unstack()
fig, ax = plt.subplots(figsize=(11, 4))
cmap2 = LinearSegmentedColormap.from_list('g', ['#4CC9F0', '#7209B7'], N=256)
sns.heatmap(pivot3, annot=True, fmt='.1f', cmap=cmap2, linewidths=0.5,
            linecolor='white', ax=ax, annot_kws={'size': 10, 'weight': 'bold'},
            cbar_kws={'label': 'Impulse Purchase Rate (%)'})
ax.set_title('Impulse Purchase Rate: Income Bracket x Promotional Trigger',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Primary Trigger'); ax.set_ylabel('Disposable Income Bracket')
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig('data/fig_16_income_trigger_heatmap.png', bbox_inches='tight')
plt.close()
print("    fig_16 saved")

# ── Segment Profiling ─────────────────────────────────────────────────────────
high_imp = df[df['Made_Impulse_Purchase'] == 'Yes']
print("\n  IMPULSE BUYER PROFILE:")
fields = [
    ('Most common mood   ', 'Current_Mood'),
    ('Top trigger        ', 'Primary_Trigger'),
    ('Preferred platform ', 'Preferred_Platform'),
    ('Top product cat    ', 'Product_Category'),
    ('Most common income ', 'Disposable_Income'),
]
for label, col in fields:
    print(f"    {label}: {high_imp[col].mode()[0]}")
print(f"    Avg shopping freq  : {high_imp['Shopping_Frequency_Per_Month'].mean():.2f} trips/month")
print(f"    Avg age            : {high_imp['Age'].mean():.1f} years")

print("\n  Mood -> Impulse Purchase Rate:")
mood_rate = df.groupby('Current_Mood')['Made_Impulse_Purchase'].apply(
    lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False)
for mood, rate in mood_rate.items():
    bar = '#' * int(rate / 2)
    print(f"    {mood:<12} {rate:.1f}%  {bar}")

# ── Save cleaned dataset ──────────────────────────────────────────────────────
print("\n[9] Saving cleaned dataset...")
df_raw.to_csv('data/impulse_buying_dataset_cleaned.csv', index=False, encoding='utf-8-sig')

# ── List all saved files ──────────────────────────────────────────────────────
print("\n[10] Summary of generated files:")
for f in sorted(os.listdir('data')):
    fpath = os.path.join('data', f)
    size  = os.path.getsize(fpath)
    print(f"    {f:50s} {size/1024:.1f} KB")

print("\n" + "=" * 60)
print("  All verification complete! Notebook is ready to use.")
print("=" * 60)
