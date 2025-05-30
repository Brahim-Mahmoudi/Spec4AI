import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('Spec4AI/Evaluation/Figures_Precision/precision_recall_par_regle_et_outil.csv')

# List of rules to display
selected_rules = ["R5", "R11", "R21", "R22", "R2", "R14", "R6", "R4", "R19", "R7"]

# Filter to keep only selected rules (for all tools)
data = data[data['Rule'].isin(selected_rules)]

# Ensure numeric types
data['Precision'] = pd.to_numeric(data['Precision'], errors='coerce')
data['Recall'] = pd.to_numeric(data['Recall'], errors='coerce')

# Compute F1-score
data['F1-Score'] = 2 * (data['Precision'] * data['Recall']) / (data['Precision'] + data['Recall'])
data['F1-Score'] = data['F1-Score'].fillna(0.0)  # Replace NaN (from 0/0) by 0.0

# Pivot for heatmap (rules as rows, tools as columns)
heatmap_data = data.pivot(index='Rule', columns='Tool', values='F1-Score')

# Prepare annotation matrix (force display of 0.00)
annotations = heatmap_data.copy()
annotations = annotations.applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else "")

plt.figure(figsize=(10, 6))
sns.heatmap(
    heatmap_data, 
    annot=annotations.values, 
    fmt="", 
    cmap='Blues',
    cbar_kws={'label': 'F1-Score'},
    annot_kws={"size": 10, "color": "black"},
    linewidths=0.5
)
plt.title('Per-Rule F1-Score Comparison Across Evaluated Approaches', fontsize=10)
plt.xlabel('Evaluated Approaches', fontsize=10)
plt.ylabel('Detection Rule', fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig("f1score_heatmap_rules_tools.png", dpi=300)
plt.show()