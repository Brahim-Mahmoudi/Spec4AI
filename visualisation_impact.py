import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Remplacez ce bloc par la lecture de votre CSV ou la reconstruction de summary_df
summary_rows = [
    {'Model': 'MLPClassifier',  'Default Precision': 0.956265, 'Default Time (s)': 0.461329, 'Best Precision': 0.932154, 'Best Time (s)': 0.442319},
    {'Model': 'Ridge',          'Default Precision': float('nan'), 'Default Time (s)': 4.264187, 'Best Precision': float('nan'), 'Best Time (s)': 4.264187},
    {'Model': 'SVR',            'Default Precision': float('nan'), 'Default Time (s)': 0.020121, 'Best Precision': float('nan'), 'Best Time (s)': 0.020121},
    {'Model': 'GradientBoostingClassifier', 'Default Precision': 0.962302, 'Default Time (s)': 0.497877, 'Best Precision': 0.974868, 'Best Time (s)': 0.490006},
    {'Model': 'Lasso',          'Default Precision': float('nan'), 'Default Time (s)': 0.006972, 'Best Precision': float('nan'), 'Best Time (s)': 0.006972},
    {'Model': 'RandomForestClassifier','Default Precision': 0.972206, 'Default Time (s)': 0.262067, 'Best Precision': 0.959797, 'Best Time (s)': 0.129264},
    {'Model': 'LogisticRegression','Default Precision': 0.945608, 'Default Time (s)': 0.051185, 'Best Precision': 0.957879, 'Best Time (s)': 0.007771},
    {'Model': 'LGBMClassifier','Default Precision': 0.986486, 'Default Time (s)': 0.155339, 'Best Precision': 0.986486, 'Best Time (s)': 0.125283},
    {'Model': 'SVC',           'Default Precision': 0.939964, 'Default Time (s)': 0.007419, 'Best Precision': 0.956265, 'Best Time (s)': 0.005822},
    {'Model': 'AdaBoostClassifier','Default Precision': 0.959797, 'Default Time (s)': 0.246897, 'Best Precision': 0.974868, 'Best Time (s)': 0.535373},
    {'Model': 'DecisionTreeClassifier','Default Precision': 0.913720, 'Default Time (s)': 0.010364, 'Best Precision': 0.926613, 'Best Time (s)': 0.008905},
    {'Model': 'KMeans',        'Default Precision': 0.470148, 'Default Time (s)': 0.262685, 'Best Precision': 0.706342, 'Best Time (s)': 0.081417},
    {'Model': 'AgglomerativeClustering','Default Precision': 0.703280, 'Default Time (s)': 0.044882, 'Best Precision': 0.722063, 'Best Time (s)': 0.007578},
    # … ajoutez les autres lignes si nécessaire …
]
summary_df = pd.DataFrame(summary_rows)

# 1) Box-plot des précisions
plt.figure(figsize=(6, 4))
plt.boxplot(
    [summary_df['Default Precision'].dropna(), summary_df['Best Precision'].dropna()],
    labels=['Default', 'Best']
)
plt.ylabel('Precision (macro)')
plt.title('Distribution of Default vs Best Precision')
plt.tight_layout()
plt.show()

# 2) Box-plot des temps de fit
plt.figure(figsize=(6, 4))
plt.boxplot(
    [summary_df['Default Time (s)'], summary_df['Best Time (s)']],
    labels=['Default', 'Best']
)
plt.ylabel('Fit Time (s)')
plt.title('Distribution of Default vs Best Fit Time')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(summary_df['Default Time (s)'], summary_df['Default Precision'], label='Default')
plt.scatter(summary_df['Best Time (s)'],    summary_df['Best Precision'],    label='Best', marker='x')
plt.xlabel('Fit Time (s)')
plt.ylabel('Precision (macro)')
plt.title('Precision vs Fit Time')
plt.legend()
plt.tight_layout()
plt.show()

models = summary_df['Model']
default = summary_df['Default Precision']
best    = summary_df['Best Precision']

ind = np.arange(len(models))
width = 0.35

plt.figure(figsize=(8,4))
plt.bar(ind - width/2, default, width, label='Default')
plt.bar(ind + width/2, best,    width, label='Best')
plt.xticks(ind, models, rotation=45, ha='right')
plt.ylabel('Precision (macro)')
plt.title('Default vs Best Precision by Model')
plt.legend()
plt.tight_layout()
plt.show()

gains = summary_df['Best Precision'] - summary_df['Default Precision']

plt.figure(figsize=(8,3))
plt.plot(summary_df['Model'], gains, marker='o')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Precision Gain')
plt.title('Gain de précision par modèle')
plt.tight_layout()
plt.show()
