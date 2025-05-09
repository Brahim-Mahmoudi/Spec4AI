import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import numpy as np
from scipy.stats import wilcoxon
from matplotlib_venn import venn2
from upsetplot import UpSet, from_memberships
import os
os.makedirs("Figures_Precision", exist_ok=True)

def extract_smell_dict_from_excel(file_path: str, file_column: str = "Full Path") -> dict:
    """
    Extracts a dictionary mapping each file to a list of (rule, line) tuples
    from a smell matrix Excel file.

    Parameters:
    - file_path: path to the Excel file
    - file_column: name of the column that contains file paths

    Returns:
    - Dictionary with file names as keys and a list of (rule, line) as values
    """
    df = pd.read_excel(file_path)

    # Detect rule columns using regex (e.g., "R2", "R11")
    rule_columns = {}
    for col in df.columns:
        match = re.search(r"\b(R\d{1,2})\b", col)
        if match:
            rule_columns[col] = match.group(1)

    result = {}
    for _, row in df.iterrows():
        file = str(row[file_column]).strip()
        smells = []
        for original_col, rule_id in rule_columns.items():
            cell_value = str(row[original_col])
            if cell_value.lower() != "nan" and cell_value.strip():
                for line in cell_value.split(";"):
                    line = line.strip()
                    if line:
                        smells.append((rule_id, line))
        result[file] = smells

    return result


# Load detections from Spec4AI and human annotations

mlpylint_smells = extract_smell_dict_from_excel(
    "/Users/bramss/Desktop/mlpylint_code_smells_matrix.xlsx", file_column="file"
)

spec4ai_smells = extract_smell_dict_from_excel(
    "/Users/bramss/Desktop/spec4ai_code_smells_matrix_evaluation.xlsx", file_column="file"
)


llama3_smells = extract_smell_dict_from_excel(
    "/Users/bramss/Desktop/llama3-8b-8192_code_smells_matrix.xlsx", file_column="file"
)

llama3_70_8192_smells = extract_smell_dict_from_excel(
    "/Users/bramss/Desktop/llama3-70b-8192_code_smells_matrix.xlsx", file_column="file"
)

deepseek_smells = extract_smell_dict_from_excel(
    "/Users/bramss/Desktop/deepseek-r1-distill-llama-70b_code_smells_matrix.xlsx", file_column="file"
)


# "Python file/ Fichier python" for spec4ai or "File path/ Chemin Complet " for mlpylit

gt_smells = extract_smell_dict_from_excel(
    "/Users/bramss/Desktop/Google_Form_Manual_Eval.xlsx", file_column="File path/ Chemin Complet " # "Python file/ Fichier python" for spec4ai_smells or "File path/ Chemin Complet " for mlpylint_smells
)

# Display the first 3 files and their detections
#for i, (file, smells) in enumerate(spec4ai_smells.items()):
#    print(f"Spec4AI - File {i+1}: {file}")
#    print("  Detections:", smells)
#    if i == 2:
#        break

# Display the first 3 manually annotated files
#for i, (file, smells) in enumerate(gt_smells.items()):
#    print(f"Ground Truth - File {i+1}: {file}")
#    print("  Annotations:", smells)
#    if i == 2:
#        break

# Keep only files that exist in both sets
filtered_spec4ai_smells = {
    file: smells
    for file, smells in spec4ai_smells.items()
    if file in gt_smells
}
print(f"Matched files: {len(filtered_spec4ai_smells)} / {len(spec4ai_smells)}")

# Keep only files that exist in both sets
filtered_mlpylint_smells = {
    file: smells
    for file, smells in mlpylint_smells.items()
    if file in gt_smells
}
#print(f"Matched files: {len(filtered_spec4ai_smells)} / {len(spec4ai_smells)}")


# Filter ground truth to keep only annotated files
filtered_gt_smells = {
    file: smells
    for file, smells in gt_smells.items()
    if len(smells) > 0
}
print(f"Annotated files with at least one smell: {len(filtered_gt_smells)}")


filtered_llama3_smells = {
    file: smells
    for file, smells in llama3_smells.items()
    if file in gt_smells
}

filtered_llama3_70_8192_smells = {
    file: smells
    for file, smells in llama3_70_8192_smells.items()
    if file in gt_smells
}

filtered_deepseek_smells = {
    file: smells
    for file, smells in deepseek_smells.items()
    if file in gt_smells
}

# Convert to sets of (file, rule, line) with normalized line numbers
detected_spec4ai = {
    (file, rule, str(int(float(line))))
    for file, smells in filtered_spec4ai_smells.items()
    for (rule, line) in smells
    if line.strip() not in ["", "-", "nan"]
}





# Convert to sets of (file, rule, line) with normalized line numbers
detected_mlpylint = {
    (file, rule, str(int(float(line))))
    for file, smells in filtered_mlpylint_smells.items()
    for (rule, line) in smells
    if line.strip() not in ["", "-", "nan"]
}

ground_truth = {
    (file, rule, str(int(float(line))))
    for file, smells in filtered_gt_smells.items()
    for (rule, line) in smells
    if line.strip() not in ["", "-", "nan"]
}

# Compute TP, FP, FN for mlpylint
tp_mlpylint = detected_mlpylint & ground_truth
fp_mlpylint = detected_mlpylint - ground_truth
fn_mlpylint = ground_truth - detected_mlpylint

# Compute precision, recall, F1 for mlpylint
precision_mlpylint = len(tp_mlpylint) / (len(tp_mlpylint) + len(fp_mlpylint)) if (len(tp_mlpylint) + len(fp_mlpylint)) > 0 else 0.0
recall_mlpylint = len(tp_mlpylint) / (len(tp_mlpylint) + len(fn_mlpylint)) if (len(tp_mlpylint) + len(fn_mlpylint)) > 0 else 0.0
f1_mlpylint = 2 * precision_mlpylint * recall_mlpylint / (precision_mlpylint + recall_mlpylint) if (precision_mlpylint + recall_mlpylint) > 0 else 0.0

# Print metrics
print("\n📊 Evaluation Metrics")
print("✅ True Positives_mlpylint:", len(tp_mlpylint))
print("❌ False Positives_mlpylint:", len(fp_mlpylint))
print("❌ False Negatives_mlpylint:", len(fn_mlpylint))
print("🎯 Precision_mlpylint:", round(precision_mlpylint, 4))
print("📈 Recall_mlpylint:", round(recall_mlpylint, 4))
print("📊 F1 Score_mlpylint:", round(f1_mlpylint, 4))




gt_smells = extract_smell_dict_from_excel(
    "/Users/bramss/Desktop/Google_Form_Manual_Eval.xlsx", file_column="Python file/ Fichier python" # "Python file/ Fichier python" for spec4ai_smells or "File path/ Chemin Complet " for mlpylint_smells
)

filtered_spec4ai_smells = {
    file: smells
    for file, smells in spec4ai_smells.items()
    if file in gt_smells
}

filtered_llama3_smells = {
    file: smells
    for file, smells in llama3_smells.items()
    if file in gt_smells
}

filtered_llama3_70_8192_smells = {
    file: smells
    for file, smells in llama3_70_8192_smells.items()
    if file in gt_smells
}

filtered_deepseek_smells = {
    file: smells
    for file, smells in deepseek_smells.items()
    if file in gt_smells
}


#print(f"llama3 fichiers matchés avec GT: {len(filtered_deepseek_smells)}")
#print(f"Total smell detections dans llama3: {sum(len(v) for v in filtered_deepseek_smells.values())}")



filtered_gt_smells = {
    file: smells
    for file, smells in gt_smells.items()
    if len(smells) > 0
}

print(f"Matched files spec: {len(filtered_spec4ai_smells)} / {len(spec4ai_smells)}")
print(f"Matched files llama: {len(filtered_llama3_smells)} / {len(llama3_smells)}")
print(f"Matched files llama_70_8192: {len(filtered_llama3_70_8192_smells)} / {len(llama3_70_8192_smells)}")
print(f"Matched files deepseek: {len(filtered_deepseek_smells)} / {len(deepseek_smells)}")

# Convert to sets of (file, rule, line) with normalized line numbers
detected_spec4ai = {
    (file, rule, str(int(float(line))))
    for file, smells in filtered_spec4ai_smells.items()
    for (rule, line) in smells
    if line.strip() not in ["", "-", "nan"]
}

ground_truth = {
    (file, rule, str(int(float(line))))
    for file, smells in filtered_gt_smells.items()
    for (rule, line) in smells
    if line.strip() not in ["", "-", "nan"]
}

# Compute TP, FP, FN for spec4ai
tp_spec4ai = detected_spec4ai & ground_truth
fp_spec4ai = detected_spec4ai - ground_truth
fn_spec4ai = ground_truth - detected_spec4ai

# Compute precision, recall, F1 for spec4ai
precision_spec4ai = len(tp_spec4ai) / (len(tp_spec4ai) + len(fp_spec4ai)) if (len(tp_spec4ai) + len(fp_spec4ai)) > 0 else 0.0
recall_spec4ai = len(tp_spec4ai) / (len(tp_spec4ai) + len(fn_spec4ai)) if (len(tp_spec4ai) + len(fn_spec4ai)) > 0 else 0.0
f1_spec4ai = 2 * precision_spec4ai * recall_spec4ai / (precision_spec4ai + recall_spec4ai) if (precision_spec4ai + recall_spec4ai) > 0 else 0.0

# Print metrics
print("\n📊 Evaluation Metrics")
print("✅ True Positives_spec4ai:", len(tp_spec4ai))
print("❌ False Positives_spec4ai:", len(fp_spec4ai))
print("❌ False Negatives_spec4ai:", len(fn_spec4ai))
print("🎯 Precision_spec4ai:", round(precision_spec4ai, 4))
print("📈 Recall_spec4ai:", round(recall_spec4ai, 4))
print("📊 F1 Score_spec4ai:", round(f1_spec4ai, 4))



detected_llama3 = {
    (file, rule, str(int(float(line))))
    for file, smells in filtered_llama3_smells.items()
    for (rule, line) in smells
    if line.strip() not in ["", "-", "nan"]
}


tp_llama3 = detected_llama3 & ground_truth
fp_llama3 = detected_llama3 - ground_truth
fn_llama3 = ground_truth - detected_llama3


precision_llama3 = len(tp_llama3) / (len(tp_llama3) + len(fp_llama3)) if (len(tp_llama3) + len(fp_llama3)) > 0 else 0.0
recall_llama3 = len(tp_llama3) / (len(tp_llama3) + len(fn_llama3)) if (len(tp_llama3) + len(fn_llama3)) > 0 else 0.0
f1_llama3 = 2 * precision_llama3 * recall_llama3 / (precision_llama3 + recall_llama3) if (precision_llama3 + recall_llama3) > 0 else 0.0

print("\n📊 Evaluation Metrics")
print("✅ True Positives_llama3:", len(tp_llama3))
print("❌ False Positives_llama3:", len(fp_llama3))
print("❌ False Negatives_llama3:", len(fn_llama3))
print("🎯 Precision_llama3:", round(precision_llama3, 4))
print("📈 Recall_llama3:", round(recall_llama3, 4))
print("📊 F1 Score_llama3:", round(f1_llama3, 4))


detected_llama3_70_8192 = {
    (file, rule, str(int(float(line))))
    for file, smells in filtered_llama3_70_8192_smells.items()
    for (rule, line) in smells
    if line.strip() not in ["", "-", "nan"]
}


tp_llama3_70_8192 = detected_llama3_70_8192 & ground_truth
fp_llama3_70_8192 = detected_llama3_70_8192 - ground_truth
fn_llama3_70_8192 = ground_truth - detected_llama3_70_8192


precision_llama3_70_8192 = len(tp_llama3_70_8192) / (len(tp_llama3_70_8192) + len(fp_llama3_70_8192)) if (len(tp_llama3_70_8192) + len(fp_llama3_70_8192)) > 0 else 0.0
recall_llama3_70_8192 = len(tp_llama3_70_8192) / (len(tp_llama3_70_8192) + len(fn_llama3_70_8192)) if (len(tp_llama3_70_8192) + len(fn_llama3_70_8192)) > 0 else 0.0
f1_llama3_70_8192 = 2 * precision_llama3_70_8192 * recall_llama3_70_8192 / (precision_llama3_70_8192 + recall_llama3_70_8192) if (precision_llama3_70_8192 + recall_llama3_70_8192) > 0 else 0.0

print("\n📊 Evaluation Metrics")
print("✅ True Positives_llama3_70_8192:", len(tp_llama3_70_8192))
print("❌ False Positives_llama3_70_8192:", len(fp_llama3_70_8192))
print("❌ False Negatives_llama3_70_8192:", len(fn_llama3_70_8192))
print("🎯 Precision_llama3_70_8192:", round(precision_llama3_70_8192, 4))
print("📈 Recall_llama3_70_8192:", round(recall_llama3_70_8192, 4))
print("📊 F1 Score_llama3_70_8192:", round(f1_llama3_70_8192, 4))


detected_deepseek = {
    (file, rule, str(int(float(line))))
    for file, smells in filtered_deepseek_smells.items()
    for (rule, line) in smells
    if line.strip() not in ["", "-", "nan"]
}


tp_deepseek = detected_deepseek & ground_truth
fp_deepseek = detected_deepseek - ground_truth
fn_deepseek = ground_truth - detected_deepseek


precision_deepseek = len(tp_deepseek) / (len(tp_deepseek) + len(fp_deepseek)) if (len(tp_deepseek) + len(fp_deepseek)) > 0 else 0.0
recall_deepseek = len(tp_deepseek) / (len(tp_deepseek) + len(fn_deepseek)) if (len(tp_deepseek) + len(fn_deepseek)) > 0 else 0.0
f1_deepseek = 2 * precision_deepseek * recall_deepseek / (precision_deepseek + recall_deepseek) if (precision_deepseek + recall_deepseek) > 0 else 0.0

print("\n📊 Evaluation Metrics")
print("✅ True Positives_deepseek:", len(tp_deepseek))
print("❌ False Positives_deepseek:", len(fp_deepseek))
print("❌ False Negatives_deepseek:", len(fn_deepseek))
print("🎯 Precision_deepseek:", round(precision_deepseek, 4))
print("📈 Recall_deepseek:", round(recall_deepseek, 4))
print("📊 F1 Score_deepseek:", round(f1_deepseek, 4))

# Print detailed errors
#print("\n🔍 False Positives (Detected but not in ground truth):")
#for file, rule, line in sorted(fp):
#    print(f"  [FP] {file} — {rule} at line {line}")

#print("\n🔍 False Negatives (In ground truth but not detected):")
#for file, rule, line in sorted(fn):
#    print(f"  [FN] {file} — {rule} at line {line}")


results = {
    "mlpylint": {
        "TP": tp_mlpylint, "FP": fp_mlpylint, "FN": fn_mlpylint,
        "Precision": precision_mlpylint, "Recall": recall_mlpylint, "F1": f1_mlpylint
    },
    "Spec4AI": {
        "TP": tp_spec4ai, "FP": fp_spec4ai, "FN": fn_spec4ai,
        "Precision": precision_spec4ai, "Recall": recall_spec4ai, "F1": f1_spec4ai
    },
    "llama3-8b-8192": {
        "TP": tp_llama3, "FP": fp_llama3, "FN": fn_llama3,
        "Precision": precision_llama3, "Recall": recall_llama3, "F1": f1_llama3
    },
    "llama3_70_8192": {
        "TP": tp_llama3_70_8192, "FP": fp_llama3_70_8192, "FN": fn_llama3_70_8192,
        "Precision": precision_llama3_70_8192, "Recall": recall_llama3_70_8192, "F1": f1_llama3_70_8192
    },
    "deepseek": {
        "TP": tp_deepseek, "FP": fp_deepseek, "FN": fn_deepseek,
        "Precision": precision_deepseek, "Recall": recall_deepseek, "F1": f1_deepseek
    }
}

# === Bar Chart global ===
tools = list(results.keys())
precision = [results[t]["Precision"] for t in tools]
recall = [results[t]["Recall"] for t in tools]
f1_score = [results[t]["F1"] for t in tools]

x = range(len(tools))
bar_width = 0.25

plt.figure(figsize=(10, 6))
plt.bar([i - bar_width for i in x], precision, width=bar_width, label="Precision", color="skyblue")
plt.bar(x, recall, width=bar_width, label="Recall", color="lightgreen")
plt.bar([i + bar_width for i in x], f1_score, width=bar_width, label="F1 Score", color="salmon")
plt.xticks(ticks=x, labels=tools, fontsize=12)
plt.ylim(0, 1.05)
plt.title("Detection Metrics Comparison", fontsize=14)
plt.ylabel("Score", fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("Figures_Precision/global_metrics_bar.png", dpi=300)
plt.close()

# === Par règle ===
def compute_rule_metrics(tp_set, fp_set, fn_set):
    per_rule = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    for _, rule, _ in tp_set:
        per_rule[rule]["TP"] += 1
    for _, rule, _ in fp_set:
        per_rule[rule]["FP"] += 1
    for _, rule, _ in fn_set:
        per_rule[rule]["FN"] += 1
    return per_rule

def metrics_dict_to_df(rule_dict, tool_name):
    rows = []
    for rule, counts in rule_dict.items():
        TP, FP, FN = counts["TP"], counts["FP"], counts["FN"]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append([rule, TP, FP, FN, precision, recall, f1, tool_name])
    return pd.DataFrame(rows, columns=["Rule", "TP", "FP", "FN", "Precision", "Recall", "F1", "Tool"])

df_spec4ai_rules = metrics_dict_to_df(compute_rule_metrics(tp_spec4ai, fp_spec4ai, fn_spec4ai), "Spec4AI")
df_mlpylint_rules = metrics_dict_to_df(compute_rule_metrics(tp_mlpylint, fp_mlpylint, fn_mlpylint), "mlpylint")
df_llama3_rules = metrics_dict_to_df(compute_rule_metrics(tp_llama3, fp_llama3, fn_llama3), "llama3-8b-8192")
df_llama3_70_8192_rules = metrics_dict_to_df(compute_rule_metrics(tp_llama3_70_8192, fp_llama3_70_8192, fn_llama3_70_8192), "llama3_70_8192")
df_deepseek_rules = metrics_dict_to_df(compute_rule_metrics(tp_deepseek, fp_deepseek, fn_deepseek), "deepseek")
df_rules = pd.concat([df_spec4ai_rules, df_mlpylint_rules, df_llama3_rules, df_llama3_70_8192_rules, df_deepseek_rules], ignore_index=True)


# === Barplot F1 par règle
plt.figure(figsize=(12, 6))
sns.barplot(data=df_rules, x="Rule", y="F1", hue="Tool")
plt.title("F1-score by Rule")
plt.ylabel("F1 Score")
plt.ylim(0, 1.05)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("Figures_Precision/f1_per_rule_barplot.png", dpi=300)
plt.close()

# === Radar chart (only rules that appear in the test set) ===

# 1) Pivot to get a Rule × Tool table of F1 scores (0 if missing)
df_radar = (
    df_rules
    .pivot(index="Rule", columns="Tool", values="F1")
    .fillna(0)
)

# 2) Keep rules where at least one tool achieved a non-zero F1
df_radar = df_radar.loc[df_radar.max(axis=1) > 0]

# 3) Build angle/label arrays
labels = df_radar.index.tolist()
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
labels.append(labels[0])             # close the loop
angles.append(angles[0])

# 4) Plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for tool in df_radar.columns:
    values = df_radar[tool].tolist() + [df_radar[tool].iloc[0]]
    ax.plot(angles, values, label=tool)
    ax.fill(angles, values, alpha=0.25)

# 5) Formatting
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels[:-1], fontsize=9)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_title("Radar Chart – F1-score per Rule", fontsize=14)
ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.12))

plt.tight_layout()
plt.savefig("Figures_Precision/radar_chart_f1_rules.png", dpi=300)
plt.close()


# === Wilcoxon test
df_paired = df_spec4ai_rules.set_index("Rule")[["F1"]].join(
    df_mlpylint_rules.set_index("Rule")[["F1"]],
    lsuffix="_spec4ai", rsuffix="_mlpylint"
).dropna()

stat, p_value = wilcoxon(df_paired["F1_spec4ai"], df_paired["F1_mlpylint"])
print(f"\n📐 Wilcoxon Test on F1-score per Rule")
print(f"Statistic = {stat}, p-value = {p_value:.5f}")
if p_value < 0.05:
    print("✅ Statistically significant difference (p < 0.05)")
else:
    print("⚠️ No statistically significant difference (p ≥ 0.05)")

# === Bootstrap CI
def bootstrap_ci(data, n_bootstraps=1000, ci=95):
    stats = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=len(data), replace=True)
        stats.append(np.mean(sample))
    lower = np.percentile(stats, (100 - ci) / 2)
    upper = np.percentile(stats, 100 - (100 - ci) / 2)
    return round(np.mean(stats), 4), round(lower, 4), round(upper, 4)

for tool, df in [("Spec4AI", df_spec4ai_rules), ("mlpylint", df_mlpylint_rules), ("llama3-8b-8192", df_llama3_rules)]:
    for metric in ["Precision", "Recall", "F1"]:
        mean, ci_low, ci_high = bootstrap_ci(df[metric].dropna().values)
        print(f"{tool} {metric}: Mean={mean}, 95% CI=({ci_low}, {ci_high})")

# === Venn Diagrams
venn2([fp_spec4ai, fp_mlpylint], set_labels=("Spec4AI FP", "mlpylint FP"))
plt.title("False Positives Overlap")
plt.savefig("Figures_Precision/venn_fp_overlap.png", dpi=300)
plt.close()

venn2([fn_spec4ai, fn_mlpylint], set_labels=("Spec4AI FN", "mlpylint FN"))
plt.title("False Negatives Overlap")
plt.savefig("Figures_Precision/venn_fn_overlap.png", dpi=300)
plt.close()


# === Wilcoxon Test: Spec4AI vs LLaMA3
df_paired_llama = df_spec4ai_rules.set_index("Rule")[["F1"]].join(
    df_llama3_rules.set_index("Rule")[["F1"]],
    lsuffix="_spec4ai", rsuffix="_llama3"
).dropna()

stat, p_value = wilcoxon(df_paired_llama["F1_spec4ai"], df_paired_llama["F1_llama3"])
print(f"\n📐 Wilcoxon Test on F1-score per Rule: Spec4AI vs LLaMA3")
print(f"Statistic = {stat}, p-value = {p_value:.5f}")
if p_value < 0.05:
    print("✅ Statistically significant difference (p < 0.05)")
else:
    print("⚠️ No statistically significant difference (p ≥ 0.05)")

# === Wilcoxon Test: mlpylint vs LLaMA3
df_paired_llama2 = df_mlpylint_rules.set_index("Rule")[["F1"]].join(
    df_llama3_rules.set_index("Rule")[["F1"]],
    lsuffix="_mlpylint", rsuffix="_llama3"
).dropna()

stat, p_value = wilcoxon(df_paired_llama2["F1_mlpylint"], df_paired_llama2["F1_llama3"])
print(f"\n📐 Wilcoxon Test on F1-score per Rule: mlpylint vs LLaMA3")
print(f"Statistic = {stat}, p-value = {p_value:.5f}")
if p_value < 0.05:
    print("✅ Statistically significant difference (p < 0.05)")
else:
    print("⚠️ No statistically significant difference (p ≥ 0.05)")


    # === Venn: FP llama3-8b-8192 vs Spec4AI
venn2([fp_llama3, fp_spec4ai], set_labels=("llama3-8b-8192 FP", "Spec4AI FP"))
plt.title("False Positives Overlap – llama3-8b-8192 vs Spec4AI")
plt.savefig("Figures_Precision/venn_fp_llama3_spec4ai.png", dpi=300)
plt.close()

# === Venn: FP llama3-8b-8192 vs mlpylint
venn2([fp_llama3, fp_mlpylint], set_labels=("llama3-8b-8192 FP", "mlpylint FP"))
plt.title("False Positives Overlap – llama3-8b-8192 vs mlpylint")
plt.savefig("Figures_Precision/venn_fp_llama3_mlpylint.png", dpi=300)
plt.close()

# === Venn: FN llama3-8b-8192 vs Spec4AI
venn2([fn_llama3, fn_spec4ai], set_labels=("llama3-8b-8192 FN", "Spec4AI FN"))
plt.title("False Negatives Overlap – llama3-8b-8192 vs Spec4AI")
plt.savefig("Figures_Precision/venn_fn_llama3_spec4ai.png", dpi=300)
plt.close()

# === Venn: FN llama3-8b-8192 vs mlpylint
venn2([fn_llama3, fn_mlpylint], set_labels=("llama3-8b-8192 FN", "mlpylint FN"))
plt.title("False Negatives Overlap – llama3-8b-8192 vs mlpylint")
plt.savefig("Figures_Precision/venn_fn_llama3_mlpylint.png", dpi=300)
plt.close()

# Venn: FP llama3_70_8192 vs Spec4AI
venn2([fp_llama3_70_8192, fp_spec4ai], set_labels=("llama3_70_8192 FP", "Spec4AI FP"))
plt.title("False Positives Overlap – llama3_70_8192 vs Spec4AI")
plt.savefig("Figures_Precision/venn_fp_llama3_70_8192_spec4ai.png", dpi=300)
plt.close()

# Venn: FP deepseek vs Spec4AI
venn2([fp_deepseek, fp_spec4ai], set_labels=("deepseek FP", "Spec4AI FP"))
plt.title("False Positives Overlap – deepseek vs Spec4AI")
plt.savefig("Figures_Precision/venn_fp_deepseek_spec4ai.png", dpi=300)
plt.close()

# Venn: FN llama3_70_8192 vs Spec4AI
venn2([fn_llama3_70_8192, fn_spec4ai], set_labels=("llama3_70_8192 FN", "Spec4AI FN"))
plt.title("False Negatives Overlap – llama3_70_8192 vs Spec4AI")
plt.savefig("Figures_Precision/venn_fn_llama3_70_8192_spec4ai.png", dpi=300)
plt.close()

# Venn: FN deepseek vs Spec4AI
venn2([fn_deepseek, fn_spec4ai], set_labels=("deepseek FN", "Spec4AI FN"))
plt.title("False Negatives Overlap – deepseek vs Spec4AI")
plt.savefig("Figures_Precision/venn_fn_deepseek_spec4ai.png", dpi=300)
plt.close()

# Venn: FP deepseek vs llama3-8b-8192
venn2([fp_deepseek, fp_llama3], set_labels=("deepseek FP", "llama3-8b-8192 FP"))
plt.title("False Positives Overlap – deepseek vs llama3-8b-8192")
plt.savefig("Figures_Precision/venn_fp_deepseek_llama3.png", dpi=300)
plt.close()

# Venn: FN deepseek vs llama3-8b-8192
venn2([fn_deepseek, fn_llama3], set_labels=("deepseek FN", "llama3-8b-8192 FN"))
plt.title("False Negatives Overlap – deepseek vs llama3-8b-8192")
plt.savefig("Figures_Precision/venn_fn_deepseek_llama3.png", dpi=300)
plt.close()


# === UpSet plot – False Positives across Spec4AI, mlpylint, llama3-8b-8192
all_fp = set.union(fp_spec4ai, fp_mlpylint, fp_llama3, fp_llama3_70_8192, fp_deepseek)
fp_memberships = []
for item in all_fp:
    tools = []
    if item in fp_spec4ai:
        tools.append("Spec4AI")
    if item in fp_mlpylint:
        tools.append("mlpylint")
    if item in fp_llama3:
        tools.append("llama3-8b-8192")
    if item in fp_llama3_70_8192:
        tools.append("llama3_70_8192")
    if item in fp_deepseek:
        tools.append("deepseek")
    fp_memberships.append(tools)

fp_data = from_memberships(fp_memberships)
# UpSet plot – False Positives
plt.figure(figsize=(10, 6))
upset_obj = UpSet(fp_data, subset_size='count', show_counts=True)
upset_obj.plot()

plt.gca().set_ylabel("Intersection size")
plt.suptitle("UpSet Plot – False Positives Overlap")
plt.tight_layout()
plt.savefig("Figures_Precision/upset_fp_overlap.png", dpi=300)
plt.close()



# === UpSet plot – False Negatives across Spec4AI, mlpylint, llama3-8b-8192
all_fn = set.union(fn_spec4ai, fn_mlpylint, fn_llama3)
fn_memberships = []
for item in all_fn:
    tools = []
    if item in fn_spec4ai:
        tools.append("Spec4AI")
    if item in fn_mlpylint:
        tools.append("mlpylint")
    if item in fn_llama3:
        tools.append("llama3-8b-8192")
    if item in fn_llama3_70_8192:
        tools.append("llama3_70_8192")
    if item in fn_deepseek:
        tools.append("deepseek")
    fn_memberships.append(tools)

fn_data = from_memberships(fn_memberships)


# UpSet plot – False Negatives
plt.figure(figsize=(10, 6))
upset_obj = UpSet(fn_data, subset_size='count', show_counts=True)
upset_obj.plot()

# Ajoute le label "Intersection size" à l’axe Y supérieur
plt.gca().set_ylabel("Intersection size")

plt.suptitle("UpSet Plot – False Negatives Overlap")
plt.tight_layout()
plt.savefig("Figures_Precision/upset_fn_overlap.png", dpi=300)
plt.close()


# ------------------------------------------------------------------
# ---------- GLOBAL BAR CHART WITH 95 % CONFIDENCE INTERVALS -------
# ------------------------------------------------------------------
from statsmodels.stats.proportion import proportion_confint

def wilson_ci(successes, total, alpha=0.05):
    lo, hi = proportion_confint(successes, total, alpha=alpha, method="wilson")
    return lo, hi                                   # already 0-1 scale

def bootstrap_f1(tp, fp, fn, n_boot=2000, alpha=0.05, rng=None):
    rng = np.random.default_rng(rng)
    stats = []
    for _ in range(n_boot):
        # Tirage bootstrap sur (TP,FP,FN)
        tp_b = rng.integers(0, tp+1)
        fp_b = rng.integers(0, fp+1)
        fn_b = rng.integers(0, fn+1)
        prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) else 0
        rec_b  = tp_b / (tp_b + fn_b) if (tp_b + fn_b) else 0
        f1_b   = 2*prec_b*rec_b/(prec_b+rec_b) if (prec_b+rec_b) else 0
        stats.append(f1_b)
    lo, hi = np.percentile(stats, [100*alpha/2, 100*(1-alpha/2)])
    return lo, hi

tools            = list(results.keys())
precision        = []
recall           = []
f1_score         = []
prec_ci, rec_ci, f1_ci = [], [], []

# ---- build vectors + CI ----
for t in tools:
    tp = len(results[t]["TP"])
    fp = len(results[t]["FP"])
    fn = len(results[t]["FN"])

    p  = results[t]["Precision"]
    r  = results[t]["Recall"]
    f1 = results[t]["F1"]

    precision.append(p)
    recall.append(r)
    f1_score.append(f1)

    prec_ci.append( wilson_ci(tp, tp+fp) )
    rec_ci.append(  wilson_ci(tp, tp+fn) )
    f1_ci.append(   bootstrap_f1(tp, fp, fn) )

# ---- prepare y-err arrays (hauteur au-dessus et au-dessous) ----
def to_yerr(ci_list, values):
    return np.abs(np.array(ci_list).T - values)

prec_yerr = to_yerr(prec_ci, precision)
rec_yerr  = to_yerr(rec_ci,  recall)
f1_yerr   = to_yerr(f1_ci,   f1_score)

# ---- plotting ----
x          = np.arange(len(tools))
bar_width  = 0.25

plt.figure(figsize=(10, 6))

plt.bar(x - bar_width, precision, width=bar_width,
        yerr=prec_yerr, capsize=4, label="Precision", color="skyblue")

plt.bar(x, recall, width=bar_width,
        yerr=rec_yerr, capsize=4, label="Recall", color="lightgreen")

plt.bar(x + bar_width, f1_score, width=bar_width,
        yerr=f1_yerr, capsize=4, label="F1 Score", color="salmon")

plt.xticks(ticks=x, labels=tools, fontsize=12)
plt.ylim(0, 1.05)
plt.title("Detection Metrics Comparison (±95 % CI)", fontsize=14)
plt.ylabel("Score", fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("Figures_Precision/global_metrics_bar.png", dpi=300)
plt.close()
