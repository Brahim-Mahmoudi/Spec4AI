import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from itertools import combinations

# Style matplotlib
sns.set(style="whitegrid")
plt.rcParams.update({"figure.autolayout": True})

def gini(array):
    array = np.array(array, dtype=np.float64).flatten()  # conversion explicite en float
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 1e-8  # évite les divisions par zéro
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))



def analyze_smell_folder(json_dir):
    project_smell_counts = []
    file_smell_counts = defaultdict(int)
    smell_occurrences = Counter()
    smell_per_project = defaultdict(set)
    project_rule_sets = defaultdict(set)
    smell_file_counts = defaultdict(set)
    co_occurrence_matrix = defaultdict(lambda: defaultdict(int))
    touched_projects = 0
    untouched_projects = 0


    project_names = []

    for filename in os.listdir(json_dir):
        if not filename.endswith(".json"):
            continue

        project_name = filename.replace(".json", "")
        project_names.append(project_name)
        file_path = os.path.join(json_dir, filename)

        with open(file_path, "r") as f:
            data = json.load(f)

        project_total_smells = 0
        rules_in_project = set()

        for file, smells in data.items():
            total_smells_in_file = sum(len(messages) for messages in smells.values())
            file_smell_counts[file] += total_smells_in_file
            project_total_smells += total_smells_in_file

            rules_in_file = set(smells.keys())
            rules_in_project.update(rules_in_file)

            for rule, messages in smells.items():
                smell_occurrences[rule] += len(messages)
                smell_per_project[rule].add(project_name)
                smell_file_counts[rule].add(file)

            for r1, r2 in combinations(rules_in_file, 2):
                co_occurrence_matrix[r1][r2] += 1
                co_occurrence_matrix[r2][r1] += 1
        if project_total_smells > 0:
            touched_projects += 1
        else:
            untouched_projects += 1

        project_smell_counts.append(project_total_smells)
        project_rule_sets[project_name] = rules_in_project

    # === STATISTIQUES ===
    project_rules_count = [len(rset) for rset in project_rule_sets.values()]
    mono_vs_multi = Counter(project_rules_count)

    stats = {
        "Total Projects": len(project_smell_counts),
        "Total Files Affected": len(file_smell_counts),
        "Mean Smells per Project": np.mean(project_smell_counts),
        "Median Smells per Project": np.median(project_smell_counts),
        "Max Smells per Project": np.max(project_smell_counts),
        "Min Smells per Project": np.min(project_smell_counts),
        "Std Smells per Project": np.std(project_smell_counts),
        "Mean Smells per File": np.mean(list(file_smell_counts.values())),
        "Median Smells per File": np.median(list(file_smell_counts.values())),
        "Max Smells per File": np.max(list(file_smell_counts.values())),
        "Min Smells per File": np.min(list(file_smell_counts.values())),
        "Std Smells per File": np.std(list(file_smell_counts.values())),
        "Top 10 Most Smelly Files": sorted(file_smell_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        "Mean Rule Types per Project": np.mean(project_rules_count),
        "Mono-smell Projects": mono_vs_multi.get(1, 0),
        "Multi-smell Projects": sum(v for k, v in mono_vs_multi.items() if k > 1),
        "Gini Index of Smell Distribution": gini(project_smell_counts),
        "Rare Smells (<5 projects)": [r for r, p in smell_per_project.items() if len(p) < 5],
        "Projects with at least one smell": touched_projects,
        "Projects with no smell": untouched_projects

    }


    df_occ = pd.DataFrame.from_dict(smell_occurrences, orient="index", columns=["Occurrences"])
    df_occ["Projects"] = df_occ.index.map(lambda r: len(smell_per_project[r]))
    df_occ["Files"] = df_occ.index.map(lambda r: len(smell_file_counts[r]))

    # Calculer la médiane des occurrences par projet pour chaque règle
    def median_occurrences_per_project(rule_id):
        counts = []
        for proj in smell_per_project[rule_id]:
            # nombre d'occurrences de cette règle dans ce projet
            count = sum(
                len(messages)
                for file, smells in json.load(open(os.path.join(json_dir, f"{proj}.json"))).items()
                if rule_id in smells
            )
            counts.append(count)
        return np.median(counts) if counts else 0

    df_occ["Median/Project"] = df_occ.index.map(median_occurrences_per_project)

    print(df_occ)
    df_co = pd.DataFrame(co_occurrence_matrix).fillna(0).astype(int)
    df_file = pd.DataFrame(list(file_smell_counts.items()), columns=["file", "smell_count"])
    df_project = pd.DataFrame(project_smell_counts, columns=["smell_count"])

    return stats, df_occ, df_co, df_file, df_project

def generate_plots(df_occ, df_co, df_file, df_project, output_dir="smell_figures"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Histogramme du nombre de smells par fichier
    plt.figure(figsize=(10, 5))
    sns.histplot(df_file["smell_count"], bins=50, kde=False)
    plt.title("Distribution of Smells per File")
    plt.xlabel("Number of Smells (capped at 50)")
    plt.ylabel("Number of Files")
    plt.xlim(0, 50)  # ou autre seuil selon la distribution
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hist_smells_per_file_log.png")


    # 2. Boxplot des smells par projet
    plt.figure(figsize=(6, 5))
    sns.boxplot(y=np.log1p(df_project["smell_count"]))  # log1p évite log(0)
    plt.title("Smells per Project (Log Scale)")
    plt.ylabel("log(Smell Count + 1)")
    plt.savefig(f"{output_dir}/boxplot_smells_per_project_log.png")


    # 3. Heatmap de co-occurrence
    top_rules = df_occ.sort_values("Occurrences", ascending=False).index
    top_rules = [r for r in top_rules if r in df_co.index and r in df_co.columns][:20]

    

    df_co_filtered = df_co.loc[top_rules, top_rules]

    plt.figure(figsize=(12, 10))
    sns.heatmap(df_co_filtered, cmap="Blues", annot=True, fmt="d", square=True)
    plt.title("Co-occurrence of Top 20 Smell Rules")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_cooccurrence_top20.png")


    # 4. Barplot des smells les plus fréquents
    plt.figure(figsize=(20, 10))
    df_occ_sorted = df_occ.sort_values("Occurrences", ascending=False)
    sns.barplot(x=df_occ_sorted.index, y="Occurrences", data=df_occ_sorted)
    plt.title("Most Common Smell Rules")
    plt.xlabel("Rule")
    plt.ylabel("Occurrences")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/barplot_smell_frequencies.png")


# === Exemple d'exécution ===
if __name__ == "__main__":
    json_folder = "/Users/bramss/Desktop/Json_Results"  # ⇦ à adapter
    stats, df_occ, df_co, df_file, df_project = analyze_smell_folder(json_folder)

    for k, v in stats.items():
        print(f"{k}: {v}")

    df_occ.to_csv("smell_summary.csv")
    df_co.to_csv("smell_cooccurrence.csv")
    generate_plots(df_occ, df_co, df_file, df_project)
