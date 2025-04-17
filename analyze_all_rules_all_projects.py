import os
import ast
import json
import subprocess
import traceback
import importlib
import time
import argparse
import pandas as pd
from functools import partial
from multiprocessing import cpu_count, get_context
import glob

_module_cache = {}  # Cache global par process

def get_all_rule_folders():
    rule_folders = glob.glob("test_rules/R*")
    return sorted([rf for rf in rule_folders if os.path.isdir(rf)])

def load_all_rules():
    rules = {}
    rule_folders = get_all_rule_folders()

    for rule_folder in rule_folders:
        rule_name = os.path.basename(rule_folder)
        dsl_file = os.path.join(rule_folder, f"test_{rule_name}.dsl")
        output_py = os.path.join(rule_folder, f"generated_rules_{rule_name}.py")

        if not os.path.exists(dsl_file):
            print(f"⏭️ Fichier DSL manquant pour {rule_name}, on saute.")
            continue

        try:
            print(f"🛠️  Génération de {rule_name} depuis {dsl_file}...")
            subprocess.run([
                "python", "parse_dsl.py",
                "--dsl", dsl_file,
                "--output", output_py
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur lors de la génération de {rule_name} : {e}")
            continue

        try:
            module_name = f"test_rules.{rule_name}.generated_rules_{rule_name}"
            module = importlib.import_module(module_name)
            rule_func = getattr(module, f"rule_{rule_name}")
            rules[rule_name] = rule_func
        except Exception as e:
            print(f"❌ Erreur lors de l'import de {module_name}: {e}")
            continue

    print(f"✅ {len(rules)} règles chargées avec succès.")
    return rules

def analyze_file(filepath, rules):
    messages_by_rule = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
        tree = ast.parse(code, filename=filepath)
    except Exception as e:
        return filepath, {"error": f"Erreur de parsing: {e}\n{traceback.format_exc()}"}

    for rule_id, rule_func in rules.items():
        messages = []

        def report(msg):
            messages.append(msg)

        try:
            if rule_id not in _module_cache:
                module_name = f"test_rules.{rule_id}.generated_rules_{rule_id}"
                _module_cache[rule_id] = importlib.import_module(module_name)

            module = _module_cache[rule_id]

            old_report = getattr(module, "report", None)
            setattr(module, "report", report)

            rule_func(tree)

            setattr(module, "report", old_report)
        except Exception as e:
            messages.append(f"Erreur lors de l'exécution de {rule_id}: {e}\n{traceback.format_exc()}")

        if messages:
            messages_by_rule[rule_id] = messages

    del tree  # ✅ libère mémoire
    return filepath, messages_by_rule if messages_by_rule else {}

def collect_python_files(root_dir):
    py_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                py_files.append(os.path.join(dirpath, filename))
    return py_files

def collect_project_paths(parent_dir):
    return [os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, d))]

def analyze_project(project_dir, rules, sequential=False):
    py_files = collect_python_files(project_dir)
    results = {}
    analyze_partial = partial(analyze_file, rules=rules)

    if sequential:
        for filepath in py_files:
            filepath, result = analyze_partial(filepath)
            if result:
                results[filepath] = result
    else:
        ctx = get_context("spawn")  # ✅ plus sûr sur macOS
        with ctx.Pool(processes=min(cpu_count(), 4)) as pool:
            for filepath, result in pool.imap_unordered(analyze_partial, py_files):
                if result:
                    results[filepath] = result

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequential", action="store_true", help="Désactive multiprocessing (mode séquentiel)")
    args = parser.parse_args()

    rules = load_all_rules()
    ALL_PROJECTS_DIR = "/Users/bramss/Documents/ETS/PhD/Code_Smells_ML/Comparison_MLpylint"
    OUTPUT_DIR = "/Users/bramss/Desktop/Results_160_files_Pter"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    project_paths = collect_project_paths(ALL_PROJECTS_DIR)
    project_timings = []  # ⏱️ Stocker les durées

    for project_path in project_paths:
        print(f"\n⏳ Analyse du projet : {project_path}")
        start = time.time()

        results = analyze_project(project_path, rules, sequential=args.sequential)

        end = time.time()
        duration = round(end - start, 2)
        print(f"✅ Terminé en {duration:.2f} secondes")

        project_name = os.path.basename(project_path)
        output_json = os.path.join(OUTPUT_DIR, f"{project_name}_results.json")

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"📁 Résultats sauvegardés dans {output_json}")
        project_timings.append((project_name, duration))

    # 📊 Export Excel des temps
    df = pd.DataFrame(project_timings, columns=["Projet", "Durée (s)"])
    excel_path = os.path.join(OUTPUT_DIR, "résumé_durations.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"\n📊 Fichier Excel généré : {excel_path}")

if __name__ == "__main__":
    main()
