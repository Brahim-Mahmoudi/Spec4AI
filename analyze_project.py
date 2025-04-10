#!/usr/bin/env python3
import os
import ast
import json
import sys
import traceback
from multiprocessing import Pool, cpu_count

# Import uniquement la règle que tu veux exécuter (R11 ici)
from test_rules.R11bis import generated_rules_11bis

def analyze_file(filepath):
    """Analyse un fichier Python et retourne un dictionnaire avec les messages d'erreur ou de rapport."""
    messages = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
        tree = ast.parse(code, filename=filepath)
    except Exception as e:
        return filepath, {"error": f"Erreur de parsing: {e}\n{traceback.format_exc()}"}

    def report(message):
        messages.append(message)

    # Remplacement temporaire de la fonction report
    old_report = generated_rules_11bis.report
    generated_rules_11bis.report = report

    try:
        generated_rules_11bis.rule_R11bis(tree)
    except Exception as e:
        messages.append(f"Erreur lors de l'exécution de la règle: {e}\n{traceback.format_exc()}")
    finally:
        generated_rules_11bis.report = old_report

    return filepath, {"messages": messages} if messages else {}

def collect_python_files(root_dir):
    py_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                py_files.append(os.path.join(dirpath, filename))
    return py_files

def analyze_project_parallel(root_dir):
    py_files = collect_python_files(root_dir)
    results = {}

    with Pool(processes=cpu_count()) as pool:
        for filepath, result in pool.imap_unordered(analyze_file, py_files):
            if "error" in result or result.get("messages"):
                results[filepath] = result

    return results

def main():
    ROOT_DIR = "/Users/bramss/Documents/ETS/PhD/Code_Smells_ML/mlpylint/Source_Code/PeterHamfelt_mlflow"
    OUTPUT_JSON = "/Users/bramss/Documents/ETS/PhD/Code_Smells_ML/Code_Smell_Detection/DSL/test_rules/R11bis/generated_rules_11bis.json"

    print(f"Analyse en cours avec {cpu_count()} processus...")
    results = analyze_project_parallel(ROOT_DIR)

    try:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Analyse terminée. Résultats écrits dans {OUTPUT_JSON}")
    except Exception as e:
        print("Erreur lors de l'écriture du fichier généré:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
