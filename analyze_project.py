#!/usr/bin/env python3
"""
Analyse mlflow-master avec toutes les règles R1–R22 + R11bis,
en fusionnant les messages de R11bis dans R11.

Exécution
---------
python analyze_all_rules.py
"""

import os
import ast
import json
import traceback
from pathlib import Path
from collections import defaultdict

# --------------------------------------------------------------------------- #
# 0) Chemins en dur                                                           #
# --------------------------------------------------------------------------- #
ROOT_DIR = Path("/Users/bramss/Desktop/mlflow_projet_sampled").resolve()
OUTPUT_JSON = Path(
    "/Users/bramss/Documents/ETS/PhD/Code_Smells_ML/mlpylint/Spec4AI_PeterHamfelt_mlflow_sampled.json"
).resolve()

# --------------------------------------------------------------------------- #
# 1) Import dynamique des modules générés                                     #
# --------------------------------------------------------------------------- #
RULE_MODULES = []

# R1 → R22
for rid in range(1, 23):
    mod_name = f"test_rules.R{rid}.generated_rules_R{rid}"
    module = __import__(mod_name, fromlist=[f"rule_R{rid}"])
    RULE_MODULES.append((module, getattr(module, f"rule_R{rid}")))

# + R11bis
mod_11bis = __import__(
    "test_rules.R11bis.generated_rules_R11bis", fromlist=["rule_R11bis"]
)
RULE_MODULES.append((mod_11bis, mod_11bis.rule_R11bis))

# --------------------------------------------------------------------------- #
# 2) Alias : comment renommer certaines règles dans la sortie                 #
# --------------------------------------------------------------------------- #
ALIAS = {"R11bis": "R11"}  # fusion R11bis -> R11


def analyze_file(filepath: str) -> dict:
    """Retourne { 'R2': [...], 'R11': [...], ... } pour un fichier."""
    try:
        with open(filepath, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)
    except Exception as e:
        return {"PARSE_ERROR": [f"Erreur de parsing : {e}"]}

    res: dict[str, list[str]] = defaultdict(list)

    for module, rule_func in RULE_MODULES:
        orig_id = rule_func.__name__.split("_")[1]  # 'R2' ou 'R11bis'
        rule_id = ALIAS.get(orig_id, orig_id)       # applique l'alias
        collected: list[str] = []

        def report(msg):
            collected.append(msg)

        saved_report = module.report
        module.report = report
        try:
            rule_func(tree)
        except Exception as e:
            collected.append(f"Erreur d’exécution : {e}\n{traceback.format_exc()}")
        finally:
            module.report = saved_report

        if collected:
            res[rule_id].extend(collected)

    return res


def analyze_project(root: Path) -> dict:
    results = {}
    for path, _, files in os.walk(root):
        for name in files:
            if name.endswith(".py"):
                full = os.path.join(path, name)
                per_file = analyze_file(full)
                if per_file:
                    results[full] = per_file
    return results


def main():
    print(f"🔍 Analyse de {ROOT_DIR}")
    data = analyze_project(ROOT_DIR)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Rapport écrit : {OUTPUT_JSON}  ({len(data)} fichiers)")

if __name__ == "__main__":
    main()
