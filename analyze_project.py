#!/usr/bin/env python3
import os
import ast
import json
import sys
import traceback

# Import des modules contenant les règles générées
from test_rules.R1 import generated_rules_R1
from test_rules.R2 import generated_rules_R2
from test_rules.R3 import generated_rules_R3
from test_rules.R4 import generated_rules_R4
from test_rules.R5 import generated_rules_R5
from test_rules.R6 import generated_rules_R6
from test_rules.R7 import generated_rules_R7
from test_rules.R8 import generated_rules_R8
from test_rules.R9 import generated_rules_R9
from test_rules.R10 import generated_rules_R10
from test_rules.R11 import generated_rules_R11
from test_rules.R12 import generated_rules_R12
from test_rules.R13 import generated_rules_R13
from test_rules.R14 import generated_rules_R14
from test_rules.R15 import generated_rules_R15
from test_rules.R16 import generated_rules_R16
from test_rules.R17 import generated_rules_R17
from test_rules.R18 import generated_rules_R18
from test_rules.R19 import generated_rules_R19
from test_rules.R20 import generated_rules_R20
from test_rules.R21 import generated_rules_R21
from test_rules.R22 import generated_rules_R22

# Liste des modules et fonctions de règles
RULE_MODULES = [
    (generated_rules_R1, generated_rules_R1.rule_R1),
    (generated_rules_R2, generated_rules_R2.rule_R2),
    (generated_rules_R3, generated_rules_R3.rule_R3),
    (generated_rules_R4, generated_rules_R4.rule_R4),
    (generated_rules_R5, generated_rules_R5.rule_R5),
    (generated_rules_R6, generated_rules_R6.rule_R6),
    (generated_rules_R7, generated_rules_R7.rule_R7),
    (generated_rules_R8, generated_rules_R8.rule_R8),
    (generated_rules_R9, generated_rules_R9.rule_R9),
    (generated_rules_R10, generated_rules_R10.rule_R10),
    (generated_rules_R11, generated_rules_R11.rule_R11),
    (generated_rules_R12, generated_rules_R12.rule_R12),
    (generated_rules_R13, generated_rules_R13.rule_R13),
    (generated_rules_R14, generated_rules_R14.rule_R14),
    (generated_rules_R15, generated_rules_R15.rule_R15),
    (generated_rules_R16, generated_rules_R16.rule_R16),
    (generated_rules_R17, generated_rules_R17.rule_R17),
    (generated_rules_R18, generated_rules_R18.rule_R18),
    (generated_rules_R19, generated_rules_R19.rule_R19),
    (generated_rules_R20, generated_rules_R20.rule_R20),
    (generated_rules_R21, generated_rules_R21.rule_R21),
    (generated_rules_R22, generated_rules_R22.rule_R22),
]

def analyze_file(filepath):
    """Analyse un fichier Python et retourne un dictionnaire avec les messages d'erreur ou de rapport."""
    messages = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
        tree = ast.parse(code, filename=filepath)
    except Exception as e:
        return {"error": f"Erreur de parsing: {e}\n{traceback.format_exc()}"}
    
    for module, rule_function in RULE_MODULES:
        local_messages = []
        def report(message):
            local_messages.append(message)
        old_report = module.report
        module.report = report
        try:
            rule_function(tree)
        except Exception as e:
            local_messages.append(f"Erreur lors de l'exécution de la règle {rule_function.__name__}: {e}\n{traceback.format_exc()}")
        module.report = old_report
        if local_messages:
            messages.extend([f"{rule_function.__name__}: {m}" for m in local_messages])
    
    return {"messages": messages}

def analyze_project(root_dir):
    """Parcourt récursivement le répertoire root_dir pour analyser tous les fichiers .py.
       Seuls les fichiers ayant une erreur ou des messages seront inclus dans le résultat."""
    results = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                filepath = os.path.join(dirpath, filename)
                file_result = analyze_file(filepath)
                if ("error" in file_result) or (file_result.get("messages") and len(file_result["messages"]) > 0):
                    results[filepath] = file_result
    return results

def main():
    ROOT_DIR = "/Users/bramss/Desktop/mlflow-master"
    OUTPUT_JSON = "/Users/bramss/Documents/ETS/PhD/Code_Smells_ML/Code_Smell_Detection/DSL/DSL_Smell_Detector/test_rules/generated_all_rules_mlflow.json"
    results = analyze_project(ROOT_DIR)
    try:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Analyse terminée. Résultats écrits dans {OUTPUT_JSON}")
    except Exception as e:
        print("Erreur lors de l'écriture du fichier généré:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
