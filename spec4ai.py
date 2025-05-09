#!/usr/bin/env python3
"""
spec4ai.py: A command‑line interface for running Spec4AI rules on a Python project.

Usage:
    python spec4ai.py --input-dir <path> [--output-file <path>] [--rules R2 R6 R11] [--list-rules]

Example:
    python spec4ai.py --input-dir ./my_project --output-file results.json --rules R2 R6 R11
"""
from __future__ import annotations
import os
import ast
import json
import traceback
import argparse
from pathlib import Path
from collections import defaultdict

# Default root for rule modules (assumes subfolders R1, R2, ... exist under this directory)
RULES_ROOT = Path(__file__).parent / "test_rules"
ALIAS = {"R11bis": "R11"}  # merge R11bis into R11


def discover_available_rules(rules_root: Path) -> list[str]:
    """Discover rule IDs by scanning subdirectories under rules_root."""
    rule_ids: list[str] = []
    for sub in sorted(rules_root.iterdir()):
        if sub.is_dir() and sub.name.startswith("R"):
            # e.g. sub.name == "R2" or "R11bis"
            rule_ids.append(sub.name)
    return rule_ids


def import_rule(rule_id: str) -> tuple:
    """Dynamically import the module and function for a given rule ID."""
    mod_name = f"test_rules.{rule_id}.generated_rules_{rule_id}"
    module = __import__(mod_name, fromlist=[f"rule_{rule_id}"])
    func = getattr(module, f"rule_{rule_id}")
    return module, func


def analyze_file(filepath: str, selected_rules: list[str]) -> dict[str, list[str]]:
    """Return a mapping of rule IDs to list of issue messages for a single file."""
    try:
        source = Path(filepath).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=filepath)
    except Exception as e:
        return {"PARSE_ERROR": [f"Parse error: {e}"]}

    results: dict[str, list[str]] = defaultdict(list)
    for rid in selected_rules:
        module, rule_func = import_rule(rid)
        canonical = ALIAS.get(rid, rid)
        messages: list[str] = []

        def report(msg):  # capture reports
            messages.append(msg)

        # monkey‑patch report function
        saved = module.report
        module.report = report
        try:
            rule_func(tree)
        except Exception:
            messages.append(f"Execution error:\n{traceback.format_exc()}")
        finally:
            module.report = saved

        if messages:
            results[canonical].extend(messages)
    return results


def analyze_project(root: Path, rules: list[str]) -> dict[str, dict[str, list[str]]]:
    """Walk through .py files under root and apply rules; return dict of filepath->results."""
    output: dict[str, dict[str, list[str]]] = {}
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if fname.endswith('.py'):
                full = Path(dirpath) / fname
                res = analyze_file(str(full), rules)
                if res:
                    output[str(full)] = res
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run Spec4AI code smell detection on a Python project."
    )
    parser.add_argument(
        "--input-dir", "-i", type=Path, required=True,
        help="Path to the root directory of Python files to analyze"
    )
    parser.add_argument(
        "--output-file", "-o", type=Path, default=Path("spec4ai_results.json"),
        help="Path to write JSON results (default: spec4ai_results.json)"
    )
    parser.add_argument(
        "--rules", "-r", nargs='+', metavar='RULE_ID',
        help="List of rule IDs to run (e.g., R2 R6 R11); default is all available rules"
    )
    parser.add_argument(
        "--list-rules", action="store_true",
        help="List all available rule IDs and exit"
    )
    args = parser.parse_args()

    available = discover_available_rules(RULES_ROOT)
    if args.list_rules:
        print("Available rules:")
        for r in available:
            print(f"  {r}")
        exit(0)

    selected = args.rules if args.rules else available
    # Validate
    invalid = [r for r in selected if r not in available]
    if invalid:
        print(f"Error: Unknown rule IDs: {invalid}\nUse --list-rules to see available rules.")
        exit(1)

    print(f"Analyzing {args.input_dir} with rules: {selected}")
    results = analyze_project(args.input_dir, selected)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results written to {args.output_file} ({len(results)} files analyzed)")
