#!/usr/bin/env python3
from __future__ import annotations

# Suppress macOS NSButton height warnings
import sys, os; sys.stderr = open(os.devnull, 'w')

import json
import pathlib
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
import spec4ai
from concurrent.futures import ProcessPoolExecutor
from threading import Thread

def init_worker(rule_ids: list[str]) -> None:
    """Preload the selected rules in each worker process."""
    global WORKER_RULES
    WORKER_RULES = [spec4ai.import_rule(r) for r in rule_ids]

def worker(path: str) -> tuple[str, dict[str, list[str]]]:
    """Analyze a single file and return any alert messages."""
    import ast
    from collections import defaultdict

    try:
        source = pathlib.Path(path).read_text(encoding='utf-8')
        tree = ast.parse(source, filename=path)
    except Exception as e:
        return path, {"PARSE_ERROR": [f"Parse error: {e}"]}

    results = defaultdict(list)
    for module, func in WORKER_RULES:
        msgs: list[str] = []
        saved_report = module.report
        module.report = lambda m, mgs=msgs: mgs.append(m)
        try:
            func(tree)
        except Exception:
            msgs.append(traceback.format_exc())
        module.report = saved_report

        if msgs:
            rid = spec4ai.ALIAS.get(func.__name__.split('_')[1], func.__name__)
            results[rid] = msgs

    return path, dict(results)

class Spec4AIGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spec4AI GUI")
        self.geometry("700x500")
        self.columnconfigure(1, weight=1)
        self._build_widgets()

    def _build_widgets(self):
        # Input directory selector
        tk.Label(self, text="Input directory:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.input_entry = tk.Entry(self)
        self.input_entry.grid(row=0, column=1, sticky="we")
        tk.Button(self, text="Browse", command=lambda: self._browse(self.input_entry, True)).grid(row=0, column=2, padx=5)

        # Output file selector
        tk.Label(self, text="Output JSON file:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.output_entry = tk.Entry(self)
        self.output_entry.insert(0, "spec4ai_results.json")
        self.output_entry.grid(row=1, column=1, sticky="we")
        tk.Button(self, text="Browse", command=lambda: self._browse(self.output_entry, False)).grid(row=1, column=2, padx=5)

        # Rule selection listbox
        tk.Label(self, text="Select rules to run:").grid(row=2, column=0, columnspan=3, sticky="w", padx=5)
        self.rule_listbox = tk.Listbox(self, selectmode="extended", height=10)
        self.rule_listbox.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        rules = spec4ai.discover_available_rules(pathlib.Path(__file__).parent / "test_rules")
        for r in rules:
            self.rule_listbox.insert(tk.END, r)
        self.rule_listbox.select_set(0, tk.END)

        # Status label
        self.status_label = tk.Label(self, text="Ready", anchor="w")
        self.status_label.grid(row=4, column=0, columnspan=3, sticky="we", padx=5)

        # Action buttons
        tk.Button(self, text="Run Analysis", command=self._on_run).grid(row=5, column=1, sticky="e", padx=5, pady=10)
        tk.Button(self, text="Exit", command=self.destroy).grid(row=5, column=2, sticky="w", padx=5, pady=10)

    def _browse(self, widget: tk.Entry, is_directory: bool):
        chooser = filedialog.askdirectory if is_directory else filedialog.asksaveasfilename
        opts = {} if is_directory else {"defaultextension": ".json", "filetypes": [("JSON files","*.json")]}
        path = chooser(**opts)
        if path:
            widget.delete(0, tk.END)
            widget.insert(0, path)

    def _on_run(self):
        input_dir = self.input_entry.get().strip()
        output_file = self.output_entry.get().strip()
        if not pathlib.Path(input_dir).is_dir():
            return messagebox.showwarning("Input Error", "Please select a valid input directory.")
        selected = [self.rule_listbox.get(i) for i in self.rule_listbox.curselection()]
        if not selected:
            return messagebox.showwarning("Selection Error", "Please select at least one rule.")
        self.status_label.config(text="Running...")
        Thread(target=self._analyze, args=(input_dir, output_file, selected), daemon=True).start()

    def _analyze(self, input_dir: str, output_file: str, rules: list[str]):
        try:
            files = [str(p) for p in pathlib.Path(input_dir).rglob("*.py")]
            total_scanned = len(files)
            results: dict[str, dict[str, list[str]]] = {}

            with ProcessPoolExecutor(
                max_workers=max(1, os.cpu_count() - 1),
                initializer=init_worker,
                initargs=(rules,)
            ) as executor:
                for path, res in executor.map(worker, files):
                    if res:
                        results[path] = res

            pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # Back to main thread for GUI updates
            msg = f"Files scanned: {total_scanned}\nFiles with alerts: {len(results)}"
            self.after(0, lambda: self._done(True, msg))
        except Exception as e:
            self.after(0, lambda: self._done(False, str(e)))

    def _done(self, success: bool, message: str):
        self.status_label.config(text="Done" if success else "Error")
        dialog = messagebox.showinfo if success else messagebox.showerror
        dialog("Analysis Result", message)

if __name__ == "__main__":
    Spec4AIGUI().mainloop()
