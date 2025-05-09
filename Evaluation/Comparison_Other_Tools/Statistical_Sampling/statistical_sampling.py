import os
import shutil
import random
import csv


SOURCE_DIR = "mlflow-master"
DEST_DIR = "mlflow_projet_sampled"
NUM_FILES = 241
RANDOM_SEED = 42
LOG_FILE = os.path.join(DEST_DIR, "selected_files.csv")


os.makedirs(DEST_DIR, exist_ok=True)


all_py_files = []
for dirpath, _, filenames in os.walk(SOURCE_DIR):
    for filename in filenames:
        if filename.endswith(".py"):
            full_path = os.path.join(dirpath, filename)
            all_py_files.append(full_path)


if len(all_py_files) < NUM_FILES:
    raise ValueError(f"Only {len(all_py_files)} .py file  found, need at least {NUM_FILES}.")


random.seed(RANDOM_SEED)#Reproductibility
selected_files = random.sample(all_py_files, NUM_FILES)


used_names = set()
log_entries = []

for full_path in selected_files:
    filename = os.path.basename(full_path)
    base, ext = os.path.splitext(filename)
    dest_filename = filename
    i = 1
    while dest_filename in used_names:
        dest_filename = f"{base}_{i}{ext}"
        i += 1

    dest_path = os.path.join(DEST_DIR, dest_filename)
    shutil.copy2(full_path, dest_path)
    used_names.add(dest_filename)

    
    log_entries.append([dest_filename, full_path])


with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Nom du fichier copié", "Chemin source complet"])
    writer.writerows(log_entries)

print(f" {NUM_FILES} extracted .py files  from {SOURCE_DIR} and past in {DEST_DIR}")
print(f"  Log in  : {LOG_FILE}")
