import os
import time
from openai import OpenAI
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import List
import json
import re



# === CONFIGURATION ===
GROQ_API_KEY = "API/KEY"
GROQ_MODEL = "llama3-70b-8192"
TARGET_DIR = "TARGET DIR PATH"
OUTPUT_EXCEL = "llama3-70b-8192_code_smells_matrix.xlsx"

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

RULES = [f"R{i}" for i in range(1, 23)]

SYSTEM_PROMPT = SYSTEM_PROMPT = """
You are an expert in Python code quality for AI and machine learning systems.  
Your task is to identify violations of 22 known ML-specific code smell rules (R1 to R22) in Python code.

Each rule has a unique identifier (e.g., R2) and applies to specific programming patterns.

---

 Rule are available here : https://hynn01.github.io/ml-smells/.
A short descriptions with examples here :

- R1: Broadcasting Feature Not Used  
  → Using manual loops instead of NumPy/PyTorch broadcasting.  
  Ex: for i in range(len(a)): c[i] = a[i] + b[i]

- R2: Randomness Uncontrolled  
  → Using randomness (e.g., np.random, torch.rand) without setting a seed.  
  Ex: np.random.rand() without np.random.seed()

- R3: TensorArray Not Used  
  → Manually managing lists of tensors instead of dedicated structures.

- R4: Training / Evaluation Mode Improper Toggling  
  → Forgetting model.eval() or model.train() when switching between training/inference.  
  Ex: Inference without calling model.eval()

- R5: Hyperparameter Not Explicitly Set  
  → Instantiating ML models without explicitly setting hyperparameters.  
  Ex: LogisticRegression() instead of LogisticRegression(C=1.0)

- R6: Deterministic Algorithm Option Not Used  
  → Missing call to torch.use_deterministic_algorithms(True)

- R7: Missing the Mask of Invalid Value  
  → Not masking or filtering invalid (e.g., NaN) values before training.

- R8: PyTorch Call Method Misused  
  → Incorrect use of PyTorch models or misuse of the forward method.

- R9: Gradients Not Cleared Before Backward Propagation  
  → Missing optimizer.zero_grad() before loss.backward()

- R10: Memory Not Freed  
  → Forgetting to detach() or clear memory/session after training.

- R11: Data Leakage  
  → Data transformation before train_test_split without using a Pipeline.  
  Ex: scaler.fit_transform() before splitting data

- R12: Matrix Multiplication API Misused  
  → Using * instead of @ for matrix multiplication.

- R13: Empty Column Misinitialization  
  → Creating empty columns with [] or None in DataFrames.

- R14: Dataframe Conversion API Misused  
  → Wrong use of DataFrame to NumPy conversion (e.g., df.values instead of df.to_numpy())

- R15: Merge API Parameter Not Explicitly Set  
  → Using merge/join without specifying a key (e.g., on=...)

- R16: In-Place APIs Misused  
  → Overusing inplace=True in pandas.

- R17: Unnecessary Iteration  
  → Using explicit loops over DataFrames or arrays when vectorization is better.

- R18: NaN Equivalence Comparison Misused  
  → Comparing with np.nan using == instead of np.isnan()

- R19: Threshold-Dependent Validation  
  → Hardcoded thresholds in validation (e.g., if acc > 0.9)

- R20: Chain Indexing  
  → Using chained indexing in pandas (e.g., df[df["x"] > 0]["y"])

- R21: Columns and DataType Not Explicitly Set  
  → Adding DataFrame columns without specifying types or names

- R22: No Scaling Before Scaling-sensitive Operation  
  → Using scale-sensitive models (e.g., SVM) without scaling inputs

---

 Common rule confusions (do not mix):

- R2 (random seed missing) ≠ R6 (deterministic algorithm setting)
- R5 (model parameters missing) ≠ R21 (DataFrame columns without dtype)

---

 Input format:

You’ll be given a code block with line numbers:

1: from sklearn.linear_model import LogisticRegression  
2: model = LogisticRegression()  
3: model.fit(X_train, y_train)

---

 Output format (strict JSON only):

Return a JSON object mapping line numbers to a list of violated rule IDs:

{
  "1": [],
  "2": ["R5"],
  "3": []
}

---

 Instructions:

-  Return a list of violated rule IDs for each line.
-  Use [] if no rule is violated.
-  Do NOT explain, justify, or comment.
-  Do NOT return anything outside the JSON.
-  Follow the format strictly — only valid JSON output is accepted.
"""


def analyze_block_with_groq(lines: List[str], start_index: int) -> dict:
   
    code_block = "\n".join([f"{i+start_index+1}: {line.strip()}" for i, line in enumerate(lines)])
    prompt = f"Here a python excerpt :\n```python\n{code_block}\n```"

    max_retries = 5
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            json_text = response.choices[0].message.content
            #smell_dict = eval(json_text.strip())
            smell_dict = json.loads(json_text.strip())

            return smell_dict
        
        except json.JSONDecodeError as json_err:
            print(f" invalide JSON  returned by Groq (bloc {start_index+1}-{start_index+len(lines)}): {json_err}")
            print(f"Received content:\n{json_text}")
            return {}
        except Exception as e:
            error_msg = str(e)
            if "rate_limit_exceeded" in error_msg or "Rate limit" in error_msg:
                wait_time = 75  # Valeur par défaut
      
            
                match = re.search(r"try again in ([0-9]+)m([0-9.]+)s", error_msg)
                if match:
                    minutes = int(match.group(1))
                    seconds = float(match.group(2))
                    wait_time = int(minutes * 60 + seconds) + 2
                print(f" Rate limit , wait {wait_time} seconds...")
                time.sleep(wait_time)
                retry_count += 1
            else:
                print(f" Error of analyze (bloc {start_index+1}-{start_index+len(lines)}): {e}")
                return {}
    print(f" Fail after {max_retries} tries for bloc {start_index+1}-{start_index+len(lines)}.")
    return {}


def detect_smells_in_file(filepath: Path) -> dict:
    results = defaultdict(list)
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    batch_size = 20
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i+batch_size]
        batch_result = analyze_block_with_groq(batch, i)
        for lineno_str, rules in batch_result.items():
            for rule in rules:
                results[rule].append(lineno_str)

    return results  # dict { "R2": ["4", "18"], ... }

def main():
    matrix_data = []

    for root, _, files in os.walk(TARGET_DIR):
        for file in files:
            if file.endswith(".py"):
                filepath = Path(root) / file
                print(f"Analyze of {filepath}")
                start = time.time()
                file_result = detect_smells_in_file(filepath)
                duration = round(time.time() - start, 2)

                row = {"file": str(filepath), "Execution Time (s)": duration}
                for rule in RULES:
                    row[rule] = ";".join(file_result[rule]) if rule in file_result else ""
                matrix_data.append(row)

    df = pd.DataFrame(matrix_data)
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f" Exported files: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()
