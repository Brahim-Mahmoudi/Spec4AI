![Overview](static/AI-based%20system.png)


# Spec4AI CLI

This command-line interface (CLI) allows you to run Spec4AI's static code analysis rules on any Python project directory.

##  Usage

```bash
python spec4ai.py --input-dir <path> [--output-file <file>] [--rules R2 R6] [--all] [--exclude R6] [--summary] [--list-rules]
```

##  Examples

### Analyze a folder using all rules
```bash
python spec4ai.py --input-dir ./my_project --all
```

### Analyze with specific rules only
```bash
python spec4ai.py --input-dir ./my_project --rules R2 R6 R11
```

### Analyze all rules except some
```bash
python spec4ai.py --input-dir ./my_project --all --exclude R6 R11
```

### Save results to a custom file
```bash
python spec4ai.py --input-dir ./my_project --all --output-file results.json
```

### List all available rules
```bash
python spec4ai.py --list-rules
```

### Show a summary of alerts by rule
```bash
python spec4ai.py --input-dir ./my_project --all --summary
```

##  Output
- A JSON file containing detected issues, grouped by file and rule
- CLI printout showing:
  - Number of `.py` files scanned
  - Number of files with alerts
  - (Optional) Summary of alerts per rule

##  Project Structure Assumptions
- Rules are located in subfolders of `test_rules/` (e.g. `test_rules/R2/generated_rules_R2.py`)
- Each rule module must define a function named `rule_<ID>` and a global `report()` function used for collecting messages.

##  Requirements
- Python 3.9+
- Lark 1.2.2
- python -m pip install -r requirements.txt
- The `test_rules/` directory must exist and contain valid rule implementations
### Example : 

- python -m venv .venv
- source .venv/bin/activate  # or .venv\\Scripts\\activate sur Windows
- python -m pip install -r requirements.txt


##  Notes
- The CLI supports aliasing (e.g., `R11bis` counts as `R11`)
- The number of analyzed files includes all `.py` files recursively found under `--input-dir`

from pathlib import Path


##  Available Rules

Spec4AI supports detection of 24 AI-specific code smells, grouped by ML concern:

| Rule ID | Name                                                                                   |
|---------|-----------------------------------------------------------------------------------------|
| R1      | Broadcasting Feature Not Used                                                          |
| R2      | Random Seed Not Set                                                                    |
| R3      | TensorArray Not Used                                                                   |
| R4      | Training / Evaluation Mode Improper Toggling                                           |
| R5      | Hyperparameter Not Explicitly Set                                                      |
| R6      | Deterministic Algorithm Option Not Used                                                |
| R7      | Missing the Mask of Invalid Value (e.g., in `tf.log`)                                  |
| R8      | PyTorch Call Method Misused                                                            |
| R9      | Gradients Not Cleared before Backward Propagation                                      |
| R10     | Memory Not Freed                                                                        |
| R11     | Data Leakage (fit_transform before split)                                              |
| R11bis  | Data Leakage (no pipeline in presence of model.fit)                                   |
| R12     | Matrix Multiplication API Misused (np.dot vs np.matmul)                               |
| R13     | Empty Column Misinitialization                                                         |
| R14     | Dataframe Conversion API Misused (e.g., `.values`)                                     |
| R15     | Merge API Parameter Not Explicitly Set                                                 |
| R16     | API Misuse (e.g., missing inplace or reassignment)                                     |
| R17     | Unnecessary Iteration                                                                  |
| R18     | NaN Comparison (`== np.nan`)                                                            |
| R19     | Threshold Validation Metrics Count                                                     |
| R20     | Chain Indexing on DataFrames                                                           |
| R21     | Columns and DataType Not Explicitly Set in pd.read                                     |
| R22     | No Scaling Before Scale-Sensitive Operations                                           |
| R23     | EarlyStopping Not Used in Model.fit                                                    |
| R24     | Index Column Not Explicitly Set in DataFrame Read                                     |

---

## 🔍 Predicate Library

The DSL uses semantic predicates over AST nodes. Below is a list of predicates used across the rules:

### 📌 General AST Analysis
- `exists ... in AST: (...)`
- `count(...)`
- `not (...)`
- `and`, `or`, etc.

### 🧩 Predicate Functions

#### Model & Fit
- `isModelFitPresent(node)`
- `isFitCall(node)`
- `isMLMethodCall(node)`
- `hasEarlyStoppingCallback(node)`
- `hasExplicitHyperparameters(node)`

#### Data Pipelines
- `pipelineUsed(node)`
- `pipelineUsedGlobally(node)`
- `usedBeforeTrainTestSplit(node)`

#### Randomness / Determinism
- `isRandomCall(node)`
- `seedSet(node)`
- `isSklearnRandomAlgo(node)`
- `hasRandomState(node)`
- `customCheckTorchDeterminism(ast_node)`

#### PyTorch / TensorFlow
- `isEvalCall(node)`
- `hasLaterTrainCall(node)`
- `isForwardCall(node)`
- `isLossBackward(node)`
- `hasPrecedingZeroGrad(node)`
- `isPytorchTensorUsage(node)`
- `isModelCreation(node)`
- `isMemoryFreeCall(node)`
- `isLog(node)`
- `hasMask(node)`

#### DataFrame / Pandas
- `isDataFrameMerge(node)`
- `singleParam(node)`
- `isApiMethod(node)`
- `hasInplaceTrue(node)`
- `isResultUsed(node)`
- `isPandasReadCall(node)`
- `hasKeyword(node, kw)`
- `isDataFrameVariable(var, node)`
- `isSubscript(node)`
- `get_base_name(node)`
- `usesIterrows(node)`
- `usesItertuples(node)`
- `isValuesAccess(node)`
- `isDataFrameColumnAssignment(node)`
- `isAssignedLiteral(node, value)`

#### Numpy / Math
- `isCompare(node)`
- `hasNpNanComparator(node)`
- `isDotCall(node)`
- `isMatrix2D(node)`

#### Scaling / Metrics
- `isScaleSensitiveFit(node, vars)`
- `hasPrecedingScaler(node, vars)`
- `isPartOfValidatedPipeline(node)`
- `isMetricCall(node)`
- `isThresholdDependent(node)`
- `isThresholdIndependent(node)`

#### Looping
- `isForLoop(node)`
- `isFunctionDef(node)`
- `hasConstantAndConcatIntersection(node)`
- `usesPythonLoopOnTensorFlow(node)`








Developed as part of the Spec4AI project to support AI-specific code smell detection.
