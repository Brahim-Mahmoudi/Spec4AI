##  Available Rules

SpecDetect4AI supports detection of 24 AI-specific code smells, grouped by ML concern:

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