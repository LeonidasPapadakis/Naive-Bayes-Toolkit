# Naive-Bayes-Toolkit
 A Python implementation of a Naive Bayes classifier with k-fold cross-validation and multi-metric evaluation. Built for my Informatics module at Cardiff University.
 
 Built using `pandas`.
 
---
 
## Project Structure
 
| File | Description |
|---|---|
| `naiveBayes.py` | Core Naive Bayes classifier |
| `evaluation.py` | Confusion matrix and classification metrics |
| `crossValidation.py` | K-fold cross-validation pipeline |
 
---
 
### Requirements
 
- Python 3.10+
- pandas
 
Install dependencies with:
 
```bash
pip install pandas
```
 
### Basic Usage
 
```python
import pandas as pd
from naiveBayes import NaiveBayes
from crossValidation import cross_validate
 
# Define class and feature schema
class_info = ("Mammal", ["Yes", "No"])
feature_info = {
    "LivesInWater":  ["Yes", "No", "Sometimes"],
    "GivesBirth":     ["Yes", "No"]
}
 
# Load your dataset
data = pd.read_csv("your_dataset.csv")
 
# Run cross-validation
nb = NaiveBayes(class_info, feature_info)
results, metrics = cross_validate(nb, data, f=10)
 
print(results)
print(metrics)
```
 
---
 
## Module Details
 
### `naiveBayes.py` — Naive Bayes Classifier
 
Implements a categorical Naive Bayes classifier from scratch.
 
**Class: `NaiveBayes`**
 
| Method | Description |
|---|---|
| `__init__(class_info, feature_info)` | Initialises the classifier with class and feature schema |
| `train_model(training_data)` | Trains the model by computing all required probabilities |
| `predict(data)` | Returns the input DataFrame extended with a `PredictedClass` column |
| `retrieve_class_probability(class_value)` | Returns the prior probability of a given class |
| `retrieve_conditional_probability(class_value, feature_name, feature_value)` | Returns a conditional probability P(feature=value \| class) |
 
- Supports any number of categorical features and class values.
- Ties in prediction are broken alphabetically.
- Any new call to `train_model` replaces previously learned probabilities.
 
---
 
### `evaluation.py` — Classification Metrics
 
Provides a full suite of evaluation tools for multi-class classifiers.
 
**Key functions:**
 
| Function | Description |
|---|---|
| `confusion_matrix(actual, predicted, class_values)` | Builds a confusion matrix (rows = predicted, columns = actual) |
| `compute_TPs / compute_FPs / compute_FNs(matrix)` | Per-class true/false positives and false negatives |
| `compute_binary_precision/recall/f_measure(tp, fp, fn)` | Binary evaluation measures |
| `compute_macro_precision/recall/f_measure(matrix)` | Macro-averaged metrics |
| `compute_weighted_precision/recall/f_measure(matrix)` | Weighted-averaged metrics |
| `compute_standard_accuracy(matrix)` | Overall accuracy |
| `compute_balanced_accuracy(matrix)` | Balanced accuracy (mean per-class recall) |
| `evaluate_classification(actual, predicted, class_values)` | Returns a dictionary of all the above metrics |
 
---
 
### `crossValidation.py` — K-Fold Cross-Validation
 
Orchestrates the full cross-validation pipeline.
 
**Key functions:**
 
| Function | Description |
|---|---|
| `partition_data(data, f)` | Splits the dataset into `f` roughly equal, disjoint partitions |
| `arrange_data_for_cv(partitions, f)` | Combines partitions into `(fold_number, train, test)` tuples |
| `evaluate_results(actual_list, predicted_list, class_values)` | Averages evaluation metrics across all folds |
| `cross_validate(nb, data, f, ...)` | Runs the full pipeline; returns extended dataset and averaged metrics |
 
The output dataset from `cross_validate` appends two columns to the original data:
- `PredictedClass` — the class predicted when that row was in the test fold
- `Fold` — the fold number in which that row was tested (0-indexed)
 
All three pipeline functions (`partition_func`, `prep_func`, `eval_func`) accept custom implementations, making the pipeline modular and extensible.
 
---
 
## Evaluation Metrics Returned
 
`cross_validate` returns a dictionary containing the following averaged metrics:
 
```
avg_macro_precision       avg_weighted_precision
avg_macro_recall          avg_weighted_recall
avg_macro_f_measure       avg_weighted_f_measure
avg_standard_accuracy     avg_balanced_accuracy
```
 
---
 
## Notes
 
- All feature and class values must be declared upfront in the schema passed to `NaiveBayes`.
- Unseen feature values during prediction are assigned a conditional probability of 0.
- If `f` exceeds the size of the dataset, it is automatically capped at the dataset size.
- The partitioning preserves original DataFrame index labels throughout.
 
