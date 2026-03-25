import pandas as pd


# This function computes the confusion matrix based on the provided series of actual and predicted classes.
# The returned data frame contains appropriate column and row names, and is filled with integers.
# The columns correspond to actual classes and rows to predicted classes, in the sense that the i-th row
# is the row representing how often entries actually belonging to some class, were predicted as the i-th class value;
# the i-th column represents how often entries predicted as some other class, actually belonged to the i-th class.
#
# At input, function takes:
# - actual_class, predicted_class - series of class values representing actual and predicted classes of some dataset.
# - class_values - all possible values of the class from which actual_class and predicted_class were drawn.
#
# As output, it produces:
# - matrix : a data frame representing the confusion matrix computed based on the offered series of actual
#            and predicted classes.

def confusion_matrix(actual_class: pd.Series, predicted_class: pd.Series, class_values: list[str]) -> pd.DataFrame:
    
    # Create new matrix
    matrix = pd.DataFrame(0, index = class_values, columns = class_values)
    
    # Loop through actual_class and predicted_class 
    for index in actual_class.index:

        # Check that both values are in class_values
        if actual_class[index] not in class_values or predicted_class[index] not in class_values:
            continue

        # Increment corresponding cell in matrix
        matrix.loc[predicted_class[index], actual_class[index]] += 1   
        
    return matrix


# These functions compute per-class true positives and false positives/negatives based on the provided confusion matrix.
#
# As input, these functions take:
# - matrix - a data frame representing the confusion matrix computed based on the offered series of actual
#            and predicted classes. See confusion_matrix function for description.
#
# As output, these functions produce:
# - tps/fps/fns - dictionaries that for every class value in the classification scheme (corresponding to names of
#                 all rows and/or all columns in the matrix) return the true positive, false positive or
#                 false negative values for that class.

def compute_TPs(matrix: pd.DataFrame) -> dict[str, int]:
    
    TPs = {}

    # True positives are on the diagonal
    for classValue in matrix.index:
        TPs[classValue] = matrix.loc[classValue, classValue]

    return TPs


def compute_FPs(matrix: pd.DataFrame) -> dict[str, int]:
    
    FPs = {}

    # False positives are on the same row, excluding diagonal
    for classValue in matrix.index:
        FPs[classValue] = matrix.loc[classValue, :].sum() - matrix.loc[classValue, classValue]

    return FPs


def compute_FNs(matrix: pd.DataFrame) -> dict[str, int]:
    
    FNs = {}

    # False negatives are in the same column, excluding diagonal
    for classValue in matrix.index:
        FNs[classValue] = matrix.loc[:, classValue].sum() - matrix.loc[classValue, classValue]

    return FNs


# These functions compute the binary measures based on the provided values. Not all measures use all the values.
# At input, the functions take:
# - tp, fp, fn : the single values of true positives, false positive and negatives
#
# As output, they produce:
# - binary precision/recall/f-measure - appropriate evaluation measure created using the binary approach.

def compute_binary_precision(tp: int, fp: int, fn: int) -> float:

    # Compute precision
    if (tp + fp) > 0:
        return tp / (tp + fp)
    
    # Return 0 if no data
    else:
        return 0.0


def compute_binary_recall(tp: int, fp: int, fn: int) -> float:

    # Compute recall
    if (tp + fn) > 0:
        return tp / (tp + fn)
    
    # Return 0 if no data
    else:
        return 0.0


def compute_binary_f_measure(tp: int, fp: int, fn: int) -> float:
    
    # Compute precision and recall
    p = compute_binary_precision(tp, fp, fn)
    r = compute_binary_recall(tp, fp, fn)

    # Compute f-measure
    if (p + r) > 0:
        return 2 * p * r / (p + r)
    
    # Handle division by zero
    else:
        return 0.0

# These functions compute the macro precision, macro recall, macro f-measure, based on the offered confusion matrix.
#
# As input, these functions take:
# - matrix - a data frame representing the confusion matrix computed based on the offered series of actual
#            and predicted classes. See confusion_matrix function for description.
# As output, they produce:
# - macro precision/recall/f-measure - appropriate evaluation measures created using the macro average approach.

def compute_macro_precision(matrix: pd.DataFrame) -> float:
    
    # Return 0 if matrix is empty
    if matrix.sum().sum() == 0:
        return 0.0
    
    # Compute true positives and false positives
    tp = compute_TPs(matrix)
    fp = compute_FPs(matrix)
    
    # Compute per-class precision
    precision = 0
    for classValue in matrix.index:
        
        # Compute and add binary precision for class
        precision += compute_binary_precision(tp[classValue], fp[classValue], None)

    precision /= len(matrix.index)

    return precision


def compute_macro_recall(matrix: pd.DataFrame) -> float:
    
    # Return 0 if matrix is empty
    if matrix.sum().sum() == 0:
        return 0.0
    
    # Compute true positives and false negatives
    tp = compute_TPs(matrix)
    fn = compute_FNs(matrix)
    
    # Compute per-class recall
    recall = 0
    for classValue in matrix.index:

        # Compute and add binary recall for class
        recall += compute_binary_recall(tp[classValue], None, fn[classValue])

    recall /= len(matrix.index)

    return recall


def compute_macro_f_measure(matrix: pd.DataFrame) -> float:
    
    # Return 0 if matrix is empty
    if matrix.sum().sum() == 0:
        return 0.0
    
    # Compute true positives, false positives and false negatives
    tp = compute_TPs(matrix)
    fp = compute_FPs(matrix)
    fn = compute_FNs(matrix)
    
    # Compute per-class f-measure
    fMeasure = 0
    for classValue in matrix.index:

        # Compute and add binary f-measure for class
        fMeasure += compute_binary_f_measure(tp[classValue], fp[classValue], fn[classValue])
        
    fMeasure /= len(matrix.index)

    return fMeasure


# These functions compute the weighted precision, macro recall, macro f-measure, based on the offered confusion matrix.
#
# As input, these functions take:
# - matrix - a data frame representing the confusion matrix computed based on the offered series of actual
#            and predicted classes. See confusion_matrix function for description.
# As output, they produce:
# - weighted precision/recall/f-measure - appropriate evaluation measures created using the weighted average approach.

def compute_weighted_precision(matrix: pd.DataFrame) -> float:

    # Return 0 if matrix is empty
    if matrix.sum().sum() == 0:
        return 0.0

    # Compute true positives and false positives
    tp = compute_TPs(matrix)
    fp = compute_FPs(matrix)
    
    # Compute per-class weighted precision
    precision = 0
    total = 0
    for classValue in matrix.index:

        # Compute class total
        classTotal = matrix[classValue].sum()
        total += classTotal

        # Compute and add binary precision for class
        precision += compute_binary_precision(tp[classValue], fp[classValue], None) * classTotal
    
    # Divide by total
    precision /= total
    
    return precision


def compute_weighted_recall(matrix: pd.DataFrame) -> float:
    
    # Return 0 if matrix is empty
    if matrix.sum().sum() == 0:
        return 0.0
    
    # Compute true positives and false positives
    tp = compute_TPs(matrix)
    fn = compute_FNs(matrix)
    
    # Compute per-class weighted recall
    recall = 0
    total = 0
    for classValue in matrix.index:

        # Compute class total
        classTotal = matrix[classValue].sum()
        total += classTotal

        # Compute and add binary recall for class
        recall += compute_binary_recall(tp[classValue], None, fn[classValue]) * classTotal

    recall /= total
    
    return recall


def compute_weighted_f_measure(matrix: pd.DataFrame) -> float:
    
    # Return 0 if matrix is empty
    if matrix.sum().sum() == 0:
        return 0.0
    
    # Compute true positives, false positives and false negatives
    tp = compute_TPs(matrix)
    fp = compute_FPs(matrix)
    fn = compute_FNs(matrix)
    
    # Compute per-class weighted f measure
    fMeasure = 0
    total = 0
    for classValue in matrix.index:

        # Compute class total
        classTotal = matrix[classValue].sum()
        total += classTotal

        # Compute and add binary f-measure for class
        fMeasure += compute_binary_f_measure(tp[classValue], fp[classValue], fn[classValue]) * classTotal
        
    fMeasure /= total

    return fMeasure


# These functions compute the standard and balanced multiclass accuracies based on the offered confusion matrix.
#
# As input, these functions take:
# - matrix - a data frame representing the confusion matrix computed based on the offered series of actual
#            and predicted classes. See confusion_matrix function for description.
# As output, they produce:
# - standard/balanced multiclass accuracy - appropriate evaluation measures created using the
#                                           standard/balanced approach.

def compute_standard_accuracy(matrix: pd.DataFrame) -> float:
    
    # Return 0 if matrix is empty
    if matrix.sum().sum() == 0:
        return 0.0

    # Compute true positives
    tp = compute_TPs(matrix)

    # Sum correct predictions
    correct = 0
    for classValue in matrix.index:
        correct += tp[classValue]

    # Compute standard accuracy
    accuracy = correct / matrix.sum().sum()

    return accuracy


def compute_balanced_accuracy(matrix: pd.DataFrame) -> float:
    
    # Return 0 if matrix is empty
    if matrix.sum().sum() == 0:
        return 0.0
    
    # Compute true positives and false negatives
    tp = compute_TPs(matrix)
    fn = compute_FNs(matrix)

    # Sum recall
    recall = 0
    for classValue in matrix.index:

        # Compute and add binary recall for class
        recall += compute_binary_recall(tp[classValue], None, fn[classValue])

    # Compute balanced accuracy
    accuracy = recall / len(matrix.index)

    return accuracy


# This function computes precision, recall, f-measure and accuracy of the classifier using
# the macro average approach.
# At input, the function takes:
# - actual_class - a pandas Series of actual class values
# - predicted_class - a pandas Series of predicted class values
# - class_values - a list of all possible class values
# - confusion_func - function to be invoked to compute the confusion matrix
# Function outputs:
# - computed measures - a dictionary of measures, explicitly listing 'macro_precision', 'macro_recall',
#                       'macro_f_measure', 'weighted_precision', 'weighted_recall', 'weighted_f_measure',
#                       'standard_accuracy' and 'balanced_accuracy'

def evaluate_classification(actual_class: pd.Series, predicted_class: pd.Series, class_values: list[str],
                            confusion_func=confusion_matrix) \
        -> dict[str, float]:
    
    matrix = confusion_func(actual_class, predicted_class, class_values)

    macro_precision = compute_macro_precision(matrix)
    macro_recall = compute_macro_recall(matrix)
    macro_f_measure = compute_macro_f_measure(matrix)

    weighted_precision = compute_weighted_precision(matrix)
    weighted_recall = compute_weighted_recall(matrix)
    weighted_f_measure = compute_weighted_f_measure(matrix)

    standard_accuracy = compute_standard_accuracy(matrix)
    balanced_accuracy = compute_balanced_accuracy(matrix)

    # Once ready, we return the values
    return {'macro_precision': macro_precision, 'macro_recall': macro_recall, 'macro_f_measure': macro_f_measure,
            'weighted_precision': weighted_precision, 'weighted_recall': weighted_recall,
            'weighted_f_measure': weighted_f_measure, 'standard_accuracy': standard_accuracy,
            'balanced_accuracy': balanced_accuracy}
