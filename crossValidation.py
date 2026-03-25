import pandas as pd

import evaluation
from naiveBayes import *


# This function takes in the data for cross evaluation and the number of partitions to split the data into.
# As input, the function takes:
# - training_data     - a pandas DataFrame containing the data set to be split
# - f                 - the number of partitions to split the data into, value is greater than 0,
#                       not guaranteed to be smaller than data size. If f exceeds the size of the data,
#                       it is capped at data size.
# As output, it produces:
# - partition_list   - a list of pandas DataFrames, where each data frame represents a partition, so a subset of entries
#                     of the original dataset s.t. all partitions are disjoint, roughly same size (can differ by
#                     at most 1), and the union of all partitions equals the original dataset. The indexing is
#                     preserved - i.e. row with row name/index 12 in the original dataset will have
#                     same name whatever partition it is in. The column and row names in the partition are the same
#                     as in training_data.
def partition_data(training_data: pd.DataFrame, f: int) -> list[pd.DataFrame]:
    
    dataSetLength = len(training_data)

    # Return empty list if data set is empty or f is less than 1
    if dataSetLength == 0 or training_data.columns.size == 0:
        return []

    # Restrict f between 2 and data size
    f = min(max(f, 2), dataSetLength)

    # Calculate the number of records in each partition and the number of partitions with an extra record
    partitionLength = dataSetLength // f
    largerPartitions = dataSetLength % f

    # Create f partitions
    partition_list = []
    for startIndex in range(f):

        # Create empty partition
        partition = pd.DataFrame(columns = training_data.columns)

        # Select records evenly from training data
        stop = startIndex + f * partitionLength + (f if startIndex < largerPartitions else 0)
        for index in range(startIndex, stop, f):

            # Extract record from training data
            partition.loc[index] = training_data.iloc[index]

        # Append the partition to the list
        partition_list.append(partition)

    return partition_list


# This function transforms partitions into training and testing data for each cross-validation round (there are
# as many rounds as there are partitions); in other words, we prepare the folds. The column and row names of the
# new testing and training datasets are preserved.
# At input, the function takes:
# - partition_list - a list of data frames, where each data frame represents a partition (see partition_data function)
# - f - the number of folds to use in cross-validation, which is the same as the number of partitions
#       the data was supposed to be split to, and the number of rounds in cross-validation. Value is greater than 0.
#
# The function produces:
# - folds - a list of 3-tuple s.t. the first element is the round number, second is the training data for that round,
#           and third is the testing data for that round. The round numbers START WITH 0.
#           The indexing is preserved - i.e. row with row name/index 12 in the original dataset will have
#           same name whatever fold it is in

def arrange_data_for_cv(partition_list: list[pd.DataFrame], f: int) \
        -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    
    # This is just for error handling, if for some magical reason f and number of partitions are not the same,
    # then something must have gone wrong in the other functions
    if len(partition_list) != f:
        print("Something went really wrong! Why is the number of partitions different from f??")
        return []
    
    # Create the folds
    folds = []
    for foldNumber in range(f):
        
        # Extract training and testing data from partitions for current fold
        trainingData = pd.concat(partition_list[:foldNumber] + partition_list[foldNumber + 1:])
        testingData = partition_list[foldNumber]
        folds.append((foldNumber, trainingData, testingData))
    
    return folds


# This function takes the lists of actual and predicted classes for each round, and produces averaged metrics.
#
# At input, it takes:
# - actual_class_list, predicted_class_list
#                           - lists of pandas Series representing the actual and predicted classes
#                           for each cross validation round
#        class_values - the list of all possible class values
# Function outputs:
# - computed measures - a dictionary of measures, explicitly listing 'average_macro_precision', 'average_macro_recall',
#                       'average_macro_f_measure', 'average_weighted_precision', 'average_weighted_recall',
#                       'average_weighted_f_measure', 'average_standard_accuracy' and 'average_balanced_accuracy'

def evaluate_results(actual_class_list: list[pd.Series], predicted_class_list: list[pd.Series],
                     class_values: list[str]) -> dict[str, float]:
    results = {'avg_macro_precision': 0.0, 'avg_macro_recall': 0.0, 'avg_macro_f_measure': 0.0,
               'avg_weighted_precision': 0.0, 'avg_weighted_recall': 0.0,
               'avg_weighted_f_measure': 0.0, 'avg_standard_accuracy': 0.0,
               'avg_balanced_accuracy': 0.0}
    
    # Calculate the number of rounds
    roundCount = len(actual_class_list)

    # Calculate metrics for each round
    for roundNumber in range(roundCount):
        roundResults = task_2_evaluation.evaluate_classification(actual_class_list[roundNumber], predicted_class_list[roundNumber], class_values)

        # Add metrics for current round to results
        for metric in roundResults.keys():
            results['avg_' + metric] += roundResults[metric]

    # Calculate average of metrics
    for metric in results.keys():
        results[metric] /= roundCount

    return results


# This function performs and evaluates cross-validation on a given dataset.
# It partitions the input dataset into f partitions, then arranges them into training and testing
# data for each cross validation round, and then trains and executes naive Bayes for each round using this data.
#
# It then produces an output dataset which extends the original input training_data by adding
# "PredictedClass" and "Fold" columns, which for each entry state what class it got predicted when it
# landed in a testing fold and what the number of that fold was. This is paired with a dictionary listing 
# average evaluation measures.
# 
# At input, the function takes:
# - nb - naive Bayes classifier
# - training_data - a pandas DataFrame representing the data
# - partition_func - the function used to partition the input dataset (by default, it is the one above)
# - prep_func - the function used to transform the partitions into appropriate folds (by default, it is the one above)
#  -eval_func - the function used to evaluate cross validation (by default, it is the one above)
#
# As output, it produces a tuple consisting of
# - output_dataset - a pandas DataFrame which extends the original input training_data by adding "PredictedClass"
#                    and "Fold" columns, which for each entry state what class it got predicted when it
#                    landed in a testing fold and what the number of that fold was (numbering starts from 0).
# - evaluation metrics - average evaluation metrics as produced by eval_func
def cross_validate(nb: NaiveBayes, training_data: pd.DataFrame, f: int,
                   partition_func=partition_data, prep_func=arrange_data_for_cv, eval_func=evaluate_results) \
        -> tuple[pd.DataFrame, dict[str, float]]:
    
    dataSetLength = len(training_data)

    # Extract name of class column
    className = nb.class_info[0]

    # Check that training data is not empty and nb classifier columns match training data
    nbColumnNames = list(nb.feature_info.keys()) + [(nb.class_info[0])]
    if dataSetLength == 0 or not all(col in training_data.columns for col in nbColumnNames):
        return

    # Restrict f between 2 and data size
    f = min(max(f, 2), dataSetLength)

    # Partition training data into f partitions
    partition_list = partition_func(training_data, f)

    # Arrange the partitions into training and testing data for each cross validation round
    folds = prep_func(partition_list, f)

    # Create lists of actual and predicted classes for evaluation
    actualClassList = []
    predictedClassList = []

    # Create output dataset and add predicted class and fold columns
    output_dataset = training_data.copy()
    output_dataset["PredictedClass"] = None
    output_dataset["Fold"] = -1

    # Train and execute naive Bayes for each round
    for foldNumber in range(f):

        # Extract training and testing data for current fold
        foldTrainingData = folds[foldNumber][1]
        foldTestingData = folds[foldNumber][2]

        # Train classifier with training data
        nb.train_model(foldTrainingData)

        # Predict classes for testing data
        predictedClasses = nb.predict(foldTestingData)["PredictedClass"]

        # Add predicted class and fold columns to output dataset
        output_dataset.loc[foldTestingData.index, "PredictedClass"] = predictedClasses
        output_dataset.loc[foldTestingData.index, "Fold"] = foldNumber

        # Add actual and predicted classes to lists for evaluation
        actualClassList.append(foldTestingData[className])
        predictedClassList.append(predictedClasses)

    # Evaluate cross validation
    evaluation = eval_func(actualClassList, predictedClassList, nb.class_info[1])

    return output_dataset, evaluation
