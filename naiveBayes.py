import pandas as pd


class NaiveBayes:

    # This function simply initializes an instance of NaiveBayes class. The constructor takes at input:
    # - class_info - pair that contains the name of the class column and its permitted values
    # - feature_info - dictionary that states attribute names and their permitted values

    def __init__(self, class_info: tuple[str, list[str]], feature_info: dict[str, list[str]]):
        self.class_info = class_info
        self.feature_info = feature_info
        
        # Create a dataframe to store probabilities
        self.probabilities = pd.DataFrame(columns = class_info[1], index = [class_info[0]] + list(feature_info.keys()))

    # This function trains the model, aka calculates all the necessary probabilities that a naive Bayes model needs.
    # For the purpose of this task, numerical values are treated just like categorical ones. Any new training
    # purges old data.
    # At input, train_model takes:
    # - training_data - a pandas DataFrame that contains all the attribute values and class value for a given entry
    def train_model(self, training_data: pd.DataFrame):                    
    
        # Loop through class values
        for classValue in self.class_info[1]:

            # Initialize class probability with placeholder
            self.probabilities[classValue][self.class_info[0]] = 0.0 

            # Isolate class data
            try:
                classData = training_data[training_data[self.class_info[0]] == classValue]
            except KeyError:
                continue
            
            # Calculate the class probability, handling division by zero
            if len(training_data) > 0:
                self.probabilities[classValue][self.class_info[0]] = len(classData) / len(training_data)

            # Loop through features
            for feature in self.feature_info.keys():

                # Initialize feature probabilities dictionary
                self.probabilities[classValue][feature] = {} 

                # Loop through feature values
                for featureValue in self.feature_info[feature]:

                    # Initialize feature probability with placeholder
                    self.probabilities[classValue][feature][featureValue] = 0.0
                    
                    # Calculate the feature probability given the class, handling division by zero
                    if len(classData) > 0:
                        self.probabilities[classValue][feature][featureValue] = (classData[feature] == featureValue).sum() / len(classData)
                    
        
    # This function predicts the classes for entries in the training_data and produces an extended data frame.
    # At input, it takes:
    # - training_data - a pandas DataFrame that contains all the attribute values and class value for a given entry
    # The function outputs:
    # classified_data - a pandas DataFrame which expands the training_data by adding the "PredictedClass" column
    #                   that for every entry states the class value predicted for that entry. In case of ties,
    #                   the chosen class is the one that appears earlier alphabetically.
    def predict(self, training_data: pd.DataFrame) -> pd.DataFrame:

        # Loop through records and predict their class
        predictions = []
        for index in training_data.index:

            # Track largest class probability
            largest = ["", -1]

            # Loop through class values
            for classValue in self.class_info[1]:
                
                # Loop through calculated probabilities and build probability for class value
                classProb = self.probabilities[classValue][self.class_info[0]]
                for feature in self.feature_info.keys():

                    # Check if feature is present in training data
                    if feature not in training_data.columns:
                        continue

                    # Get feature value
                    value = training_data.loc[index, feature]
                    if isinstance(value, pd.Series): # Handle duplicate index
                        value = value.iloc[0]

                    # Update class probability
                    classProb *= self.probabilities.loc[feature, classValue].get(value, 0)

                # Update largest class probability
                if classProb > largest[1]:
                    largest[0] = classValue
                    largest[1] = classProb

                # Tie breaker
                elif classProb == largest[1]:
                    if classValue < largest[0]:
                        largest[0] = classValue

            # Add predicted class to predictions
            predictions.append(largest[0])

        # Add predicted class column to output
        classified_data = training_data.copy()
        classified_data["PredictedClass"] = predictions
        return classified_data

    # The function returns the probability of a given class value. A value of 0 is 
    # returned if no training took place.
    # At input, it takes:
    # - class_value - the class value for which we want to calculate the probability
    # The function outputs:
    # - probability - float representing the probability of the given class value
    def retrieve_class_probability(self, class_value: str) -> float:
        
        # Return 0 if no training took place
        if class_value not in self.probabilities.columns:
            return 0.0
        
        # Retreive class probability
        prob = self.probabilities.loc[self.class_info[0], class_value]

        # Return 0 if probability is somehow NaN
        if pd.isna(prob):
            return 0.0
        
        return prob

    # The function returns the conditional probably of a feature value assuming a given class 
    # value. A value of 0 is returned if no training took place.
    # At input, it takes:
    # - class_value - the class value on which the feature_value is conditional
    # - feature_name - the name of the feature we want to calculate for
    # - feature_value - the feature value we want to calculate the conditional probability for
    # The function outputs:
    # - probability - float representing the calculated conditional probability
    #
    def retrieve_conditional_probability(self, class_value: str, feature_name: str, feature_value: str) -> float:
        
        # Return 0 if no training took place
        if class_value not in self.probabilities.columns:
            return 0.0
        
        # Return 0 if feature does not exist
        if feature_name not in self.probabilities.index:
            return 0.0
        
        # Retreive probabilities dictionary
        probs = self.probabilities.loc[feature_name, class_value]

        # Return 0 if dictionary does not exist
        if not isinstance(probs, dict):
            return 0.0
        
        # Retreive conditional probability
        prob = probs.get(feature_value, 0.0)

        # Return 0 if probability is somehow NaN
        if pd.isna(prob):
            return 0.0
        
        return prob
