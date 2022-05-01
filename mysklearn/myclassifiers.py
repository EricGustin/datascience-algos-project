import copy
import math
from collections import Counter
from mysklearn import myutils as utils
import numpy as np

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.
    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors dict(str: dict(obj: dict(obj: float))): The posterior probabilities computed for each
            attribute value/label pair in the training set.
        labels
    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None
        self.labels = set()

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.
        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        if X_train and y_train:
            self.labels = set(y_train)
            self.priors = self._compute_priors(y_train)
            self.posteriors = self._compute_posteriors(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for row in X_test:
            probabilities = []
            for label in self.labels:
                probability = self.priors[label]
                for attribute_num, value in enumerate(row):
                    probability *= self.posteriors["att" + str(attribute_num + 1)][
                        value
                    ][label]
                probabilities.append((label, probability))
            y_predicted.append(max(probabilities, key=lambda x: x[1])[0])

        return y_predicted

    def _compute_priors(self, y_train):
        """
        Computes the priors for each label
        Parameters:
        -----------
        y_train list(obj)
        Returns:
        --------
        dict(obj: float)
        """
        label_occurrences = Counter(y_train)
        training_size = len(y_train)
        return {
            label: occurrences / training_size
            for label, occurrences in label_occurrences.items()
        }

    def _compute_posteriors(self, X_train, y_train):
        """
        Computes posterior probabilities for each attribute
        value/label pair in the training set. The structure of
        the posteriors instance attribute is a 3-dimensional
        dictionary. e.g.
        {
            attribute_name1: {
                value1: {
                    class_label1: posterior,
                    class_label2: posterior
                }
                value2: {
                    class_label1: posterior,
                    class_label2: posterior
                }
            },
            attribute_name2: {
                value3: {
                    class_label1: posterior,
                    class_label2: posterior
                }
                value4: {
                    class_label1: posterior,
                    class_label2: posterior
                }
            }
        }
        Parameters:
        -----------
        X_train: list(list(obj))
        y_train: list(obj)
        Returns:
        --------
        dict(str: dict(obj: dict(obj: float)))
        """
        attributes = ["att" + str(num + 1) for num in range(len(X_train[0]))]
        # get the value spaces for each attribute (all possible values for each attribute)
        attribute_to_value_space = {attribute: set() for attribute in attributes}
        for row in X_train:
            for attribute_num, value in enumerate(row):
                attribute_to_value_space["att" + str(attribute_num + 1)].add(value)
        # create nested dictionary structure for posteriors, initializing each posterior to zero
        posteriors = {attribute: dict() for attribute in attributes}
        for attribute in attributes:
            posteriors[attribute] = {
                value: dict() for value in attribute_to_value_space[attribute]
            }
            for value in attribute_to_value_space[attribute]:
                posteriors[attribute][value] = {label: 0 for label in self.labels}
        # compute each posterior's numerator and denominator, giving us all posteriors
        posterior_numerators = copy.deepcopy(posteriors)
        for row_num, row in enumerate(X_train):
            for attribute_num, value in enumerate(row):
                posterior_numerators["att" + str(attribute_num + 1)][value][
                    y_train[row_num]
                ] += 1
        for attribute in posteriors.keys():
            for label in self.labels:
                denominator = 0
                for value in posterior_numerators[attribute]:
                    denominator += posterior_numerators[attribute][value][label]
                for value in posteriors[attribute]:
                    posteriors[attribute][value][label] = (
                        posterior_numerators[attribute][value][label] / denominator
                    )
        return posteriors


