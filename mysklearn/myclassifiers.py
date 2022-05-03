import copy
import math
import random
import time
from collections import Counter
from mysklearn import myutils as utils
from mysklearn import myevaluation
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


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, header=None):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        self._class_labels = None
        self._header = header
        self._attribute_domains = None

    def _create_attribute_domains(self):
        """
        Sets the _attribute_domains dictionary.
        The _attribute_domains instance variable
        has the type dict(obj: set(obj)) and it
        serves the purpose of mapping attribute
        names to their domain i.e. to all possible
        values of the attribute
        """
        self._attribute_domains = {
            attribute_name: [] for attribute_name in self._header
        }
        for row in self.X_train:
            for column_number, value in enumerate(row):
                attribute = self._header[column_number]
                if value not in self._attribute_domains[attribute]:
                    self._attribute_domains[attribute].append(value)
        # sort domains alphabetically for reproducibility
        for attribute in self._attribute_domains.keys():
            self._attribute_domains[attribute].sort()

    def _select_attribute(self, attributes, instances):
        """
        Uses the entropy-based attribute selection method
        to select an attribute i.e. selects the attribute
        with the smallest E_new value
        Parameters:
        -----------
        attributes: list(obj)
        Returns:
        --------
        obj
        """
        new_entropies = []
        for attribute in attributes:
            attribute_entropies = []
            attribute_domain = self._attribute_domains[attribute]
            for value in attribute_domain:
                proportions = utils.compute_proportions(
                    attribute, value, instances, self._header, self._class_labels
                )
                entropy = utils.compute_entropy(proportions)
                attribute_entropies.append(entropy)
            likelihoods = utils.compute_likelihoods(
                attribute, attribute_domain, instances, self._header
            )
            new_entropy = utils.compute_new_entropy(likelihoods, attribute_entropies)
            new_entropies.append(new_entropy)
        return attributes[new_entropies.index(min(new_entropies))]

    def _partition_instances(self, instances, split_attribute):
        """
        Performs a group by on attribute domain
        Parameters:
        -----------
        instances: list(list(obj))
        split_attribute: obj
        Returns:
        --------
        dict(obj: list(list(obj)))
        """
        # this is a group by attribute domain
        partitions = dict()  # key (attribute value): value (subtable)
        att_index = self._header.index(split_attribute)  # e.g. level -> 0
        att_domain = self._attribute_domains[
            split_attribute
        ]  # e.g. ["Junior", "Mid", "Senior"]
        for att_value in att_domain:
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)
        return partitions

    def _tdidt(self, current_instances, available_attributes):
        """
        Run TDIDT algorithm to build up the decision tree
        Parameters:
        -----------
        current instances: list(list(obj))
        available_attributes: list(obj)
        Returns:
        --------
        nested list
        """
        attribute = self._select_attribute(available_attributes, current_instances)
        available_attributes.remove(attribute)
        # start to build the tree
        tree = ["Attribute", attribute]
        # group data by attribute domains (pairwise disjoint partitions)
        partitions = self._partition_instances(current_instances, attribute)

        for attribute_value, attribute_partition in partitions.items():
            # attribute_value e.g. Java, Python, R for the third attribute
            # attribute_partition: all instances with the given attribute_value
            value_subtree = ["Value", attribute_value]
            # CASE 1: If all class labels of the partition
            # are the same, then create lead node
            if len(attribute_partition) > 0 and utils.all_same_class(
                attribute_partition
            ):
                class_label = attribute_partition[0][-1]
                leaf_node = [
                    "Leaf",
                    class_label,
                    len(attribute_partition),
                    len(current_instances),
                ]
                value_subtree.append(leaf_node)
            # CASE 2: If there are no more attributes to select
            # (clash), then handle the clash with majority vote
            elif len(attribute_partition) > 0 and not available_attributes:
                # We have a mix of class labels, handle clash with majority vote leaf node
                class_label = utils.get_majority_label(attribute_partition)
                leaf_node = [
                    "Leaf",
                    class_label,
                    utils.get_class_label_occurrences(class_label, attribute_partition),
                    len(current_instances),
                ]
                value_subtree.append(leaf_node)
            # CASE 3: If there are no more instances to partition (empty partition),
            # then backtrack and replace attribute node with majority vote leaf node
            elif not attribute_partition:
                # "backtrack" and replace this attribute node with a majority vote leaf node.
                # we "change our mind" about the attribute being selected and return a leaf node instead
                return [
                    "Leaf",
                    utils.get_majority_label(current_instances),
                    len(current_instances),
                    len(current_instances),
                ]
            else:
                # append subtree to value_subtree and tree appropriately
                subtree = self._tdidt(attribute_partition, available_attributes.copy())
                value_subtree.append(subtree)
            tree.append(value_subtree)
        return tree

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        self._class_labels = set(self.y_train)
        train = [self.X_train[i] + [self.y_train[i]] for i in range(len(self.X_train))]
        if not self._header:
            self._header = [number for number in range(len(self.X_train[0]))]
        self._create_attribute_domains()
        self.tree = self._tdidt(train, copy.deepcopy(self._header))

    def _dfs_predict(self, instance, tree):
        """
        Helper function for predict. Recursively
        traverses through a tree until a class
        label is found for a given instance.
        Parameters:
        -----------
        instance: list(obj)
        tree: nested list
        Returns:
        --------
        (obj) the predicted class label for the given instance
        """
        if tree[0] == "Leaf":
            return tree[1]
        attribute = tree[1]
        value = instance[self._header.index(attribute)]
        for subtree in tree[2:]:
            if subtree[1] == value:
                return self._dfs_predict(instance, subtree[2])

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [self._dfs_predict(test_instance, self.tree) for test_instance in X_test]

    def _dfs_rules(self, attribute_names, class_name, tree, antecedent):
        """
        """
        if tree[0] == "Leaf":
            print(f"IF {' AND '.join(antecedent)} THEN {class_name} = {tree[1]}")
            return
        attribute_name = tree[1]
        for subtree in tree[2:]:
            attribute_value = subtree[1]
            new_antecedent = antecedent[:] + [
                str(attribute_name) + " == " + str(attribute_value)
            ]
            self._dfs_rules(attribute_names, class_name, subtree[2], new_antecedent)

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.
        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        self._dfs_rules(
            attribute_names if attribute_names else self._header,
            class_name,
            self.tree,
            [],
        )


class MyRandomForestClassifier:
    """
    Represents a Random Forest Classifier

    Attributes:
    -----------
    weak_learners list(MyDecisionTreeClassifier): A list of
        weak classifiers of length N, i.e. an ensemble
    best_learners list(MyDecisionTreeClassifier): A list of the top
        M best classifiers. Parallel to self.best_learners_selected_attributes
    best_learners_selected_attributes: The selected attributes for each
        of the M best learners. Parallel to self.best_learners
    num_attributes_to_select int: The number of attributes that each
        decision tree will randomly select to be trained on
    M int: The number of "better" learners that will be used for prediction

    """

    def __init__(self, N, M, F):
        """
        Initializer for MyRandomForestClassifier

        Parameters:
        -----------
        N (int): the number of "weak" learners that work together
        M (int): the number of "better" learners from the N "weak" learners. M < N
        F (int): the number of randonly selected attributes that each decision tree will be trained on
        """
        self.weak_learners = []
        self.N = N
        self.best_learners = []
        self.best_learners_selected_attributes = []
        self.num_attributes_to_select = F
        self.M = M
        self.estimated_ensemble_accuracy = 0.0

    def _compute_estimated_ensemble_accuracy(self, X_test, y_test):
        """
        Uses the ensemble to classify each of the instances in the test set selected
        and uses the result as an estimate of the performance of the ensemble
        on (genuinely) unseen data. Sets the estimated_ensemble_accuracy instance
        attribute with the result.

        Parameters:
        -----------
        X_test: list(list(obj))
        y_test: list(obj)
        """
        for learner, selected_attributes in zip(
            self.best_learners, self.best_learners_selected_attributes
        ):
            X_test_selected_attributes = [
                [instance[index] for index in selected_attributes] for instance in X_test
            ]
            y_predicted = learner.predict(X_test_selected_attributes)
            self.estimated_ensemble_accuracy += myevaluation.accuracy_score(
                y_test, y_predicted
            )
        self.estimated_ensemble_accuracy /= self.M

    def _select_attributes(self, num_attributes):
        """
        When building the decision trees, we randomly select "F" attributes
        as candidates to partition on. This method randomly selects those
        attributes for a decision tree.

        Parameters:
        -----------
        num_attributes: int

        Returns:
        --------
        list(int)
        """
        selected_attributes = []
        while len(selected_attributes) < self.num_attributes_to_select:
            attribute = np.random.randint(0, num_attributes)
            if attribute not in selected_attributes:
                selected_attributes.append(attribute)
        return selected_attributes

    def _select_best_learners(self, learners_information):
        """
        Selects the M most accurate of the N decision trees using the
        corresponding validation sets. The learners_information
        is a list of triple tuples in the form of
        (index of the learner in self.weak_learners, the learner's accuracy, the learner's selected attributes)

        Parameters:
        -----------
        learners_information: tuple(int, float, list(int))
        """
        learners_information.sort(key=lambda x: x[1], reverse=True)
        self.best_learners = [
            self.weak_learners[learner_index] for learner_index, _, _ in learners_information[: self.M]
        ]
        self.best_learners_selected_attributes = [
            selected_attributes for _, _, selected_attributes in learners_information[: self.M]
        ]

    def fit(self, X, y):
        """
        Builds a forest of trees from the training set

        Parameters:
        -----------
        X_train: list(list(obj))
        y_train: list(obj)
        """
        # divide the available data into a test set and the remainder set. The test set
        # is 1/3 of available data and the remainder set is 2/3 of the available data
        X_remainder, X_test, y_remainder, y_test = utils.kfold_test_train_split(
            X, y, n_splits=3, random_state=0, shuffle=True, stratified=True
        )
        # weak_learner_accuracies is a list of triple tuples where each tuple is in the form
        # (index of the learner in self.weak_learners, the learner's accuracy, the learner's selected attributes)
        weak_learners_information = []
        for learner_index in range(self.N):
            # divide the remainder set into training and validation data using bootstrapping
            X_train, X_validation, y_train, y_validation = myevaluation.bootstrap_sample(
                X_remainder, y_remainder, random_state=np.random.randint(0, 2 ** 32 - 1)
            )
            # randomly select 'F' attributes as candidates to partition on
            selected_attributes = self._select_attributes(len(X_train[0]))
            X_train = [
                [instance[index] for index in selected_attributes] for instance in X_train
            ]
            X_validation = [
                [instance[index] for index in selected_attributes] for instance in X_validation
            ]
            # fit the decision tree and record its accuracy against the validation data
            self.weak_learners.append(MyDecisionTreeClassifier(header=selected_attributes))
            self.weak_learners[learner_index].fit(X_train, y_train)
            y_predicted = self.weak_learners[learner_index].predict(X_validation)
            weak_learners_information.append(
                (
                    learner_index,
                    myevaluation.accuracy_score(y_validation, y_predicted),
                    selected_attributes,
                )
            )
        # select the M most accurate of the N decision trees
        self._select_best_learners(weak_learners_information)
        # estimate the performance of the ensemble on (genuinely) unseen data
        self._compute_estimated_ensemble_accuracy(X_test, y_test)

    def predict(self, X_test):
        """
        Predict the class for each instance in the test data

        Parameters:
        -----------
        X_test: list(list(obj))

        Returns:
        --------
        list(obj)
        """
        y_predicted = []
        for test_instance in X_test:
            class_predictions = []
            for learner_index, learner in enumerate(self.best_learners):
                only_selected_attributes_instance = [test_instance[i] for i in self.best_learners_selected_attributes[learner_index]]
                prediction = learner.predict([only_selected_attributes_instance])[0]
                class_predictions.append(prediction)
            y_predicted.append(utils.get_mode(class_predictions))
        return y_predicted
