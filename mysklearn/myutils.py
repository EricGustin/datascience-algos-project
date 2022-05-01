def grouping(attributes):
    count = {}
    grouping = []
    names = []
    total = 0
    for val in attributes:
        total +=1
        if val not in count:
            count[val] = 1
        else:
            count[val] +=1
    for k, v in count.items():
        val = round(v/total, 2)
        grouping.append(val)
        names.append(k)

    return grouping, names


import math

from mysklearn import mypytable, myevaluation
import numpy as np
from collections import Counter


def get_table_with_file_data(filename):
    """
    This function takes in the name of a file and returns a MyPyTable
    object containing the file's data
    Parameters:
    -----------
    filename (str): the name of a file
    Returns:
    --------
    MyPyTable
    """
    table = mypytable.MyPyTable()
    table.load_from_file(filename)
    return table


def my_discretizer(value):
    """
    Discretizes values into "high" and "low" labels
    Parameters:
    -----------
    value: int
    Returns:
    --------
    str
    """
    return "high" if value >= 100 else "low"


def compute_euclidean_distance(train_instance, test_instance):
    """
    Computes the euclidean distance between two n-dimensional vectors
    defined as the square root of the summation of (a_i - b_i)**2 from
    i=1 to i=len(train_instance) where a is train_instance and b is
    test_instance. train_instance and test_instance are parallel lists.
    Parameters:
    -----------
    train_instance: list(numeric)
    test_instance: list(numeric)
    Returns:
    --------
    numeric
    """
    distance = 0
    for train_value, test_value in zip(train_instance, test_instance):
        if isinstance(train_value, str):
            distance += 1 * (train_value != test_value)
        else:
            distance += (train_value - test_value) ** 2
    distance = distance ** 0.5
    return distance


def randomize_in_place(a_list, parallel_list=None):
    """
    Shuffle in-place. If there are two lists passed as
    parameters, then the parallel order will be preserved
    Parameters:
    -----------
    a_list: list(obj)
    parallel_list (default None otherwise list(obj))
    """
    for i in range(len(a_list)):
        rand_index = np.random.randint(0, len(a_list))
        a_list[i], a_list[rand_index] = a_list[rand_index], a_list[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = (
                parallel_list[rand_index],
                parallel_list[i],
            )


def divide_indices_into_k_folds(indices, n_samples, n_splits):
    """
    This function takes in a list of indices and returns a 2D list
    which represents the k folds.
    Parameters:
    -----------
    indices: list(int)
    n_samples: int
    n_splits: int
    Returns:
    --------
    list(list(int))
    """
    k_folds = []
    start_index = 0
    for fold_num in range(n_splits):
        fold_size = (
            n_samples // n_splits + 1
            if fold_num < (n_samples % n_splits)
            else n_samples // n_splits
        )
        fold = [indices[index] for index in range(start_index, start_index + fold_size)]
        k_folds.append(fold)
        start_index += fold_size
    return k_folds


def confusion_matrix_cell(y_true, y_pred, true_label, pred_label):
    """
    This function determines the value of a given cell in a confusion
    matrix. The cell that is being computed is located at the true_label
    row and the pred_label column. For example, if true_label=0 and
    pred_label=1, then this function computes the number of times a
    value was incorrectly labelled as 1 when it was supposed to be labelled
    as 0.
    Parameters:
    -----------
    y_true: list(obj)
    y_pred: list(obj)
    true_label: obj
    pred_label: obj
    Returns:
    --------
    int
    """
    cell = 0
    for true_class, pred_class in zip(y_true, y_pred):
        if (true_class, pred_class) == (true_label, pred_label):
            cell += 1
    return cell


def group_by_indices(class_labels):
    """
    This function takes in a list of class labels and groups
    then by class label and then returns their original indices
    in grouped by order.
    Parameters:
    -----------
    class_labels: list(obj)
    Returns:
    --------
    list(int)
    """
    label_to_indices = dict()
    for index, class_label in enumerate(class_labels):
        label_to_indices[class_label] = label_to_indices.get(class_label, []) + [index]
    indices = []
    for label_indices in label_to_indices.values():
        indices += [index for index in label_indices]
    return indices


def get_top_k_distances_and_indices(distances, k):
    """
    Given a list of distances and a number k, this function finds
    and returns the k smallest distances and their original indices
    in the provided distances list.
    Parameters:
    -----------
    distances: list(numeric)
    k: int
    Returns:
    --------
    list(numeric), list(int)
    """
    indices = range(len(distances))
    # create a list of (distance, index) tuples sorted by their distance
    sorted_tuples = sorted(zip(distances, indices))[:k]
    # separate distances from indices
    unzipped_sorted_tuples = list(zip(*sorted_tuples))
    top_k_distances = list(unzipped_sorted_tuples[0])
    top_k_indices = list(unzipped_sorted_tuples[1])
    return top_k_distances, top_k_indices


def kfold_test_train_split(
    x_data, y_data, n_splits, random_state, shuffle, stratified=False
):
    """
    This function produces the training and testing data for a dataset by
    using either k fold cross validation or stratified k fold cross validation.
    The returned values represent X_train, X_test, y_train, and y_test,
    respectfully.
    Parameters:
    -----------
    x_data: list(list(obj))
    y_data: list(obj)
    n_splits: int
    random_state: int
    shuffle: bool
    stratified: bool
    Returns:
    --------
    list(list(obj)), list(list(obj)), list(obj), list(obj)
    """
    X_train, X_test, y_train, y_test = [], [], [], []
    X_train_folds, X_test_folds = None, None

    if stratified:
        X_train_folds, X_test_folds = myevaluation.stratified_kfold_cross_validation(
            x_data, y_data, n_splits, random_state, shuffle
        )
    else:
        X_train_folds, X_test_folds = myevaluation.kfold_cross_validation(
            x_data, n_splits, random_state, shuffle
        )

    for train_indices, test_indices in zip(X_train_folds, X_test_folds):
        X_train += [x_data[train_index] for train_index in train_indices]
        X_test += [x_data[test_index] for test_index in test_indices]
        y_train += [y_data[train_index] for train_index in train_indices]
        y_test += [y_data[test_index] for test_index in test_indices]
    return X_train, X_test, y_train, y_test


def enhance_confusion_matrix(confusion_matrix, labels_column):
    """
    This function is used to 'enhance' a given confusion matrix.
    The enhancements include adding a column for labels, a column
    for totals, and a column for recognition percentage to create
    a new enhanced confusion matrix.
    Parameters:
    -----------
    confusion_matrix: list(list(obj))
    labels_column: list(obj)
    Returns:
    --------
    list(list(obj))
    """
    enhanced_confusion_matrix = []
    total_column = [sum(row) for row in confusion_matrix]
    # recognition_column = [confusion_matrix[index][index] / total_column[index] * 100 for index in range(len(confusion_matrix))]
    recognition_column = []
    for index in range(len(confusion_matrix)):
        if total_column[index] == 0:
            recognition_column.append(0)
        else:
            recognition_column.append(
                confusion_matrix[index][index] / total_column[index] * 100
            )
    for row_num, original_row in enumerate(confusion_matrix):
        enhanced_confusion_matrix.append(
            [labels_column[row_num]]
            + original_row
            + [total_column[row_num], recognition_column[row_num]]
        )
    return enhanced_confusion_matrix


"""Helpers for MyRandomForestClassifier"""


def get_majority_label(instances):
    """
    Helper function for tdidt case 2 and 3.
    Given a list of instances, determine the
    most frequently occurring class label. The class
    label is the last column in each row. If there is
    a tie for the majority label, then return the label
    that comes first alphabetically/numerically
    Parameters:
    -----------
    instances: list(list(obj))
    Returns:
    --------
    obj
    """
    classes = sorted([instance[-1] for instance in instances])
    return Counter(classes).most_common(1)[0][0]


def get_class_label_occurrences(class_label, instances):
    """
    Helper function for creating the third element in a leaf node
    for the tdidt algorithm.
    Finds the number of times the given class label occurs in a
    list of instances.
    Parameters:
    -----------
    class_label: ojb
    instances: list(list(obj))
    Returns:
    --------
    int
    """
    return sum([1 * (class_label == instance[-1]) for instance in instances])


def all_same_class(attribute_partition):
    """
    Given a partition, determines whether all class
    values are the same for each instance in the
    partition. Zips the partition, then casts to a
    list, then takes the last element in the list
    (which are the class labels) and then checks
    if all class labels are the same by casting
    to a set and checking if its length is 1.
    Parameters:
    -----------
    attribute_partition: list(list(obj))
    Returns:
    --------
    bool
    """
    return len(set(list(zip(*attribute_partition))[-1])) == 1


def compute_new_entropy(likelihoods, attribute_entropies):
    """
    Computes E_new for an arbitrary attribute.
    likelihoods and attribute_entropies are
    parallel lists
    Parameters:
    -----------
    likelihoods: list(float)
    attribute_entropies: list(float)
    Returns:
    --------
    float
    """
    return sum(
        [
            likelihood * entropy
            for likelihood, entropy in zip(likelihoods, attribute_entropies)
        ]
    )


def compute_likelihoods(attribute, attribute_domain, instances, header):
    """
    Computes the likelihood for each value for the given attribute.
    A likelihood can be described as a ratio between the number
    of instances that have an attribute's value divided by the
    total number of instances
    """
    numerators = []
    attribute_index = header.index(attribute)
    for value in attribute_domain:
        numerator = 0
        for instance in instances:
            if instance[attribute_index] == value:
                numerator += 1
        numerators.append(numerator)
    likelihoods = [numerator / len(instances) for numerator in numerators]
    return likelihoods


def compute_entropy(proportions):
    """
    Helper function for self._select_attribute.
    Computes Entropy, given a list of proportions.
    Parameters:
    -----------
    list(float)
    Returns:
    --------
    float
    """
    entropy = 0
    for proportion in proportions:
        if proportion == 0:
            continue
        entropy += -1 * proportion * math.log(proportion, 2)
    return entropy


def compute_proportions(attribute, value, instances, header, class_labels):
    """
    Helper function for self._select_attributes.
    Computes the proportions (p_i) for a given
    attribute's value. The returned list represents
    the attribute/value pair's proportions for each
    class label.
    Parameters:
    -----------
    attribute: obj
    value: obj
    instances: list(list(obj))
    Returns:
    --------
    list(float)
    """
    proportions = []
    denominator = 0
    attribute_index = header.index(attribute)
    for label in class_labels:
        numerator = 0
        for instance in instances:
            if instance[attribute_index] == value and instance[-1] == label:
                numerator += 1
                denominator += 1
        proportions.append(numerator)
    if denominator == 0:
        return [0.0] * len(class_labels)
    proportions = [num / denominator for num in proportions]
    return proportions