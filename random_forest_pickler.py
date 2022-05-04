import os
import pickle
import numpy as np
import mysklearn.myutils as myutils
from mysklearn.mypytable import MyPyTable
from mysklearn.myclassifiers import MyRandomForestClassifier


def pickle_classifier(classifier):
    """
    Pickles an MyRandomForestClassifier object
    """
    outfile = open("trained_random_forest.pkl", "wb")
    pickle.dump(classifier, outfile)
    outfile.close()


def get_training_sets(table):
    """
    Creates X and y training sets from a MyPyTable
    """
    X_train = [row[:-1] for row in table.data]
    y_train = table.get_column("Return to Prison")
    return X_train, y_train


def get_table():
    """
    Puts a file's data into a MyPyTable object and returns it
    """
    filename = os.path.join("data", "cleaned-recidivism-data-NA.csv")
    table = myutils.get_table_with_file_data(filename)
    return table


def main():
    np.random.seed(0)

    table = get_table()
    X_train, y_train = get_training_sets(table)

    N = 20
    M = 8
    F = 4
    random_forest_classifier = MyRandomForestClassifier(N, M, F)
    random_forest_classifier.fit(X_train, y_train)

    pickle_classifier(random_forest_classifier)
    print(f"Successfully pickled a random forest classifier with an estimated accuracy of {random_forest_classifier.estimated_ensemble_accuracy * 100}%")
    print("The random forest's decision trees:")
    for i, learner in enumerate(random_forest_classifier.best_learners):
        print(f"\nTree {i}:")
        print(learner.tree)
    print("\nThe decision rules for each tree in the forest:")
    for i, learner in enumerate(random_forest_classifier.best_learners):
        print(f"\nDecision rules for tree {i}:")
        learner.print_decision_rules()



if __name__ == "__main__":
    main()