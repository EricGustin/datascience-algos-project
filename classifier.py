import os
from mysklearn.mypytable import MyPyTable
import importlib

import mysklearn.myutils
importlib.reload(mysklearn.myutils)
import mysklearn.myutils as myutils

from mysklearn.myclassifiers import MyNaiveBayesClassifier
import mysklearn.myevaluation as myevaluation
from tabulate import tabulate


filename = os.path.join("data", "cleaned-recidivism-data-NA.csv")
table = myutils.get_table_with_file_data(filename)

x_table_data = [MyPyTable.get_partial_row(row, exclude_indices=[10]) for row in table.data]
y_table_data = table.get_column("Return to Prison")

X_table = MyPyTable(column_names=table.column_names[:-1], data=x_table_data)
y_table = MyPyTable(column_names=table.column_names[-1:], data=y_table_data)

k = 10
labels = ["Yes", "No"]
positive_label = "Yes"
confusion_matrix_header = ["Return To Prison"] + labels + ["Total", "Recognition (%)"]

X_train_1, X_test_1, y_train_1, y_test_1 = myutils.kfold_test_train_split(X_table.data, y_table.data, k, 0, False, stratified=True)

classifiers = [
    MyNaiveBayesClassifier()
]
for classifier in classifiers:
    classifier.fit(X_train_1, y_train_1)
    y_predicted = classifier.predict(X_test_1)
    accuracy = myevaluation.accuracy_score(y_test_1, y_predicted)
    error_rate = 1 - accuracy
    precision = myevaluation.binary_precision_score(y_test_1, y_predicted, labels, positive_label)
    recall = myevaluation.binary_recall_score(y_test_1, y_predicted, labels, positive_label)
    f1_measure = myevaluation.binary_f1_score(y_test_1, y_predicted, labels, positive_label)
    confusion_matrix = myevaluation.confusion_matrix(y_test_1, y_predicted, labels)
    enhanced_confusion_matrix = myutils.enhance_confusion_matrix(confusion_matrix, labels)
    print(f"Evaluation metrics for {classifier.__class__.__name__}:")
    print(f"Accuracy: {accuracy}")
    print(f"Error Rate: {error_rate}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 measure: {f1_measure}")
    print(tabulate(enhanced_confusion_matrix, headers=confusion_matrix_header) + "\n\n")