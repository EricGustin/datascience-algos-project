import numpy as np

import mysklearn.myutils as utils

from mysklearn import myevaluation
from mysklearn.myclassifiers import (
    MyNaiveBayesClassifier,
    MyDecisionTreeClassifier,
    MyRandomForestClassifier
)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score


"""
**************************************
Tests for Naive Bayes Classifier
**************************************
"""
# in-class Naive Bayes example (lab task #1)
inclass_example_col_names = ["att1", "att2"]
X_train_inclass_example = [
    [1, 5],  # yes
    [2, 6],  # yes
    [1, 5],  # no
    [1, 5],  # no
    [1, 6],  # yes
    [2, 6],  # no
    [1, 5],  # yes
    [1, 6],  # yes
]
y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

# RQ5 (fake) iPhone purchases dataset
iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
iphone_table = [
    [1, 3, "fair", "no"],
    [1, 3, "excellent", "no"],
    [2, 3, "fair", "yes"],
    [2, 2, "fair", "yes"],
    [2, 1, "fair", "yes"],
    [2, 1, "excellent", "no"],
    [2, 1, "excellent", "yes"],
    [1, 2, "fair", "no"],
    [1, 1, "fair", "yes"],
    [2, 2, "fair", "yes"],
    [1, 2, "excellent", "yes"],
    [2, 2, "excellent", "yes"],
    [2, 3, "fair", "yes"],
    [2, 2, "excellent", "no"],
    [2, 3, "fair", "yes"],
]

# Bramer 3.2 train dataset
train_col_names = ["day", "season", "wind", "rain", "class"]
train_table = [
    ["weekday", "spring", "none", "none", "on time"],
    ["weekday", "winter", "none", "slight", "on time"],
    ["weekday", "winter", "none", "slight", "on time"],
    ["weekday", "winter", "high", "heavy", "late"],
    ["saturday", "summer", "normal", "none", "on time"],
    ["weekday", "autumn", "normal", "none", "very late"],
    ["holiday", "summer", "high", "slight", "on time"],
    ["sunday", "summer", "normal", "none", "on time"],
    ["weekday", "winter", "high", "heavy", "very late"],
    ["weekday", "summer", "none", "slight", "on time"],
    ["saturday", "spring", "high", "heavy", "cancelled"],
    ["weekday", "summer", "high", "slight", "on time"],
    ["saturday", "winter", "normal", "none", "late"],
    ["weekday", "summer", "high", "none", "on time"],
    ["weekday", "winter", "normal", "heavy", "very late"],
    ["saturday", "autumn", "high", "slight", "on time"],
    ["weekday", "autumn", "none", "heavy", "on time"],
    ["holiday", "spring", "normal", "slight", "on time"],
    ["weekday", "spring", "normal", "none", "on time"],
    ["weekday", "spring", "normal", "slight", "on time"],
]


def check_priors_are_equal(actual, expected):
    """
    Checks for the equality of two priors dictionaries.

    Parameters:
    -----------
    actual: dict(str: float)
    expected: dict(str: float)
    """
    assert len(actual) == len(expected)
    for label, actual_prior in actual.items():
        assert label in expected
        assert np.isclose(actual_prior, expected[label])


def check_posteriors_are_equal(actual, expected):
    """
    Helper function for checking whether two nested dictionaries
    are equal. Since the posteriors are floats, I decided to manually
    write the code for this check since we should be using np.isclose
    rather than '=='

    Parameter:
    ----------
    actual: dict(str: dict(obj: dict(obj: float)))
    expected: dict(str: dict(obj: dict(obj: float)))
    """
    assert len(actual) == len(expected)
    for attribute in actual.keys():
        assert attribute in expected
        assert len(actual[attribute]) == len(expected[attribute])
        for value in actual[attribute].keys():
            assert value in expected[attribute]
            assert len(actual[attribute][value]) == len(expected[attribute][value])
            for label in actual[attribute][value].keys():
                assert label in expected[attribute][value]
                assert np.isclose(
                    actual[attribute][value][label], expected[attribute][value][label]
                )


def test_naive_bayes_classifier_fit_1():
    expected_priors = {"no": 3 / 8, "yes": 5 / 8}
    expected_posteriors = {
        "att1": {1: {"yes": 4 / 5, "no": 2 / 3}, 2: {"yes": 1 / 5, "no": 1 / 3},},
        "att2": {5: {"yes": 2 / 5, "no": 2 / 3}, 6: {"yes": 3 / 5, "no": 1 / 3},},
    }

    naive_bayes_classifier = MyNaiveBayesClassifier()
    naive_bayes_classifier.fit(X_train_inclass_example, y_train_inclass_example)
    check_priors_are_equal(naive_bayes_classifier.priors, expected_priors)
    check_posteriors_are_equal(naive_bayes_classifier.posteriors, expected_posteriors)


def test_naive_bayes_classifier_fit_2():
    X_train = [row[:-1] for row in iphone_table]
    y_train = [row[-1] for row in iphone_table]

    expected_priors = {"no": 5 / 15, "yes": 10 / 15}
    expected_posteriors = {
        "att1": {1: {"yes": 2 / 10, "no": 3 / 5}, 2: {"yes": 8 / 10, "no": 2 / 5}},
        "att2": {
            1: {"yes": 3 / 10, "no": 1 / 5},
            2: {"yes": 4 / 10, "no": 2 / 5},
            3: {"yes": 3 / 10, "no": 2 / 5},
        },
        "att3": {
            "fair": {"yes": 7 / 10, "no": 2 / 5},
            "excellent": {"yes": 3 / 10, "no": 3 / 5},
        },
    }

    naive_bayes_classifier = MyNaiveBayesClassifier()
    naive_bayes_classifier.fit(X_train, y_train)
    check_priors_are_equal(naive_bayes_classifier.priors, expected_priors)
    check_posteriors_are_equal(naive_bayes_classifier.posteriors, expected_posteriors)


def test_naive_bayes_classifier_fit_3():
    X_train = [row[:-1] for row in train_table]
    y_train = [row[-1] for row in train_table]

    expected_priors = {
        "on time": 14 / 20,
        "late": 2 / 20,
        "very late": 3 / 20,
        "cancelled": 1 / 20,
    }
    expected_posteriors = {
        "att1": {
            "weekday": {
                "on time": 9 / 14,
                "late": 1 / 2,
                "very late": 3 / 3,
                "cancelled": 0 / 1,
            },
            "saturday": {
                "on time": 2 / 14,
                "late": 1 / 2,
                "very late": 0 / 3,
                "cancelled": 1 / 1,
            },
            "sunday": {
                "on time": 1 / 14,
                "late": 0 / 2,
                "very late": 0 / 3,
                "cancelled": 0 / 1,
            },
            "holiday": {
                "on time": 2 / 14,
                "late": 0 / 2,
                "very late": 0 / 3,
                "cancelled": 0 / 1,
            },
        },
        "att2": {
            "spring": {
                "on time": 4 / 14,
                "late": 0 / 2,
                "very late": 0 / 3,
                "cancelled": 1 / 1,
            },
            "summer": {
                "on time": 6 / 14,
                "late": 0 / 2,
                "very late": 0 / 3,
                "cancelled": 0 / 1,
            },
            "autumn": {
                "on time": 2 / 14,
                "late": 0 / 2,
                "very late": 1 / 3,
                "cancelled": 0 / 1,
            },
            "winter": {
                "on time": 2 / 14,
                "late": 2 / 2,
                "very late": 2 / 3,
                "cancelled": 0 / 1,
            },
        },
        "att3": {
            "none": {
                "on time": 5 / 14,
                "late": 0 / 2,
                "very late": 0 / 3,
                "cancelled": 0 / 1,
            },
            "high": {
                "on time": 4 / 14,
                "late": 1 / 2,
                "very late": 1 / 3,
                "cancelled": 1 / 1,
            },
            "normal": {
                "on time": 5 / 14,
                "late": 1 / 2,
                "very late": 2 / 3,
                "cancelled": 0 / 1,
            },
        },
        "att4": {
            "none": {
                "on time": 5 / 14,
                "late": 1 / 2,
                "very late": 1 / 3,
                "cancelled": 0 / 1,
            },
            "slight": {
                "on time": 8 / 14,
                "late": 0 / 2,
                "very late": 0 / 3,
                "cancelled": 0 / 1,
            },
            "heavy": {
                "on time": 1 / 14,
                "late": 1 / 2,
                "very late": 2 / 3,
                "cancelled": 1 / 1,
            },
        },
    }

    naive_bayes_classifier = MyNaiveBayesClassifier()
    naive_bayes_classifier.fit(X_train, y_train)
    check_priors_are_equal(naive_bayes_classifier.priors, expected_priors)
    check_posteriors_are_equal(naive_bayes_classifier.posteriors, expected_posteriors)


def test_naive_bayes_classifier_predict_1():
    X_test = [[1, 5]]
    y_expected = ["yes"]
    naive_bayes_classifier = MyNaiveBayesClassifier()
    naive_bayes_classifier.fit(X_train_inclass_example, y_train_inclass_example)
    y_predicted = naive_bayes_classifier.predict(X_test)
    assert y_predicted == y_expected


def test_naive_bayes_classifier_predict_2():
    X_train = [row[:-1] for row in iphone_table]
    y_train = [row[-1] for row in iphone_table]
    X_test = [[2, 2, "fair"], [1, 1, "excellent"]]
    y_expected = ["yes", "no"]
    naive_bayes_classifier = MyNaiveBayesClassifier()
    naive_bayes_classifier.fit(X_train, y_train)
    y_predicted = naive_bayes_classifier.predict(X_test)
    assert y_predicted == y_expected


def test_naive_bayes_classifier_predict_3():
    X_train = [row[:-1] for row in train_table]
    y_train = [row[-1] for row in train_table]
    X_test = [
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "high", "heavy"],
        ["sunday", "summer", "normal", "slight"],
    ]
    y_expected = ["very late", "on time", "on time"]
    naive_bayes_classifier = MyNaiveBayesClassifier()
    naive_bayes_classifier.fit(X_train, y_train)
    y_predicted = naive_bayes_classifier.predict(X_test)
    assert y_predicted == y_expected


"""
**************************************
Tests for Decision Tree Classifier
**************************************
"""
interview_header = ["level", "lang", "tweets", "phd"]
interview_X_train = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"],
]
interview_y_train = [
    "False",
    "False",
    "True",
    "True",
    "True",
    "False",
    "True",
    "False",
    "True",
    "True",
    "True",
    "True",
    "True",
    "False",
]
interview_expected_tree = [
    "Attribute",
    "level",
    [
        "Value",
        "Junior",
        [
            "Attribute",
            "phd",
            ["Value", "no", ["Leaf", "True", 3, 5]],
            ["Value", "yes", ["Leaf", "False", 2, 5]],
        ],
    ],
    ["Value", "Mid", ["Leaf", "True", 4, 14]],
    [
        "Value",
        "Senior",
        [
            "Attribute",
            "tweets",
            ["Value", "no", ["Leaf", "False", 3, 5]],
            ["Value", "yes", ["Leaf", "True", 2, 5]],
        ],
    ],
]

interview_X_test = [
    ["Junior", "Java", "yes", "no"],
    ["Junior", "Java", "yes", "yes"],
]
interview_y_test = ["True", "False"]

header_2 = ["SoftEng", "ARIN", "HCI", "CSA", "Project"]
X_train_2 = [
    ["A", "B", "A", "B", "B"],
    ["A", "B", "B", "B", "A"],
    ["A", "A", "A", "B", "B"],
    ["B", "A", "A", "B", "B"],
    ["A", "A", "B", "B", "A"],
    ["B", "A", "A", "B", "B"],
    ["A", "B", "B", "B", "B"],
    ["A", "B", "B", "B", "B"],
    ["A", "A", "A", "A", "A"],
    ["B", "A", "A", "B", "B"],
    ["B", "A", "A", "B", "B"],
    ["A", "B", "B", "A", "B"],
    ["B", "B", "B", "B", "A"],
    ["A", "A", "B", "A", "B"],
    ["B", "B", "B", "B", "A"],
    ["A", "A", "B", "B", "B"],
    ["B", "B", "B", "B", "B"],
    ["A", "A", "B", "A", "A"],
    ["B", "B", "B", "A", "A"],
    ["B", "B", "A", "A", "B"],
    ["B", "B", "B", "B", "A"],
    ["B", "A", "B", "A", "B"],
    ["A", "B", "B", "B", "A"],
    ["A", "B", "A", "B", "B"],
    ["B", "A", "B", "B", "B"],
    ["A", "B", "B", "B", "B"],
]
y_train_2 = [
    "SECOND",
    "FIRST",
    "SECOND",
    "SECOND",
    "FIRST",
    "SECOND",
    "SECOND",
    "SECOND",
    "FIRST",
    "SECOND",
    "SECOND",
    "SECOND",
    "SECOND",
    "FIRST",
    "SECOND",
    "SECOND",
    "SECOND",
    "FIRST",
    "SECOND",
    "SECOND",
    "SECOND",
    "SECOND",
    "FIRST",
    "SECOND",
    "SECOND",
    "SECOND",
]
expected_tree_2 = [
    "Attribute",
    "SoftEng",
    [
        "Value",
        "A",
        [
            "Attribute",
            "Project",
            ["Value", "A", ["Leaf", "FIRST", 5, 14]],
            [
                "Value",
                "B",
                [
                    "Attribute",
                    "CSA",
                    [
                        "Value",
                        "A",
                        [
                            "Attribute",
                            "ARIN",
                            ["Value", "A", ["Leaf", "FIRST", 1, 2]],
                            ["Value", "B", ["Leaf", "SECOND", 1, 2]],
                        ],
                    ],
                    ["Value", "B", ["Leaf", "SECOND", 7, 9]],
                ],
            ],
        ],
    ],
    ["Value", "B", ["Leaf", "SECOND", 12, 26]],
]
X_test_2 = [
    ["B", "B", "B", "B", "B"],
    ["A", "A", "A", "A", "A"],
    ["A", "A", "A", "A", "B"],
]
y_test_2 = ["SECOND", "FIRST", "FIRST"]

header_3 = ["standing", "job_status", "credit_rating"]
X_train_3 = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
]
y_train_3 = [
    "no",
    "no",
    "yes",
    "yes",
    "yes",
    "no",
    "yes",
    "no",
    "yes",
    "yes",
    "yes",
    "yes",
    "yes",
    "no",
    "yes",
]
expected_tree_3 = [
    "Attribute",
    "standing",
    [
        "Value",
        1,
        [
            "Attribute",
            "job_status",
            ["Value", 1, ["Leaf", "yes", 1, 5]],
            [
                "Value",
                2,
                [
                    "Attribute",
                    "credit_rating",
                    ["Value", "excellent", ["Leaf", "yes", 1, 2]],
                    ["Value", "fair", ["Leaf", "no", 1, 2]],
                ],
            ],
            ["Value", 3, ["Leaf", "no", 2, 5]],
        ],
    ],
    [
        "Value",
        2,
        [
            "Attribute",
            "credit_rating",
            ["Value", "excellent", ["Leaf", "no", 4, 4]],
            ["Value", "fair", ["Leaf", "yes", 6, 10]],
        ],
    ],
]
X_test_3 = [
    [2, 2, "fair"],
    [1, 1, "excellent"],
]
y_test_3 = ["yes", "yes"]


# def test_decision_tree_classifier_fit_1():
#     """Uses interview dataset from class"""
#     decision_tree_classifier = MyDecisionTreeClassifier(interview_header)
#     decision_tree_classifier.fit(interview_X_train, interview_y_train)
#     assert decision_tree_classifier.X_train == interview_X_train
#     assert decision_tree_classifier.y_train == interview_y_train
#     assert decision_tree_classifier.tree == interview_expected_tree
# test_decision_tree_classifier_fit_1()

# def test_decision_tree_classifier_fit_2():
#     """Uses dataset from Bramer 4.1 Figure 4.3"""
#     decision_tree_classifier = MyDecisionTreeClassifier(header_2)
#     decision_tree_classifier.fit(X_train_2, y_train_2)
#     assert decision_tree_classifier.X_train == X_train_2
#     assert decision_tree_classifier.y_train == y_train_2
#     assert decision_tree_classifier.tree == expected_tree_2


# def test_decision_tree_classifier_fit_3():
#     """Uses dataset from RQ5"""
#     decision_tree_classifier = MyDecisionTreeClassifier(header_3)
#     decision_tree_classifier.fit(X_train_3, y_train_3)
#     assert decision_tree_classifier.X_train == X_train_3
#     assert decision_tree_classifier.y_train == y_train_3
#     assert decision_tree_classifier.tree == expected_tree_3


# def test_decision_tree_classifier_predict_1():
#     """Uses interview dataset from class"""
#     decision_tree_classifier = MyDecisionTreeClassifier(interview_header)
#     decision_tree_classifier.fit(interview_X_train, interview_y_train)
#     y_actual = decision_tree_classifier.predict(interview_X_test)
#     assert y_actual == interview_y_test


# def test_decision_tree_classifier_predict_2():
#     """Uses dataset from Bramer 4.1 Figure 4.3"""
#     decision_tree_classifier = MyDecisionTreeClassifier(header_2)
#     decision_tree_classifier.fit(X_train_2, y_train_2)
#     y_actual = decision_tree_classifier.predict(X_test_2)
#     assert y_actual == y_test_2


# def test_decision_tree_classifier_predict_3():
#     """Uses dataset from RQ5"""
#     decision_tree_classifier = MyDecisionTreeClassifier(header_3)
#     decision_tree_classifier.fit(X_train_3, y_train_3)
#     y_actual = decision_tree_classifier.predict(X_test_3)
#     assert y_actual == y_test_3

"""
**********************************
Tests for Random Forest Classifier
**********************************
"""
expected_trees_1 = [
    ['Attribute', 2,
        ['Value', 'no',
            ['Attribute', 0,
                ['Value', 'Junior',
                    ['Attribute', 3,
                        ['Value', 'no',
                            ['Leaf', 'True', 1, 3]
                        ],
                        ['Value', 'yes',
                            ['Leaf', 'False', 2, 3]
                        ]
                    ]
                ],
                ['Value', 'Mid',
                    ['Leaf', 'True', 2, 12]
                ],
                ['Value', 'Senior',
                    ['Leaf', 'False', 7, 12]
                ]
            ]
        ],
        ['Value', 'yes',
            ['Attribute', 3,
                ['Value', 'no',
                    ['Leaf', 'True', 10, 16]
                ],
                ['Value', 'yes',
                    ['Leaf', 'True', 6, 6]
                ]
            ]
        ]
    ],
    ['Attribute', 0,
        ['Value', 'Junior',
            ['Attribute', 3,
                ['Value', 'no',
                    ['Leaf', 'True', 3, 7]
                ],
                ['Value', 'yes',
                    ['Leaf', 'False', 4, 7]
                ]
            ]
        ],
        ['Value', 'Mid',
            ['Leaf', 'True', 12, 28]
        ],
        ['Value', 'Senior',
            ['Attribute', 1,
                ['Value', 'Java',
                    ['Leaf', 'False', 5, 9]
                ],
                ['Value', 'Python',
                    ['Attribute', 3,
                        ['Value', 'no',
                            ['Leaf', 'False', 1, 3]
                        ],
                        ['Value', 'yes',
                            ['Leaf', 'True', 2, 3]
                        ]
                    ]
                ],
                ['Value', 'R',
                    ['Leaf', 'True', 1, 9]
                ]
            ]
        ]
    ]
]

best_decision_trees_selected_attributes_1 = [[0, 2, 3], [1, 0, 3]]

expected_trees_2 = [
    ['Attribute', 4,
        ['Value', 'A',
            ['Attribute', 0,
                ['Value', 'A',
                    ['Leaf', 'FIRST', 10, 15]
                ],
                ['Value', 'B',
                    ['Leaf', 'SECOND', 5, 15]
                ]
            ]
        ],
        ['Value', 'B',
            ['Attribute', 3,
                ['Value', 'A',
                    ['Attribute', 0,
                        ['Value', 'A',
                            ['Leaf', 'FIRST', 4, 4]
                        ],
                        ['Value', 'B',
                            ['Leaf', 'SECOND', 4, 8]
                        ]
                    ]
                ],
                ['Value', 'B',
                    ['Leaf', 'SECOND', 29, 37]
                ]
            ]
        ]
    ],
    ['Attribute', 4,
        ['Value', 'A',
            ['Attribute', 0,
                ['Value', 'A',
                    ['Leaf', 'FIRST', 11, 20]
                ],
                ['Value', 'B',
                    ['Leaf', 'SECOND', 9, 20]
                ]
            ]
        ],
        ['Value', 'B',
            ['Attribute', 2,
                ['Value', 'A',
                    ['Leaf', 'SECOND', 17, 32]
                ],
                ['Value', 'B',
                    ['Attribute', 1,
                        ['Value', 'A',
                            ['Attribute', 0,
                                ['Value', 'A',
                                    ['Leaf', 'FIRST', 2, 5]
                                ],
                                ['Value', 'B',
                                    ['Leaf', 'SECOND', 1, 5]
                                ]
                            ]
                        ],
                        ['Value', 'B',
                            ['Leaf', 'SECOND', 10, 15]
                        ]
                    ]
                ]
            ]
        ]
    ]
]

best_decision_trees_selected_attributes_2 = [[3, 4, 0, 2], [2, 4, 0, 1]]

expected_trees_3 = [
    ['Attribute', 2,
        ['Value', 'excellent',
            ['Attribute', 0,
                ['Value', 1,
                    ['Leaf', 'yes', 5, 17]
                ],
                ['Value', 2,
                    ['Leaf', 'no', 5, 17]
                ]
            ]
        ],
        ['Value', 'fair',
            ['Attribute', 0,
                ['Value', 1,
                    ['Leaf', 'no', 2, 13]
                ],
                ['Value', 2,
                    ['Leaf', 'yes', 10, 13]
                ]
            ]
        ]
    ],
    ['Attribute', 1,
        ['Value', 1,
            ['Attribute', 0,
                ['Value', 1,
                    ['Leaf', 'yes', 4, 9]
                ],
                ['Value', 2,
                    ['Leaf', 'yes', 4, 9]
                ]
            ]
        ],
        ['Value', 2,
            ['Attribute', 0,
                ['Value', 1,
                    ['Leaf', 'no', 2, 11]
                ],
                ['Value', 2,
                    ['Leaf', 'yes', 5, 11]
                ]
            ]
        ],
        ['Value', 3,
            ['Attribute', 0,
                ['Value', 1,
                    ['Leaf', 'no', 5, 10]
                ],
                ['Value', 2,
                    ['Leaf', 'yes', 5, 10]
                ]
            ]
        ]
    ],
    ['Attribute', 2,
        ['Value', 'excellent',
            ['Attribute', 1,
                ['Value', 1,
                    ['Leaf', 'no', 4, 11]
                ],
                ['Value', 2,
                    ['Leaf', 'yes', 4, 11]
                ],
                ['Value', 3,
                    ['Leaf', 'no', 1, 11]
                ]
            ]
        ],
        ['Value', 'fair',
            ['Attribute', 1,
                ['Value', 1,
                    ['Leaf', 'yes', 5, 19]
                ],
                ['Value', 2,
                    ['Leaf', 'yes', 3, 19]
                ],
                ['Value', 3,
                    ['Leaf', 'yes', 7, 19]
                ]
            ]
        ]
    ],
]

best_decision_trees_selected_attributes_3 = [[0, 2], [0, 1], [1, 2]]

def test_random_forest_classifier_fit_1():
    np.random.seed(0)
    N = 5
    M = 2
    F = 3
    random_forest_classifier = MyRandomForestClassifier(N, M, F)
    random_forest_classifier.fit(interview_X_train, interview_y_train)
    for i, selected_attributes in enumerate(random_forest_classifier.best_learners_selected_attributes):
        assert selected_attributes == best_decision_trees_selected_attributes_1[i]
    for learner, expected_header in zip(random_forest_classifier.best_learners, best_decision_trees_selected_attributes_1):
        assert learner._header == expected_header
    for learner, expected_tree in zip(random_forest_classifier.best_learners, expected_trees_1):
        assert learner.tree == expected_tree

def test_random_forest_classifier_fit_2():
    np.random.seed(0)
    N = 6
    M = 2
    F = 4
    random_forest_classifier = MyRandomForestClassifier(N, M, F)
    random_forest_classifier.fit(X_train_2, y_train_2)
    for i, selected_attributes in enumerate(random_forest_classifier.best_learners_selected_attributes):
        assert selected_attributes == best_decision_trees_selected_attributes_2[i]
    for learner, expected_header in zip(random_forest_classifier.best_learners, best_decision_trees_selected_attributes_2):
        assert learner._header == expected_header
    for learner, expected_tree in zip(random_forest_classifier.best_learners, expected_trees_2):
        assert learner.tree == expected_tree

def test_random_forest_classifier_fit_3():
    np.random.seed(0)
    N = 10
    M = 3
    F = 2
    random_forest_classifier = MyRandomForestClassifier(N, M, F)
    random_forest_classifier.fit(X_train_3, y_train_3)
    for i, selected_attributes in enumerate(random_forest_classifier.best_learners_selected_attributes):
        assert selected_attributes == best_decision_trees_selected_attributes_3[i]
    for learner, expected_header in zip(random_forest_classifier.best_learners, best_decision_trees_selected_attributes_3):
        assert learner._header == expected_header
    for learner, expected_tree in zip(random_forest_classifier.best_learners, expected_trees_3):
        assert learner.tree == expected_tree

def test_random_forest_classifier_predict_1():
    np.random.seed(0)
    N = 5
    M = 2
    F = 3
    random_forest_classifier = MyRandomForestClassifier(N, M, F)
    random_forest_classifier.fit(interview_X_train, interview_y_train)
    y_predicted = random_forest_classifier.predict(interview_X_test)
    assert y_predicted == ["True", "True"]

def test_random_forest_classifier_predict_2():
    np.random.seed(0)
    N = 6
    M = 2
    F = 4
    random_forest_classifier = MyRandomForestClassifier(N, M, F)
    random_forest_classifier.fit(X_train_2, y_train_2)
    y_predicted = random_forest_classifier.predict(X_test_2)
    assert y_predicted == ["SECOND", "FIRST", "FIRST"]

def test_random_forest_classifier_predict_3():
    np.random.seed(0)
    N = 10
    M = 3
    F = 2
    random_forest_classifier = MyRandomForestClassifier(N, M, F)
    random_forest_classifier.fit(X_train_3, y_train_3)
    y_predicted = random_forest_classifier.predict(X_test_3)
    assert y_predicted == ["yes", "yes"]