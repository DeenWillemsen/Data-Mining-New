print("MAIN SCRIPT STARTED")

from preprocess_data import preprocess_data
from multinomial_naive_bayes import multinomial_naive_bayes
from random_forests import random_forest
from gradient_boosting import gradient_boosting

"""
To test whether the difference in accuracy between two classifiers
is significant, use McNemar’s test [2], with significance level α = 0.01
"""

print("Data preprocessing started")
# Preprocess the data
X_train_bow, X_test_bow, y_train, y_test = preprocess_data()

print("MNNB started")
# Run Multinomial Naive Bayes
multinomial_naive_bayes(
    X_train_bow,
    X_test_bow,
    y_train,
    y_test
)


print("\n -- Random Forest -- ")
random_forest(
    X_train_bow,
    X_test_bow,
    y_train,
    y_test
)

print("\n -- Gradient Boosting -- ")
gradient_boosting(
    X_train_bow,
    X_test_bow,
    y_train,
    y_test
)