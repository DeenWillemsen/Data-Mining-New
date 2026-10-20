import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def gradient_boosting(X_train_bow, X_test_bow, y_train, y_test):

    gb = GradientBoostingClassifier(
        loss='log_loss',
        learning_rate=0.1,
        n_estimators=900,
        subsample=0.9,
        min_samples_split=20,
        min_samples_leaf=2,
        min_weight_fraction_leaf=0,
        max_depth=4,
        min_impurity_decrease=0,
        max_features=4,
        verbose=0,
        max_leaf_nodes=None,
        validation_fraction=0.0001,
        n_iter_no_change=None,
        tol=0.0001,
        ccp_alpha=0
    )

    gb = gb.fit(X_train_bow, y_train)
    gb_pred = gb.predict(X_test_bow)

    
    accuracy = accuracy_score(y_test, gb_pred)
    precision = precision_score(y_test, gb_pred)
    recall = recall_score(y_test, gb_pred)
    f1 = f1_score(y_test, gb_pred)

    print("\nGradient Boosting Results")
    print("--------------------------------")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1)

    return gb, gb_pred
