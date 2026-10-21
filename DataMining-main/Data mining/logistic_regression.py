
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def logistic_regression(X_train_bow, X_test_bow, y_train, y_test, vectorizer):
    #candidate hyperperameters(C)
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]

    #use stratified folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n--- Logistic Regression: Lasso ---")
    print("Selecting C using 5-fold cross-validation...")

    #evaluate each possible C value based on training data
    cv_results = []

    for C in C_values:
        model = LogisticRegression(
            C=C,
            l1_ratio=1,
            solver="liblinear",
            max_iter=5000,
            random_state=42
        )

        scores = cross_val_score(
            model,
            X_train_bow,
            y_train,
            cv=cv,
            scoring="accuracy"
        )

        mean_accuracy = np.mean(scores)
        standard_error = np.std(scores, ddof=1) / np.sqrt(len(scores))

        cv_results.append({
            "C": C,
            "mean_accuracy": mean_accuracy,
            "standard_error": standard_error
        })

        print(
            f"C={C:<7} "
            f"Mean accuracy={mean_accuracy} "
            f"SE={standard_error}"
        )

    best_result = max(
        cv_results,
        key=lambda result: result["mean_accuracy"]
    )


    #from all models within one SE of the mean accuracy, choose the smallest C for the strongest regularization
    threshold = (
        best_result["mean_accuracy"]
        - best_result["standard_error"]
    )

    eligible_results = [
        result for result in cv_results
        if result["mean_accuracy"] >= threshold
    ]

    selected_result = min(
        eligible_results,
        key=lambda result: result["C"]
    )

    best_C = selected_result["C"]

    print(f"\nHighest mean CV accuracy: "
          f"{best_result['mean_accuracy']}")
    print(f"One-SE accuracy threshold: {threshold}")
    print(f"Selected C using the one-SE rule: {best_C}")

    #fit the wining model
    final_model = LogisticRegression(
        C=best_C,
        l1_ratio=1,
        solver="liblinear",
        max_iter=5000,
        random_state=42
    )
    #fit
    final_model.fit(X_train_bow, y_train)

    # evaluate model
    y_pred = final_model.predict(X_test_bow)

    print("\n--- Test Set Results ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0)}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0)}")
    print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0)}")

    print("\nConfusion Matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Genuine reviews", "AI-generated reviews"],
            zero_division=0
        )
    )

    return final_model, y_pred
