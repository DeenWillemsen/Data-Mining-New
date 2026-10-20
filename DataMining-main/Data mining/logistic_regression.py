
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


def logistic_regression(
    X_train_bow,
    X_test_bow,
    y_train,
    y_test,
    vectorizer
):
    # Candidate values for C = 1 / lambda.
    # Smaller C means stronger L1 regularization.
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]

    # Use the same stratified folds for every candidate C.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    print("\n--- Logistic Regression: Lasso ---")
    print("Selecting C using 5-fold cross-validation...")

    # Evaluate each value of C using training data only.
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
            f"Mean accuracy={mean_accuracy:.4f} "
            f"SE={standard_error:.4f}"
        )

    # Find the candidate with the highest mean CV accuracy.
    best_result = max(
        cv_results,
        key=lambda result: result["mean_accuracy"]
    )

    # One-standard-error rule:
    # accept models whose accuracy is within one SE
    # of the best mean accuracy.
    threshold = (
        best_result["mean_accuracy"]
        - best_result["standard_error"]
    )

    eligible_results = [
        result for result in cv_results
        if result["mean_accuracy"] >= threshold
    ]

    # Choose the smallest C among eligible models.
    # This gives the strongest regularization within the
    # one-standard-error threshold.
    selected_result = min(
        eligible_results,
        key=lambda result: result["C"]
    )

    best_C = selected_result["C"]

    print(f"\nHighest mean CV accuracy: "
          f"{best_result['mean_accuracy']:.4f}")
    print(f"One-SE accuracy threshold: {threshold:.4f}")
    print(f"Selected C using the one-SE rule: {best_C}")

    # Fit the selected model on the complete training set.
    final_model = LogisticRegression(
        C=best_C,
        penalty="l1",
        solver="liblinear",
        max_iter=5000,
        random_state=42
    )

    final_model.fit(X_train_bow, y_train)

    # Evaluate only once on the held-out test set.
    y_pred = final_model.predict(X_test_bow)

    print("\n--- Test Set Results ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")

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

    # Inspect the final model's word coefficients.
    feature_names = vectorizer.get_feature_names_out()
    coefficients = final_model.coef_[0]

    ai_indices = [
        i for i in np.argsort(coefficients)[::-1]
        if coefficients[i] > 0
    ][:5]

    human_indices = [
        i for i in np.argsort(coefficients)
        if coefficients[i] < 0
    ][:5]

    print("\nTop words indicating AI-generated reviews:")
    for i in ai_indices:
        print(f"{feature_names[i]}: {coefficients[i]:.4f}")

    print("\nTop words indicating genuine reviews:")
    for i in human_indices:
        print(f"{feature_names[i]}: {coefficients[i]:.4f}")

    return final_model, y_pred
