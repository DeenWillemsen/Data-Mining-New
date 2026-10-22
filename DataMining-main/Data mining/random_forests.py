from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def random_forest(X_train_bow, X_test_bow, y_train, y_test):
    param_grid = {
        'n_estimators': [100, 300, 500],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['log2', 'sqrt', 0.05, 0.1, 0.2]
    }


    keys, values = zip(*param_grid.items())
    grid = [dict(zip(keys, v)) for v in product(*values)]

    print("Tuning RF hyperparameters with OOB accuracy")
    results = []

    for params in grid:
        clf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            criterion="log_loss",
            class_weight="balanced",
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )


        clf.fit(X_train_bow, y_train)
        results.append((params, clf.oob_score_))


    best_params, best_oob = max(results, key=lambda x: x[1])
    print(f"\nBest params selected: {best_params}")
    print(f"OOB accuracy: {best_oob:.4f}")


    rf = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        criterion="log_loss",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train_bow, y_train)
    preds = rf.predict(X_test_bow)

    print("\n--- RF Test Set Results ---")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, preds, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_test, preds, zero_division=0):.4f}")

    return rf, preds