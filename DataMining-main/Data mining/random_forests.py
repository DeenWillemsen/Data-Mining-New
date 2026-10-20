from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def random_forest(X_train_bow, X_test_bow, y_train, y_test):
    param_grid = {
        'n_estimators': [100, 300, 500],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [1, 2, 4]
    }
    
    keys, values = zip(*param_grid.items())
    grid = [dict(zip(keys, v)) for v in product(*values)]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Grid searching RF hyperparameters with 5-fold cv")
    results = []

    for params in grid:
        clf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            criterion="log_loss",
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        
        scores = cross_val_score(clf, X_train_bow, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
        mean_acc = scores.mean()
        
        results.append((params, mean_acc))
       

    best_params = max(results, key=lambda x: x[1])[0]
    print(f"\nBest params selected: {best_params}")

    rf = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        criterion="log_loss",
        max_features="sqrt",
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