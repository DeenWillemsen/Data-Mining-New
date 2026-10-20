from itertools import product
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def gradient_boosting(X_train_bow, X_test_bow, y_train, y_test):

    param_grid = {
        'learning_rate': [0.01,0.05,0.1],
        'min_samples_split': [10,20,50],
        'min_samples_leaf': [1,2,5],
        'max_depth': [3,4,5,6]
    }

    keys, values = zip(*param_grid.items())
    grid = [dict(zip(keys, v)) for v in product(*values)]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Grid searching GB hyperparameters with 5-fold cv")
    results = []

    for params in grid:
        clf = GradientBoostingClassifier(
            learning_rate=params['learning_rate'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            n_estimators=100, #test with only 100 estimators. More estimators takes far too long to test.
            max_depth=params['max_depth'],
            subsample=0.9,
            max_features=4,
            random_state=42
        )
        
        scores = cross_val_score(clf, X_train_bow, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
        mean_acc = scores.mean()
    
        results.append((params, mean_acc))


    best_params = max(results, key=lambda x: x[1])[0]
    print(f"\nBest params selected: {best_params}")

    gb = GradientBoostingClassifier(
        learning_rate=best_params['learning_rate'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        n_estimators=900,
        max_depth=best_params['max_depth'],
        subsample=0.9,
        max_features=4,
        random_state=42
    )
    gb.fit(X_train_bow, y_train)
    preds = gb.predict(X_test_bow)

    print("\n--- GB Test Set Results ---")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, preds, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_test, preds, zero_division=0):.4f}")

    return gb, preds