import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

def random_forest(X_train_bow, X_test_bow, y_train, y_test):
    
    rf = RandomForestClassifier(
        n_estimators = 200,
        criterion = "log_loss",
        max_depth = None,
        min_samples_split = 10,
        min_samples_leaf = 2,
        min_weight_fraction_leaf = 0.0,
        max_features = "sqrt",
        max_leaf_nodes = None,
        min_impurity_decrease = 0.0,
        bootstrap = True,
        oob_score = False,
        n_jobs = None,
        verbose = 0,
        max_samples = None,
        class_weight="balanced"
    )


    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

    cv_results = cross_validate(rf, X_train_bow, y_train, cv=5, scoring=scoring_metrics)


    print("\n5-Fold Cross-Validation Results")
    print("--------------------------------")
    print(f"Mean Accuracy:  {np.mean(cv_results['test_accuracy']):.4f}")
    print(f"Mean Precision: {np.mean(cv_results['test_precision']):.4f}")
    print(f"Mean Recall:    {np.mean(cv_results['test_recall']):.4f}")
    print(f"Mean F1 Score:  {np.mean(cv_results['test_f1']):.4f}")

    rf.fit(X_train_bow, y_train)

    print("Evaluating final model on the Test Set...")
    final_preds = rf.predict(X_test_bow)

    test_accuracy = accuracy_score(y_test, final_preds)
    test_precision = precision_score(y_test, final_preds)
    test_recall = recall_score(y_test, final_preds)
    test_f1 = f1_score(y_test, final_preds)

    print(f"Final Test Accuracy:  {test_accuracy:.4f}")
    print(f"Final Test Precision: {test_precision:.4f}")
    print(f"Final Test Recall:    {test_recall:.4f}")
    print(f"Final Test F1 Score:  {test_f1:.4f}")
    
    return rf