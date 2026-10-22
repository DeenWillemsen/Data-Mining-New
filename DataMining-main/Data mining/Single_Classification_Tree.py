import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def SCTree(X_train_bow, X_test_bow, y_train, y_test):

    depths = [None, 5, 10, 20, 30, 50, 100]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Tuning max_depth with 5-fold cv")
    results = []

    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42, class_weight="balanced")
        scores = cross_val_score(clf, X_train_bow, y_train, cv=skf, scoring="accuracy")
        
        mean_acc = scores.mean()
        results.append((d, mean_acc))
        

    best_depth, best_acc = max(results, key=lambda x: x[1])
    print(f"\nBest depth selected: {best_depth}")
    print(f"Best CV accuracy: {best_acc:.4f}")

    dt = DecisionTreeClassifier(max_depth=best_depth, random_state=42, class_weight="balanced")
    dt.fit(X_train_bow, y_train)
    preds = dt.predict(X_test_bow)

    print("\n--- Test Set Results ---")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, preds, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_test, preds, zero_division=0):.4f}")


    return dt, preds