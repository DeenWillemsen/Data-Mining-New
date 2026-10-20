from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score


def gradient_boosting(X_train_bow, X_test_bow, y_train, y_test):

    gb = GradientBoostingClassifier()
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

    