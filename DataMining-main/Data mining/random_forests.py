from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier


#need "out of bag" evaluation instead of cross-validation
def random_forest(X_train_bow, X_test_bow, y_train, y_test):

    rf = RandomForestClassifier()
    rf.fit(X_train_bow,y_train)

    rf_pred = rf.predict(X_test_bow)

    accuracy = accuracy_score(y_test, rf_pred)
    precision = precision_score(y_test, rf_pred)
    recall = recall_score(y_test, rf_pred)
    f1 = f1_score(y_test, rf_pred)

    print("\nRandom Forests Results")
    print("--------------------------------")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1)

    return rf, rf_pred