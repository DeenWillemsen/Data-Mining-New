from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def multinomial_naive_bayes(X_train_bow, X_test_bow, y_train, y_test):

    #create model
    model = MultinomialNB()

    #train model
    model.fit(X_train_bow, y_train)

    #make predictions
    y_pred = model.predict(X_test_bow)

    #calculate performance measures
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    #print results
    print("\nMultinomial Naive Bayes Results")
    print("--------------------------------")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1)

    return model, y_pred

    #new