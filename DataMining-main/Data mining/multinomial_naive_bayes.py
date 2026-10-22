from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def multinomial_naive_bayes(X_train_bow, X_test_bow, y_train, y_test):

    #candidate hyperparameters for smoothing
    param_grid = {
        "alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        "fit_prior": [True, False]
    }

    #select the best alpha using 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=MultinomialNB(),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    #train and tune 
    grid_search.fit(X_train_bow, y_train)

    # best model via cross validation
    model = grid_search.best_estimator_

    # test
    y_pred = model.predict(X_test_bow)

    #performance measures
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\nMultinomial Naive Bayes Results")
    print("--------------------------------")
    print("Best alpha:", grid_search.best_params_["alpha"])
    print("Best CV accuracy:", grid_search.best_score_)
    print("Test accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, y_pred
