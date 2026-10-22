import numpy as np

def top_features(model, model_name, vectorizer, X_train_bow, y_train, k=5):
    feature_names = vectorizer.get_feature_names_out()
    y_train = np.asarray(y_train)

    print(f"\n--- Top {k} features: {model_name} ---")

    if model_name == "MNNB":
        scores = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    elif model_name == "Logistic Regression":
        scores = model.coef_[0]
    else:
        scores = None

    if scores is not None:
        order = np.argsort(scores)

        print(f"Towards class {model.classes_[1]}:")
        for i in order[::-1][:k]:
            print(f"  {feature_names[i]:<15} {scores[i]:.3f}")

        print(f"Towards class {model.classes_[0]}:")
        for i in order[:k]:
            print(f"  {feature_names[i]:<15} {scores[i]:.3f}")

    else:
        importances = model.feature_importances_
        mean_1 = np.asarray(X_train_bow[y_train == model.classes_[1]].mean(axis=0)).ravel()
        mean_0 = np.asarray(X_train_bow[y_train == model.classes_[0]].mean(axis=0)).ravel()

        for i in np.argsort(importances)[::-1][:k]:
            if mean_1[i] > mean_0[i]:
                more_in = model.classes_[1]
            else:
                more_in = model.classes_[0]
            print(f"  {feature_names[i]:<15} {importances[i]:.4f}  (more in class {more_in})")