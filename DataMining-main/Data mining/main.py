print("MAIN SCRIPT STARTED")
import itertools
from preprocess_data import preprocess_data
from multinomial_naive_bayes import multinomial_naive_bayes
from random_forests import random_forest
from gradient_boosting import gradient_boosting
from logistic_regression import logistic_regression
from Single_Classification_Tree import SCTree
from mcnemar_test import mcnemar_test


print("Data preprocessing started")
# Preprocess the data
X_train_bow, X_test_bow, y_train, y_test, vectorizer = preprocess_data()

predictions = {}

print(" -- MNNB -- ")
_, predictions['MNNB'] = multinomial_naive_bayes(
    X_train_bow,
    X_test_bow,
    y_train,
    y_test
)

print("\n -- Random Forest -- ")
_, predictions['Random Forests'] = random_forest(
    X_train_bow,
    X_test_bow,
    y_train,
    y_test
)

print("\n -- Gradient Boosting -- ")
_, predictions['Gradient Boosting'] = gradient_boosting(
    X_train_bow,
    X_test_bow,
    y_train,
    y_test
)

print("\n -- Logistic Regression -- ")
_, predictions['Logistic Regression'] = logistic_regression(
    X_train_bow,
    X_test_bow,
    y_train,
    y_test,
    vectorizer
)

print(" -- Single Classification Tree --")
_, predictions['SCTree'] = SCTree(
    X_train_bow,
    X_test_bow,
    y_train,
    y_test
)

print (" -- McNemar's Test -- ")
model_names = list(predictions.keys())
pairs = list(itertools.combinations(model_names, 2))

# Collect results quietly
results_data = []
for name1, name2 in pairs:
    test_result = mcnemar_test(
        y_test, 
        predictions[name1], 
        predictions[name2], 
        name1, 
        name2, 
        alpha=0.01
    )
    results_data.append(test_result)


# ----- AI GENERATED CODE ---- -> Generated purely for easier visualization of McNemar's Test
print(f"\n{'Model A':<22} | {'Model B':<22} | {'P-Value':<12} | {'Significant?'}")
print("-" * 75)

for res in results_data:
    pval = f"{res['P-Value']:.4e}"
    sig = res['Significant (alpha=0.01)']

    sig_marker = f"🟢 {sig}" if sig == "Yes" else f"⚪ {sig}"
    
    print(f"{res['Model A']:<22} | {res['Model B']:<22} | {pval:<12} | {sig_marker}")

print("-" * 75)
