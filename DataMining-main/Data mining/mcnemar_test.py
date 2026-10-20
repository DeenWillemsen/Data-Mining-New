import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def mcnemar_test(y_true, preds_1, preds_2, name_1, name_2, alpha=0.01):

    correct_1 = (preds_1 == y_true)
    correct_2 = (preds_2 == y_true)
    

    # [[Both correct, Model 1 correct / Model 2 wrong],
    #  [Model 1 wrong / Model 2 correct, Both wrong]]
    yy = np.sum(correct_1 & correct_2)
    yn = np.sum(correct_1 & ~correct_2)
    ny = np.sum(~correct_1 & correct_2)
    nn = np.sum(~correct_1 & ~correct_2)
    
    table = [[yy, yn],
             [ny, nn]]
    
    result = mcnemar(table, exact=False, correction=True)
    is_significant = "Yes" if result.pvalue < alpha else "No"
    
    return {
        "Model A": name_1,
        "Model B": name_2,
        "Chi-Square": result.statistic,
        "P-Value": result.pvalue,
        f"Significant (alpha={alpha})": is_significant
    }