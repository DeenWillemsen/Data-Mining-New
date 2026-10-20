import numpy as np
from scipy.stats import chi2_contingency

def test_sentiment_accuracy(y_test, preds, sent_test, model_name):
    correct_predictions = (np.array(y_test) == np.array(preds))
    sent_array = np.array(sent_test)
    
    pos_mask = (sent_array == 'POS')
    neg_mask = (sent_array == 'NEG')
    
    pos_correct = np.sum(correct_predictions[pos_mask])
    pos_incorrect = np.sum(~correct_predictions[pos_mask])
    
    neg_correct = np.sum(correct_predictions[neg_mask])
    neg_incorrect = np.sum(~correct_predictions[neg_mask])
    
    contingency_table = [
        [pos_correct, pos_incorrect],
        [neg_correct, neg_incorrect]
    ]
    
    chi2, p_val, _, _ = chi2_contingency(contingency_table)
    
    acc_pos = pos_correct / (pos_correct + pos_incorrect) if (pos_correct + pos_incorrect) > 0 else 0
    acc_neg = neg_correct / (neg_correct + neg_incorrect) if (neg_correct + neg_incorrect) > 0 else 0
    
    print(f"Accuracy on POS Reviews: {acc_pos:.4f}")
    print(f"Accuracy on NEG Reviews: {acc_neg:.4f}")
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p_val:.4e}")
    
    if p_val < 0.05:
        better_group = "POS" if acc_pos > acc_neg else "NEG"
        print(f"Conclusion: SIGNIFICANT difference (p < 0.05).")
        print(f"RF is significantly better at distinguishing {better_group} reviews.")
    else:
        print("Conclusion: NO significant difference in accuracy between POS and NEG reviews.")