import numpy as np
from sklearn import metrics
f1 = [1, 2, 3]
f2 = [1, 2, 3]
#fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
print(metrics.auc(f2, f1))