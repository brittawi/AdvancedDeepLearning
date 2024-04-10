from sklearn.metrics import jaccard_score
from sklearn.metrics import hinge_loss
import numpy as np

y_true = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [1, 0, 0, 0],
                   [1, 0, 0, 0],
                   [0, 1, 1, 0],
                   [0, 1, 1, 0],
                   [0, 1, 1, 0],
                   [0, 1, 1, 0]])
y_pred = np.array([[1, 0, 0, 0],
                   [1, 0, 0, 0],
                   [0, 1, 1, 0],
                   [1, 0, 1, 0],
                   [1, 0, 0, 1],
                   [0, 1, 0, 1],
                   [0, 1, 0, 0],
                   [0, 1, 1, 0]])

print(jaccard_score(y_true, y_pred, average="micro"))

print(jaccard_score(y_true, y_pred, average=None))

print(hinge_loss([0,1,0], [0.25,0.6,0.15]))
