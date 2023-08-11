from log_res import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import signal
import traceback
import sys


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

LR_PARAMS_DICT = {
    'C': 0.1,
    'random_state': 777,
    'iters': 5,
    'batch_size': 1000,
    'step': 1.65
}



if __name__ == '__main__':

    X_data, y_data = make_classification(
        n_samples=10000, n_features=20, 
        n_classes=2, n_informative=20, 
        n_redundant=0,random_state=42
        )
    
    model = LogisticRegression(**LR_PARAMS_DICT)

    print(np.mean(cross_val_score(model, X_data, y_data, cv=5, scoring='accuracy')))
