import optuna
from log_res import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

LR_PARAMS_DICT = {
    'C': 0.,
    'random_state': 777,
    'iters': 5,
    'batch_size': 1000,
    'step': 1.65
}

# 1. Define an objective function to be maximized.
def objective(trial):

    X_data, y_data = make_classification(
        n_samples=50000, n_features=20, 
        n_classes=2, n_informative=20, 
        n_redundant=0, random_state=777
    )

    # 2. Suggest values for the hyperparameters using a trial object.
    
    model_iters = trial.suggest_int('model_iters', 5, 9)
    model_step = trial.suggest_float("model_step", 1, 2)
    #model_c = trial.suggest_float("model_c", 1e-2, 2)

    classifier_obj = LogisticRegression(
        C=0.1, 
        random_state=LR_PARAMS_DICT['random_state'],
        iters=model_iters,
        batch_size=LR_PARAMS_DICT['batch_size'],
        step=model_step
    )

    score = cross_val_score(classifier_obj, X_data, y_data, n_jobs=-1, cv=4, scoring='accuracy')
    accuracy = score.mean()
    
    return accuracy


if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print(study.best_trial)
