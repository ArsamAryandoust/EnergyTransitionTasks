from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import time

def RFBaseline(train_data: tuple[np.ndarray, np.ndarray], test_data: tuple[np.ndarray, np.ndarray], num_estimators: int, seed: int, task: str="regression"):
    """
    Fits a Random Forest model to the training data. 
    
    The model can be either a regressor or a classifier for the respective task "regression" or "classification"
    """

    print("Fitting a RF regressor:")
    t = time.time()
    if task == "regression":
        model = RandomForestRegressor(n_estimators=num_estimators, random_state=seed, n_jobs=-1)
    elif task == "classification":
        model = RandomForestClassifier(n_estimators=num_estimators, random_state=seed, n_jobs=-1)
    else:
        raise ValueError(f"Unknown task for RF model: {task}. Options are 'regression' and 'classification'.")
    model.fit(*train_data)
    print(time.time() - t, "seconds elapsed!")
    score = model.score(*test_data)
    print("score:", score)

    return score