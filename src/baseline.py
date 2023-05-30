import selberai.data.load_data as load_data
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
import shutil
from pathlib import Path
from xgboost import XGBClassifier, XGBRegressor
import os
from inject_polianna import inject_pa
from inject_opencatalyst import inject_oc

import time
import math
energy_file = "energy_consumption.csv"
csv_handler = CSVHandler(energy_file)

from codecarbon import track_emissions

@track_emissions(offline=True, country_iso_code="CHE")
@measure_energy(handler = csv_handler)
def train_RF_baseline(train_data: tuple[np.ndarray, np.ndarray], test_data: tuple[np.ndarray, np.ndarray], num_estimators: int, seed: int = 42, task: str="regression", max_samples:float = 0.1, max_depth:int=None):
    """
    Fits a Random Forest model to the training data. 
    
    The model can be either a regressor or a classifier for the respective task "regression" or "classification"
    """

    print("Fitting a RF model:")

    print(train_data[0].shape)
    print(train_data[1].shape)

    t = time.time()
    if task == "regression":
        model = RandomForestRegressor(n_estimators=num_estimators, 
                                      random_state=seed, 
                                      n_jobs=os.cpu_count(),
                                      max_depth=max_depth,
                                      max_samples=max_samples,
                                      verbose=1
                                      )
    elif task == "classification":
        model = RandomForestClassifier(n_estimators=num_estimators, random_state=seed, n_jobs=os.cpu_count(), verbose=1, max_samples=1.0, max_depth=max_depth)
    else:
        raise ValueError(f"Unknown task for RF model: {task}. Options are 'regression' and 'classification'.")
    X_train, Y_train = train_data
    model.fit(X_train, Y_train.squeeze())
    print(time.time() - t, "seconds elapsed!")
    X_test, Y_test = test_data
    MAX_PREDICTION_SAMPLES = 200_000_000
    if len(X_test) > MAX_PREDICTION_SAMPLES:
        scores = []
        for i in range(math.ceil(len(X_test) / MAX_PREDICTION_SAMPLES)):
            x = X_test[i*MAX_PREDICTION_SAMPLES:(i+1)*MAX_PREDICTION_SAMPLES]
            y = Y_test[i*MAX_PREDICTION_SAMPLES:(i+1)*MAX_PREDICTION_SAMPLES]
            scores.append(model.score(x, y.squeeze()))
        print(scores)
        score = np.array(scores).mean()
    else:
        score = model.score(X_test, Y_test.squeeze())
    print("score:", score)
    print("depth:", max([estimator.tree_.max_depth for estimator in model.estimators_]))


    return score

@measure_energy(handler = csv_handler)
def train_XGBoost_baseline(train_data: tuple[np.ndarray, np.ndarray], test_data: tuple[np.ndarray, np.ndarray], num_estimators: int, task: str="regression"):
    print("Fitting a XGBoost model:")
    t = time.time()
    if task == "regression":
        model = XGBRegressor(n_estimators=num_estimators, tree_method="gpu_hist", gpu_id=2, max_depth=2, learning_rate=1)
    elif task == "classification":
        model = XGBClassifier(n_estimators=num_estimators, tree_method="gpu_hist", gpu_id=2, max_depth=2, learning_rate=1)
    else:
        raise ValueError(f"Unknown task for XGB model: {task}. Options are 'regression' and 'classification'.")
    model.fit(*train_data)
    print(time.time() - t, "seconds elapsed!")
    score = model.score(*test_data)
    print("score:", score)

    return score

def run_baseline(config: dict, name: str, subtask: str, max_samples=1.0, standardize=False):
    """
    Trains a RF baseline on the specified dataset. Also calculates energy consumption and score and saves both into baseline_results/
    """
    max_depth = 20
    num_trees = 128

    print(f"Running a baseline on {name} {subtask}, with max_depth:{max_depth}, num_trees:{num_trees}, max_samples:{max_samples}")
    path_to_data = config['general']['path_to_data']
    if not Path(path_to_data).exists():
        path_to_data = "/home/shared_cp/"
    path_to_token = config['dataverse']['path_to_token']
    results_dir = Path(config['general']['baseline_results_dir']) / f"{name}_{subtask}/"
    results_dir.mkdir(exist_ok=True, parents=True)

    form = "tabular"
    if name == "OpenCatalyst":
        form = "uniform"

    # load data
    dataset = load_data.load(name, subtask, sample_only=False, form=form, path_to_data=path_to_data, path_to_token=path_to_token)

    if name == "OpenCatalyst":
        dataset = inject_oc(dataset)
    
    if name == "Polianna":
        dataset = inject_pa(dataset, subtask, f"/home/shared_cp/Polianna/glove6B/glove.6B.{50}d.txt", mode="bow")

    # normalize the data to have 0 mean, 1 variance
    if standardize:
        print("Standardizing data...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X, Y = dataset.train
        dataset.train = scaler.fit_transform(X), Y
        X, Y = dataset.test
        dataset.test = scaler.transform(X), Y
    
    if name == "Polianna":
        score = train_RF_baseline(dataset.train, dataset.test, num_trees, max_samples=max_samples, max_depth=max_depth, task="classification")
    else:
        score = train_RF_baseline(dataset.train, dataset.test, num_trees, max_samples=max_samples, max_depth=max_depth)
    
    
    with open(results_dir / f"score_RF.txt", "w") as f:
        f.write(str(score))
    csv_handler.save_data()
    shutil.move(energy_file, results_dir / f"energy_RF_{num_trees}_{max_samples}_{max_depth if max_depth is not None else 'inf'}.csv")
    shutil.move("emissions.csv", results_dir / f"emissions_RF_{num_trees}_{max_samples}_{max_depth if max_depth is not None else 'inf'}.csv")

if __name__ == "__main__":

    pass