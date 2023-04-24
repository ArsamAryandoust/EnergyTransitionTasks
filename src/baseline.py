import selberai.data.load_data as load_data
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
import shutil
from pathlib import Path

import time
energy_file = "energy_consumption.csv"
csv_handler = CSVHandler(energy_file)

@measure_energy(handler = csv_handler)
def train_RF_baseline(train_data: tuple[np.ndarray, np.ndarray], test_data: tuple[np.ndarray, np.ndarray], num_estimators: int, seed: int = 42, task: str="regression"):
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

def run_baseline(config: dict, name: str, subtask: str):
    """
    Trains a RF baseline on the specified dataset. Also calculates energy consumption and score and saves both into baseline_results/
    """
    path_to_data = config['general']['path_to_data']
    path_to_token = config['dataverse']['path_to_token']
    results_dir = Path(config['general']['baseline_results_dir'])
    results_dir.mkdir(exist_ok=True)

    # load data
    dataset = load_data.load(name, subtask, sample_only=False, tabular=True, path_to_data=path_to_data, path_to_token=path_to_token)

    # normalize the data to have 0 mean, 1 variance
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X, Y = dataset.train
    dataset.train = scaler.fit_transform(X), Y
    X, Y = dataset.test
    dataset.test = scaler.transform(X), Y

    score = train_RF_baseline(dataset.train, dataset.test, 128)
    with open(results_dir / f"{name}_{subtask}_score.txt", "w") as f:
        f.write(str(score))
    csv_handler.save_data()
    shutil.move(energy_file, results_dir / f"{name}_{subtask}_energy.csv")