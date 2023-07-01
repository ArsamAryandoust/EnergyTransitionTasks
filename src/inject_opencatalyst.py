from collections import Counter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
import os
home_dir = Path(os.path.expanduser('~'))
sys.path.append(str(home_dir / "selberai"))
from selberai.data.load_data import load

def get_max_atomic_number(dataset):
    max_atom = 0
    for sample_id in dataset.train["x_s"].keys():
        for atom in dataset.train["x_s"][sample_id]:
            max_atom = max(max_atom, atom)

    for sample_id in dataset.val["x_s"].keys():
        for atom in dataset.val["x_s"][sample_id]:
            max_atom = max(max_atom, atom)

    for sample_id in dataset.test["x_s"].keys():
        for atom in dataset.test["x_s"][sample_id]:
            max_atom = max(max_atom, atom)
    
    return max_atom

def calculate_bag_of_atoms(X_split: dict[int, list[int]], add: np.ndarray, num_atoms: int) -> np.ndarray:
    features = []
    for sample_id in X_split.keys():
        sample_count = np.zeros((num_atoms))   
        counter = Counter()
        counter.update(X_split[sample_id])
        for atom, count in counter.items():
            sample_count[atom - 1] = count

        sample_additional = [[add[:, atom]] * count for atom, count in counter.items()]
        sample_additional = np.array([s[0] for s in sample_additional])
        sample_features = np.concatenate([sample_count, sample_additional.mean(axis=0), sample_additional.std(axis=0)])
        features.append(sample_features)
    return np.stack(features)

def inject_oc(dataset, mode: str = "boa"):

    print("Injecting dataset...")


    if "y_t" not in dataset.train:
        raise ValueError("Dataset has no single label, cannot inject.")

    num_atoms = get_max_atomic_number(dataset)

    additional_data = np.concatenate([dataset.add["x_s_1"], dataset.add["x_s_2"], dataset.add["x_s_3"]])

    X_train, Y_train = dataset.train["x_s"], dataset.train["y_t"]
    X_val, Y_val = dataset.val["x_s"], dataset.val["y_t"]
    X_test, Y_test = dataset.test["x_s"], dataset.test["y_t"]
        
    if mode == "boa":
        X_train = calculate_bag_of_atoms(X_train, additional_data, num_atoms)
        X_val = calculate_bag_of_atoms(X_val, additional_data, num_atoms)
        X_test = calculate_bag_of_atoms(X_test, additional_data, num_atoms)

    Y_train = np.array(list(Y_train.values())).reshape(-1, 1)
    Y_val = np.array(list(Y_val.values())).reshape(-1, 1)
    Y_test = np.array(list(Y_test.values())).reshape(-1, 1)

    dataset.train = X_train, Y_train
    dataset.val = X_val, Y_val
    dataset.test = X_test, Y_test

    return dataset

def inject_oc_split(split:dict, additional_data:dict[str, np.ndarray]):

    if "y_t" not in split:
        raise ValueError("Dataset has no single label, cannot inject.")

    num_atoms = 118

    additional_data = np.concatenate([additional_data["x_s_1"], additional_data["x_s_2"], additional_data["x_s_3"]])

    X_split, Y_split = split["x_s"], split["y_t"]
        
    X_split = calculate_bag_of_atoms(X_split, additional_data, num_atoms)

    Y_split = np.array(list(Y_split.values())).reshape(-1, 1)

    return X_split, Y_split


if __name__ == "__main__":
    name = "OpenCatalyst"
    # subtask = "oc20_s2ef"
    subtask = "oc20_is2re"
    # subtask = "oc22_s2ef"
    # subtask = "oc22_is2res"

    path_to_data = "/home/shared_cp/"

    dataset = load(name, subtask, sample_only=True, form="uniform", path_to_data=path_to_data)

    dataset = inject_oc(dataset)

    print(dataset.train[0].shape)
