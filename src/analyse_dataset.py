import selberai.data.load_data as load_data
import torch
import numpy as np
from pathlib import Path
import csv
from analyse_functions import simb_score, stood_score, io_score, outlier_score, plot_scores, plot_heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

def standardize(dataset):
    print("Standardizing...")
    X_train, Y_train = dataset.train
    X_val, Y_val = dataset.val
    X_test, Y_test = dataset.test

    X_scaler = StandardScaler(copy=False)
    Y_scaler = StandardScaler(copy=False)

    X_train = X_scaler.fit_transform(X_train)
    Y_train = Y_scaler.fit_transform(Y_train)
    dataset.train = X_train, Y_train

    X_val = X_scaler.transform(X_val)
    Y_val = Y_scaler.transform(Y_val)
    dataset.val = X_val, Y_val

    X_test = X_scaler.transform(X_test)
    Y_test = Y_scaler.transform(Y_test)
    dataset.test = X_test, Y_test

    return dataset

def apply_PCA(dataset):
    X_train, Y_train = dataset.train
    X_val, Y_val = dataset.val
    X_test, Y_test = dataset.test
    pca = PCA(n_components=0.99)

    print("Fitting PCA...")
    pca.fit(X_train[:1_000_000])
    print("Num features:", X_train.shape[-1])
    print("Num components:", pca.n_components_)
    # print("Explained variance:", pca.explained_variance_ratio_)
    print("Transforming...")
    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    dataset.train = X_train, Y_train
    dataset.val = X_val, Y_val
    dataset.test = X_test, Y_test

    return dataset, pca

def get_subset(dataset, max_samples):
    X_train, Y_train = dataset.train
    X_val, Y_val = dataset.val
    X_test, Y_test = dataset.test

    if len(X_train) > max_samples:
        chosen_train = np.random.choice(len(X_train), max_samples, replace=False)
        X_train = X_train[chosen_train]
        Y_train = Y_train[chosen_train]

    if len(X_val) > max_samples:
        chosen_val = np.random.choice(len(X_val), max_samples, replace=False)
        X_val = X_val[chosen_val]
        Y_val = Y_val[chosen_val]

    if len(X_test) > max_samples:
        chosen_test = np.random.choice(len(X_test), max_samples, replace=False)
        X_test = X_test[chosen_test]
        Y_test = Y_test[chosen_test]

    dataset.train = X_train, Y_train
    dataset.val = X_val, Y_val
    dataset.test = X_test, Y_test

    return dataset

def analyse(config: dict, name: str, subtask: str):
    """
    """
    print('Analysing {} for subtask {}'.format(name, subtask))
    
    ###
    # Load data
    ###
    
    # set the path to data and token we want to load
    path_to_data = config['general']['path_to_data']
    if not Path(path_to_data).exists():
        path_to_data = "/home/shared_cp/"
    path_to_token = config['dataverse']['path_to_token']
    
    results_dir = Path(config['general']['analyse_results_dir']) / f"{name}_{subtask}/"
    results_dir.mkdir(parents=True, exist_ok=True)

    form = "tabular"
    if name == "OpenCatalyst":
        form = "uniform"

    # load data
    dataset = load_data.load(name, subtask, sample_only=False, form=form, path_to_data=path_to_data, path_to_token=path_to_token)

    if name == "UberMovement" and subtask == "cities_43":
        dataset = get_subset(dataset, 2_000_000_000)

    if name == "Polianna":
        from inject_polianna import inject_pa
        dataset = inject_pa(dataset, subtask, f"/home/shared_cp/Polianna/glove6B/glove.6B.300d.txt", mode="bow")

    if name == "OpenCatalyst":
        from inject_opencatalyst import inject_oc
        dataset = inject_oc(dataset)

    print("Number of training samples:", len(dataset.train[0]))
    print("Number of validation samples:", len(dataset.val[0]))
    print("Number of test samples:", len(dataset.test[0]))

    print("standardizing values...")
    dataset = standardize(dataset)

    print("Calculating ios score...")
    full_X = torch.cat([torch.from_numpy(dataset.train[0]), torch.from_numpy(dataset.val[0]), torch.from_numpy(dataset.test[0])], dim=0)
    full_Y = torch.cat([torch.from_numpy(dataset.train[1]), torch.from_numpy(dataset.val[1]), torch.from_numpy(dataset.test[1])], dim=0)
    ios_value, ios_heatmap = io_score(full_X, full_Y)
    torch.save(ios_heatmap, results_dir / "ios_heatmap.pt")
    plot_heatmap(ios_heatmap, "Labels", "Features", results_dir / "ios_full.svg")
    plot_heatmap(ios_heatmap, "Labels", "Features", results_dir / "ios_full.png")
    del full_X
    del full_Y
    print("Done!")

    print("calculating PCA...")
    dataset, pca = apply_PCA(dataset)

    pickle.dump(pca, open(results_dir /"pca.pkl","wb"))

    print("Calculating stOOD score...")
    # features
    val_stood, test_stood, val_js, test_js = stood_score(torch.from_numpy(dataset.train[0]), torch.from_numpy(dataset.val[0]), torch.from_numpy(dataset.test[0]))
    torch.save(val_js, results_dir / "stood_val.pt")
    torch.save(test_js, results_dir / "stood_test.pt")
    plot_scores(val_js, "Features", "Validation stOOD score", results_dir / "stood_val.svg")
    plot_scores(test_js, "Features", "Testing stOOD score", results_dir / "stood_test.svg")

    # labels
    val_label_stood, test_label_stood, val_js, test_js = stood_score(torch.from_numpy(dataset.train[1]), torch.from_numpy(dataset.val[1]), torch.from_numpy(dataset.test[1]))
    torch.save(val_js, results_dir / "stood_label_val.pt")
    torch.save(test_js, results_dir / "stood_label_test.pt")
    plot_scores(val_js, "Features", "Validation stOOD score", results_dir / "stood_label_val.svg")
    plot_scores(test_js, "Features", "Testing stOOD score", results_dir / "stood_label_test.svg")
    print("Done!")


    print("Calculating simb score...")
    # features
    train_simb, train_js = simb_score(torch.from_numpy(dataset.train[0]))
    val_simb, val_js = simb_score(torch.from_numpy(dataset.val[0]))
    test_simb, test_js = simb_score(torch.from_numpy(dataset.test[0]))
    torch.save(train_js, results_dir / "simb_train.pt")
    torch.save(val_js, results_dir / "simb_val.pt")
    torch.save(test_js, results_dir / "simb_test.pt")
    plot_scores(train_js, "Features", "Training simb score", results_dir / "simb_train.svg")
    plot_scores(val_js, "Features", "Validation simb score", results_dir / "simb_val.svg")
    plot_scores(test_js, "Features", "Testing simb score", results_dir / "simb_test.svg")

    # labels
    train_label_simb, train_js = simb_score(torch.from_numpy(dataset.train[1]))
    val_label_simb, val_js = simb_score(torch.from_numpy(dataset.val[1]))
    test_label_simb, test_js = simb_score(torch.from_numpy(dataset.test[1]))
    torch.save(train_js, results_dir / "simb_label_train.pt")
    torch.save(val_js, results_dir / "simb_label_val.pt")
    torch.save(test_js, results_dir / "simb_label_test.pt")
    plot_scores(train_js, "Labels", "Training simb score", results_dir / "simb_label_train.svg")
    plot_scores(val_js, "Labels", "Validation simb score", results_dir / "simb_label_val.svg")
    plot_scores(test_js, "Labels", "Testing simb score", results_dir / "simb_label_test.svg")

    print("Done!")



    print("Calculating outlier score...")
    train_outlier_value, train_outlier_hist = outlier_score(torch.from_numpy(dataset.train[0]))
    val_outlier_value, val_outlier_hist = outlier_score(torch.from_numpy(dataset.val[0]))
    test_outlier_value, test_outlier_hist = outlier_score(torch.from_numpy(dataset.test[0]))

    full_X = torch.cat([torch.from_numpy(dataset.train[0]), torch.from_numpy(dataset.val[0]), torch.from_numpy(dataset.test[0])], dim=0)
    outlier_value, outlier_hist = outlier_score(full_X)
    plot_scores(train_outlier_hist, "Features", "Train outlier score", results_dir/ "train_outliers.svg")
    plot_scores(val_outlier_hist, "Features", "Validation outlier score", results_dir/ "val_outliers.svg")
    plot_scores(test_outlier_hist, "Features", "Test outlier score", results_dir/ "test_outliers.svg")
    plot_scores(outlier_hist, "Features", "Outlier score", results_dir/ "outliers.svg")
    torch.save(train_outlier_hist, results_dir / "train_outlier_hist.pt")
    torch.save(val_outlier_hist, results_dir / "val_outlier_hist.pt")
    torch.save(test_outlier_hist, results_dir / "test_outlier_hist.pt")
    torch.save(outlier_hist, results_dir / "outlier_hist.pt")


    with open(results_dir / "scores.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["val_stood", 
                         "test_stood", 
                         "val_label_stood",
                         "test_label_stood",

                         "train_simb", 
                         "val_simb",
                         "test_simb", 
                         "train_label_simb",
                         "val_label_simb",
                         "test_label_simb",

                         "io_score", 

                         "train_outlier_score",
                         "val_outlier_score",
                         "test_outlier_score",
                         "outlier_score",
                         ])
        writer.writerow([val_stood.item(), 
                         test_stood.item(), 
                         val_label_stood.item(),
                         test_label_stood.item(),

                         train_simb.item(), 
                         val_simb.item(), 
                         test_simb.item(), 
                         train_label_simb.item(), 
                         val_label_simb.item(), 
                         test_label_simb.item(), 

                         ios_value.item(), 
                         
                         train_outlier_value.item(),
                         val_outlier_value.item(),
                         test_outlier_value.item(),
                         outlier_value.item(),
                         ])
    print("Done!")


def plot_figures():
    from pathlib import Path
    from tqdm import tqdm
    results_dir = Path("analyse_results/")
    tensors = list(results_dir.glob("*/ios*.pt"))
    for f in tqdm(tensors):
        data = torch.load(f)
        if "ios" in f.name:
            plot_heatmap(data, "Labels", "Features", f.with_suffix(".png"))
            plot_heatmap(data, "Labels", "Features", f.with_suffix(".svg"))
        if "stood" in f.name:
            split = f.with_suffix("").name.split("_")[1]
            if split == "val":
                plot_scores(data, "Features", "Validation stOOD score", f.with_suffix(".svg"))
            if split == "test":
                plot_scores(data, "Features", "Testing stOOD score", f.with_suffix(".svg"))
        if "simb" in f.name:
            split = f.with_suffix("").name.split("_")[1]
            if split == "train":
                plot_scores(data, "Features", "Training SImb score", f.with_suffix(".svg"))
            if split == "val":
                plot_scores(data, "Features", "Validation SImb score", f.with_suffix(".svg"))
            if split == "test":
                plot_scores(data, "Features", "Testing SImb score", f.with_suffix(".svg"))

if __name__ == "__main__":
    plot_figures()