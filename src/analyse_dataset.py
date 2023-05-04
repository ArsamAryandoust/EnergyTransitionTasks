import selberai.data.load_data as load_data
import torch
from pathlib import Path
import csv
from analyse_functions import simb_score, stood_score, ios_score, plot_scores, plot_heatmap


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
    
    results_dir = Path(f"analyse_results/{name}_{subtask}/")
    results_dir.mkdir(parents=True, exist_ok=True)

    # load data
    dataset = load_data.load(name, subtask, sample_only=True, form="tabular", path_to_data=path_to_data, path_to_token=path_to_token)

    print("Calculating stOOD score...")
    val_stood, test_stood, val_js, test_js = stood_score(torch.from_numpy(dataset.train[0]), torch.from_numpy(dataset.val[0]), torch.from_numpy(dataset.test[0]))

    plot_scores(val_js, "Features", "Validation stOOD score", results_dir / "stood_val.svg")
    plot_scores(test_js, "Features", "Testing stOOD score", results_dir / "stood_test.svg")
    print("Done!")

    print("Calculating simb score...")
    train_simb, train_js = simb_score(torch.from_numpy(dataset.train[0]))
    val_simb, val_js = simb_score(torch.from_numpy(dataset.val[0]))
    test_simb, test_js = simb_score(torch.from_numpy(dataset.test[0]))

    plot_scores(train_js, "Features", "Training simb score", results_dir / "simb_train.svg")
    plot_scores(val_js, "Features", "Validation simb score", results_dir / "simb_val.svg")
    plot_scores(test_js, "Features", "Testing simb score", results_dir / "simb_test.svg")
    print("Done!")

    print("Calculating ios score...")
    full_X = torch.cat([torch.from_numpy(dataset.train[0]), torch.from_numpy(dataset.val[0]), torch.from_numpy(dataset.test[0])], dim=0)
    full_Y = torch.cat([torch.from_numpy(dataset.train[1]), torch.from_numpy(dataset.val[1]), torch.from_numpy(dataset.test[1])], dim=0)
    ios_value, ios_heatmap = ios_score(full_X, full_Y)

    plot_heatmap(ios_heatmap, "Labels", "Features", results_dir / "ios_full.svg")
    with open(results_dir / "scores.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["val_stood", "test_stood", "train_simb", "val_simb", "test_simb", "ios_score"])
        writer.writerow([val_stood.item(), test_stood.item(), train_simb.item(), val_simb.item(), test_simb.item(), ios_value.item()])
    print("Done!")