import math
import torch
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

LOG_EPSILON = 1e-39
sns.set(style='darkgrid')

def remove_outliers_with_std(x):
    mean, std = torch.mean(x), torch.std(x)
    threshold = 3
    outlier_indices = torch.abs(x - mean) > threshold * std
    x = x[~outlier_indices]
    return x

def get_quantiles(x, quantiles_list):
    # torch raises a RuntimeException when calling quantile with a tensor with more than 16M elements.
    try:
        quantiles = torch.quantile(x, torch.tensor(quantiles_list))
    except RuntimeError:
        import numpy as np
        quantiles = np.quantile(x.numpy(), quantiles_list)
        quantiles = torch.from_numpy(quantiles)
    return quantiles

def remove_outliers_with_quantiles(x, q):
    right_q = q + (1 - q)/2
    left_q = (1 - q)/2
    left_quantile, right_quantile = get_quantiles([left_q, right_q])
    outlier_indices = torch.logical_or(x < left_quantile, x > right_quantile)
    x = x[~outlier_indices]
    return x

def remove_outliers_with_tukey_fences(x):
    q1, q3 = get_quantiles(x, [0.25, 0.75])
    iqr = q3 - q1
    f1 = q1 - 1.5*iqr
    f2 = q3 + 1.5*iqr
    outliers_indices = torch.logical_or(x < f1, x > f2)
    x = x[~outliers_indices]
    return x

def create_distribution(x, min, max, n_bins=10000):
    histogram = torch.histc(x, bins=n_bins, min=min, max=max)
    distribution = histogram / torch.sum(histogram)
    return distribution
    
def compute_js_divergence(p, q):
    part1 = torch.sum(p * torch.log2(2.0 * p + LOG_EPSILON) - p * torch.log2(p + q + LOG_EPSILON))
    part2 = torch.sum(q * torch.log2(2.0 * q + LOG_EPSILON) - q * torch.log2(q + p + LOG_EPSILON))
    return 0.5 * part1 + 0.5 * part2

def plot_scores(scores, x_label, y_label, path, show=False):
    indices = [i for i in range(len(scores))]
    fig, ax = plt.subplots()
    sns.set(style='darkgrid')
    ax.bar(indices, scores, edgecolor="none")
    ax.set(xlabel=x_label, ylabel=y_label)
    plt.tight_layout()
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()

def plot_heatmap(x, x_label, y_label, path, show=False):
    annot = True
    if x.shape[0] > 20 or x.shape[1] > 20:
        annot = False
    ax = sns.heatmap(x.numpy(), annot=annot, cmap=sns.cm.rocket_r)
    sns.set(style='darkgrid')
    ax.set(xlabel=x_label, ylabel=y_label)
    plt.tight_layout()
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()

def simb_score(dataset, n_bins=10000, features_to_skip=[], device='cpu'):
    n_features = dataset.shape[1]
    js_divergences = []
    uniform_distribution = torch.ones(n_bins).to(device) / n_bins

    for feature in tqdm(range(n_features)):
        if feature in features_to_skip:
            continue
        dataset_slice = dataset[:, feature]
        dataset_slice = remove_outliers_with_tukey_fences(dataset_slice)
        min, max = torch.min(dataset_slice), torch.max(dataset_slice)
        distribution = create_distribution(dataset_slice, min, max, n_bins=n_bins)
        js_divergence= compute_js_divergence(distribution, uniform_distribution)
        js_divergences.append(js_divergence)
    
    js_divergences = torch.tensor(js_divergences).to(device)
    simb_score = torch.mean(js_divergences)
    return simb_score, js_divergences

def stood_score(train_set, validation_set, test_set, n_bins=10000, features_to_skip=[], device='cpu'):
    n_features = train_set.shape[1]
    js_divergences_validation, js_divergences_test = [], []

    for feature in tqdm(range(n_features)):
        if feature in features_to_skip:
            continue
        train_slice = train_set[:, feature]
        validation_slice = validation_set[:, feature]
        test_slice = test_set[:, feature]

        train_slice = remove_outliers_with_tukey_fences(train_slice)
        validation_slice = remove_outliers_with_tukey_fences(validation_slice)
        test_slice = remove_outliers_with_tukey_fences(test_slice)

        slice = torch.cat((train_slice, validation_slice, test_slice), dim=0)
        min, max = torch.min(slice), torch.max(slice)

        train_distribution = create_distribution(train_slice, min, max, n_bins=n_bins)
        validation_distribution = create_distribution(validation_slice, min, max, n_bins=n_bins)
        test_distribution = create_distribution(test_slice, min, max, n_bins=n_bins)

        js_divergence_validation = compute_js_divergence(train_distribution, validation_distribution)
        js_divergence_test = compute_js_divergence(train_distribution, test_distribution)

        js_divergences_validation.append(js_divergence_validation)
        js_divergences_test.append(js_divergence_test)
    
    js_divergences_validation = torch.tensor(js_divergences_validation).to(device)
    js_divergences_test = torch.tensor(js_divergences_test).to(device)

    stood_score_val = torch.mean(js_divergences_validation)
    stood_score_test = torch.mean(js_divergences_test)
    return stood_score_val, stood_score_test, js_divergences_validation, js_divergences_test

def outlier_score(dataset, features_to_skip=[], device='cpu'):
    n_features = dataset.shape[1]
    outlier_subscores = []

    for feature in tqdm(range(n_features)):
        if feature in features_to_skip:
            continue
        dataset_slice = dataset[:, feature].to(device)
        q1, q3 = get_quantiles(dataset_slice, [0.25, 0.75])

        iqr = q3 - q1
        f1 = q1 - 1.5*iqr
        f2 = q3 + 1.5*iqr
        e1 = q1 - 3*iqr
        e2 = q3 + 3*iqr
        extreme_outliers_indices = torch.logical_or(dataset_slice <= e1, dataset_slice >= e2)
        left_outliers_indices = torch.logical_and(dataset_slice <= f1, dataset_slice >= e1)
        right_outliers_indices = torch.logical_and(dataset_slice <= e2, dataset_slice >= f2)
        extreme_outliers_scores = dataset_slice[extreme_outliers_indices]
        left_outliers_scores = dataset_slice[left_outliers_indices]
        right_outliers_scores = dataset_slice[right_outliers_indices]
        extreme_outliers_scores[:] = 1
        left_outliers_scores = (left_outliers_scores - e1)/(f1 - e1)
        right_outliers_scores = (right_outliers_scores - f2)/(e2 - f2)
        outliers_scores_sum = extreme_outliers_scores.sum() + left_outliers_scores.sum() + right_outliers_scores.sum()
        outliers_scores_len = extreme_outliers_scores.shape[0] + left_outliers_scores.shape[0] + right_outliers_scores.shape[0]
        outlier_subscore = outliers_scores_sum/outliers_scores_len
        if outlier_subscore.isnan().item():
            outlier_subscore = torch.tensor(0)
        outlier_subscores.append(outlier_subscore)
        
    outlier_subscores = torch.tensor(outlier_subscores).detach().cpu()
    outlier_score = torch.mean(outlier_subscores)
    return outlier_score, outlier_subscores

def io_score(features, labels, features_to_skip=[], device='cpu'):
    n_features = features.shape[1]
    n_labels = labels.shape[1]
    mean_deltas = torch.zeros(n_features - len(features_to_skip), n_labels).to(device)

    for feature in tqdm(range(n_features)):
        slice = torch.hstack((features[:, feature].unsqueeze(1), labels))
        sorted_indices = slice[:, 0].sort()[1]
        slice = slice[sorted_indices]
        for label in range(n_labels):
            label_slice = slice[:, label + 1]
            feature_slice = slice[:, 0]
            delta_label = label_slice[1:] - label_slice[:-1]
            delta_feature = feature_slice[1:] - feature_slice[:-1]
            delta = delta_label/delta_feature
            delta = delta[~torch.isnan(delta) & ~torch.isinf(delta)]
            delta = torch.arctan(torch.abs(delta))/(math.pi/2)
            delta = delta[~torch.isnan(delta) & ~torch.isinf(delta)]
            mean_delta = torch.mean(delta)
            mean_deltas[feature][label] = mean_delta
        
    io_score = mean_deltas[~torch.isnan(mean_deltas) & ~torch.isinf(mean_deltas)].mean()
    return io_score, mean_deltas

if __name__ == "__main__":
    data = torch.rand((30, 70))
    plot_heatmap(data, "x", "y", "test.png")
    data2 = torch.rand((2000))
    plot_scores(data2, "x", "y", "test2.png")