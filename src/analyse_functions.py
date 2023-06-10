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

def remove_outliers_with_quantiles(x, q):
    right_q = q + (1 - q)/2
    left_q = (1 - q)/2
    right_quantile = torch.quantile(x, right_q)
    left_quantile = torch.quantile(x, left_q)
    outlier_indices = torch.logical_or(x < left_quantile, x > right_quantile)
    x = x[~outlier_indices]
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
    ax = sns.heatmap(x.numpy(), annot=annot)
    sns.set(style='darkgrid')
    ax.set(xlabel=x_label, ylabel=y_label)
    plt.tight_layout()
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()

def simb_score(dataset, n_bins=10000, features_to_skip=[], q=0.99, device='cpu'):
    n_features = dataset.shape[1]
    js_divergences = []
    uniform_distribution = torch.ones(n_bins).to(device) / n_bins

    for feature in tqdm(range(n_features)):
        if feature in features_to_skip:
            continue
        dataset_slice = dataset[:, feature]
        dataset_slice = remove_outliers_with_quantiles(dataset_slice, q)
        min, max = torch.min(dataset_slice), torch.max(dataset_slice)
        distribution = create_distribution(dataset_slice, min, max, n_bins=n_bins)
        js_divergence= compute_js_divergence(distribution, uniform_distribution)
        js_divergences.append(js_divergence)
    
    js_divergences = torch.tensor(js_divergences).to(device)
    simb_score = torch.mean(js_divergences)
    return simb_score, js_divergences

def stood_score(train_set, validation_set, test_set, n_bins=10000, features_to_skip=[], q=0.99, device='cpu'):
    n_features = train_set.shape[1]
    js_divergences_validation, js_divergences_test = [], []

    for feature in tqdm(range(n_features)):
        if feature in features_to_skip:
            continue
        train_slice = train_set[:, feature]
        validation_slice = validation_set[:, feature]
        test_slice = test_set[:, feature]

        train_slice = remove_outliers_with_quantiles(train_slice, q)
        validation_slice = remove_outliers_with_quantiles(validation_slice, q)
        test_slice = remove_outliers_with_quantiles(test_slice ,q)

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
        
    io_score = mean_deltas.mean()
    return io_score, mean_deltas

if __name__ == "__main__":
    data = torch.rand((30, 70))
    plot_heatmap(data, "x", "y", "test.png")
    data2 = torch.rand((2000))
    plot_scores(data2, "x", "y", "test2.png")