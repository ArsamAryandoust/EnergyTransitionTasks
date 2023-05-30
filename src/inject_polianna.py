
import torch
import torch.nn as nn
from torchtext.vocab import vocab, Vocab
from collections import Counter
from pathlib import Path
import numpy as np
import sys
import os
home_dir = Path(os.path.expanduser('~'))
sys.path.append(str(home_dir / "selberai"))
from selberai.data.load_data import load

EMBEDDING_SIZE = 50

def create_vocab_pa(add: dict[str]):
    """
    Load the full text and create a vocabulary object
    """
    max_len = 0
    counter = Counter()
    for id in add["x_st"].keys():
        data = add["x_st"][str(id)]
        tokenized_text = [data[key]["text"] for key in data.keys()]
        max_len = max(max_len, len(tokenized_text))
        counter.update(tokenized_text)
    return vocab(counter, specials=["<pad>"]), max_len

def calculate_embeddings(embedding_module: nn.Module, X_split: np.ndarray, add: np.ndarray, vocab: Vocab) -> torch.Tensor:
    embeddings = []
    for sample in X_split:
        id = sample[-1]
        data = add["x_st"][str(id)]
        tokenized_text = [data[key]["text"] for key in data.keys()]
        sample_embeddings = embedding_module(torch.tensor(vocab(tokenized_text))).mean(dim=0)
        embeddings.append(sample_embeddings)
    return torch.stack(embeddings)

def calculate_bag_of_words(vocab: Vocab, X_split: np.ndarray, add: np.ndarray):
    counts = []
    for sample in X_split:
        id = sample[-1]
        data = add["x_st"][str(id)]
        tokenized_text = [data[key]["text"] for key in data.keys()]
        counter = Counter(tokenized_text)
        sample_counts = torch.zeros(len(vocab))
        for word, count in counter.items():
            sample_counts[vocab([word])[0]] = count
        counts.append(sample_counts)

    return torch.stack(counts)

def calculate_sequence_of_words(vocab: Vocab, X_split: np.ndarray, add: np.ndarray, max_len: int):
    sequences = []
    for sample in X_split:
        id = sample[-1]
        data = add["x_st"][str(id)]
        tokenized_text = [data[key]["text"] for key in data.keys()]
        sample_sequence = vocab(tokenized_text)
        sample_sequence = nn.functional.pad(torch.tensor(sample_sequence), (0, max_len - len(sample_sequence)))
        sequences.append(sample_sequence)

    return torch.stack(sequences)



def inject_additional(split: torch.Tensor, data_to_inject: torch.Tensor):
    split = split[:, :-1]
    split = torch.cat([split, data_to_inject], dim=1)
    return split

def inject_pa(dataset, subtask: str, glove_path: str| Path, mode: str = "embeddings"):
    """
    Injects the additional data into the datasets for the tabular data format.

    For the 'article_level' subtask it 
    """
    vocab, max_sequence_len = create_vocab_pa(dataset.add)
    print("len vocab:", len(vocab))

    X_train, Y_train = dataset.train
    X_val, Y_val = dataset.val
    X_test, Y_test = dataset.test
    
    if mode == "embeddings":
        embeddings_module = get_glove_embeddings_pa(glove_path, vocab)

        train_embeddings = calculate_embeddings(embeddings_module, X_train, dataset.add, vocab)
        X_train = inject_additional(torch.from_numpy(X_train), train_embeddings).numpy()
        val_embeddings = calculate_embeddings(embeddings_module, X_val, dataset.add, vocab)
        X_val = inject_additional(torch.from_numpy(X_val), val_embeddings).numpy()
        test_embeddings = calculate_embeddings(embeddings_module, X_test, dataset.add, vocab)
        X_test = inject_additional(torch.from_numpy(X_test), test_embeddings).numpy()
        
    elif mode == "bow":
        train_counts = calculate_bag_of_words(vocab, X_train, dataset.add)
        X_train = inject_additional(torch.from_numpy(X_train), train_counts).numpy()
        val_counts = calculate_bag_of_words(vocab, X_val, dataset.add)
        X_val = inject_additional(torch.from_numpy(X_val), val_counts).numpy()
        test_counts = calculate_bag_of_words(vocab, X_test, dataset.add)
        X_test = inject_additional(torch.from_numpy(X_test), test_counts).numpy()

    elif mode == "sow":
        train_sequences = calculate_sequence_of_words(vocab, X_train, dataset.add, max_sequence_len)
        X_train = inject_additional(torch.from_numpy(X_train), train_sequences).numpy()
        val_sequences = calculate_sequence_of_words(vocab, X_val, dataset.add, max_sequence_len)
        X_val = inject_additional(torch.from_numpy(X_val), val_sequences).numpy()
        test_sequences = calculate_sequence_of_words(vocab, X_test, dataset.add, max_sequence_len)
        X_test = inject_additional(torch.from_numpy(X_test), test_sequences).numpy()
    else:
        raise ValueError(f"Unrecognized mode: {mode}")

    if subtask == "article_level":
        Y_train = np.argmax(Y_train, 1, keepdims=False).flatten()
        Y_val = np.argmax(Y_val, 1, keepdims=False).flatten()
        Y_test = np.argmax(Y_test, 1, keepdims=False).flatten()

    dataset.train = X_train, Y_train.reshape(-1, 1)
    dataset.val = X_val, Y_val.reshape(-1, 1)
    dataset.test = X_test, Y_test.reshape(-1, 1)

    print("Successfully injected dataset!")

    return dataset
  

def get_glove_embeddings_pa(path, vocabulary):
    embeddings_index = dict()

    with open(path, "r") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Loaded %s word vectors.' % len(embeddings_index))

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((len(vocabulary), EMBEDDING_SIZE))

    for word, i in vocabulary.get_stoi().items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print("Built embedding matrix!")
    return nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix))

if __name__ == "__main__":
    name = "Polianna"
    subtask = "text_level"
    path_to_data = "/home/shared_cp/"

    dataset = load(name, subtask, sample_only=False, form="tabular", path_to_data=path_to_data)

    dataset = inject_pa(dataset, subtask, f"/home/shared_cp/Polianna/glove6B/glove.6B.{EMBEDDING_SIZE}d.txt", mode="sow")

    print(dataset.train[0].shape)
    print(dataset.train[0])