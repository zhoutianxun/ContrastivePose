import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
from itertools import combinations
from utils import distance, sliding_windows
from imblearn.under_sampling import RandomUnderSampler


def get_dataset(dataset_path,
                seq_length,
                width_original=1024,
                height_original=570,
                undersample=None,
                sample_amount=10000,
                random_seed=None):
    # Load data
    files = list(os.listdir(dataset_path))
    df = None
    for f in files:
        path = os.path.join(dataset_path, f)
        df = pd.concat([df, pd.read_csv(path)], ignore_index=True)
    df = df.rename(columns={"Unnamed: 0": "frame"})
    df = df.astype({'frame': 'int32', 'label':'int32'})

    # Normalize data
    X = df.iloc[:, 1:-1]
    Y_true = df.iloc[:, -1]
    frame_no = df.iloc[:, 0]
    del df
    X.iloc[:, ::2] = X.iloc[:, ::2] / width_original
    X.iloc[:, 1::2] = X.iloc[:, 1::2] / height_original

    # Concatenate velocity features
    X_v = X.copy().iloc[1:, :].reset_index(drop=True) - X.copy().iloc[:-1, :]
    X_v = X_v.set_index(np.arange(1, len(X_v) + 1))
    X_v = X_v.add_prefix('v_')
    X_v = pd.concat([X, X_v], axis=1).fillna(0)

    # Form handcrafted features
    comb = np.array(list(combinations([i for i in range(14)], 2)))
    comb = comb[np.where((comb[:, 1] > comb[:, 0]) & (comb[:, 1] > 6) & (comb[:, 0] < 7))]
    X_np = X.to_numpy()
    del X
    X_handcraft = np.zeros((X_np.shape[0], len(comb)))
    for i, c in enumerate(comb):
        part_1 = c[0]
        part_2 = c[1]
        X_handcraft[:, i] = distance(X_np[:, part_1 * 2: part_1 * 2 + 2], X_np[:, part_2 * 2: part_2 * 2 + 2])
    X_handcraft = np.concatenate((X_handcraft, X_v.to_numpy()), axis=1) #X_v.to_numpy()[:, 28:]
    X_np = np.concatenate((X_np, X_v.to_numpy()[:, 28:]), axis=1)
    del X_v

    # Form sliding windows
    X, Y = sliding_windows(X_np, Y_true.to_numpy().reshape(-1, 1), seq_length)
    X_handcraft, _ = sliding_windows(X_handcraft, Y_true.to_numpy().reshape(-1, 1), seq_length)

    X = X[frame_no.to_numpy()[seq_length:] >= seq_length]
    Y = Y[frame_no.to_numpy()[seq_length:] >= seq_length]
    X_handcraft = X_handcraft[frame_no.to_numpy()[seq_length:] >= seq_length]
    Y = Y.reshape(-1)

    if undersample == "equalize":
        sampler = RandomUnderSampler(random_state=random_seed)
        n, s, f = X.shape
        fhc = X_handcraft.shape[-1]
        X_temp, Y = sampler.fit_resample(np.concatenate((X.reshape(n, -1), X_handcraft.reshape(n, -1)), axis=1), Y)
        X = X_temp[:, :s * f].reshape(-1, s, f)
        X_handcraft = X_temp[:, s * f:].reshape(-1, s, fhc)

    elif undersample == "random":
        # Undersample to fixed number
        rng = np.random.default_rng(random_seed)
        keep = rng.choice(len(Y), sample_amount, replace=False)
        #keep = np.random.choice(len(Y), sample_amount)
        X = X[keep]
        Y = Y[keep]
        X_handcraft = X_handcraft[keep]

    # form data set
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    X_handcraft = torch.from_numpy(X_handcraft).float()
    X = torch.unsqueeze(X, 1)
    X_tensordata = TensorDataset(X, Y)

    return X, X_handcraft, X_tensordata, Y


def encode_data(tensordata, model, device):
    model = model.to(device)
    encoded_samples = []
    for sample in tqdm(tensordata):
        x = sample[0].to(device)
        model.eval()
        with torch.no_grad():
            encoded_x, _ = model(torch.unsqueeze(x, 0))
        # Append to list
        encoded_x = encoded_x.squeeze().cpu().numpy()
        encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_x)}
        encoded_samples.append(encoded_sample)
    encoded_samples = pd.DataFrame(encoded_samples)
    return encoded_samples.to_numpy()

def predict_data(tensordata, model, device):
    model = model.to(device)
    preds = []
    for sample in tqdm(tensordata):
        x = sample[0].to(device)
        model.eval()
        with torch.no_grad():
            _, pred = model(torch.unsqueeze(x, 0))
        # Append to list
        pred = pred.squeeze().cpu().numpy()
        preds.append(pred)
    return np.array(preds)