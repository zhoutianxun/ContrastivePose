import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
from itertools import combinations
from utils import distance, overlap, sliding_windows, generate_overlap_feature
from imblearn.under_sampling import RandomUnderSampler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def box_to_points(box):
    # type np array, shape of input: (n, 4), columns: top left x, top left y, width, height
    # returns corner points in order: top-left, top-right, bot-left, bot-right
    pts = np.zeros((box.shape[0], box.shape[1]*2))
    pts[:, :2] = box[:, :2]
    pts[:, 2] = box[:, 0] + box[:, 2]
    pts[:, 3] = box[:, 1]
    pts[:, 4] = box[:, 0]
    pts[:, 5] = box[:, 1] + box[:, 3]
    pts[:, 6:] = box[:, :2] + box[:, 2:]
    return pts


def box_to_centerpoint(box):
    # type np array, shape of input: (n, 4), columns: top left x, top left y, width, height
    pts = np.zeros((box.shape[0], box.shape[1]//2))
    pts[:, 0] = box[:, 0] + box[:, 2]/2
    pts[:, 1] = box[:, 1] + box[:, 3]/2
    return pts


def get_dataset(dataset_path,
                seq_length,
                width_original=1920,
                height_original=1080,
                impute=True,
                undersample=None,
                sample_amount=10000):
    # Load data
    if type(dataset_path) == list:
        df = None
        for f in dataset_path:
            df = pd.concat([df, pd.read_csv(f)], ignore_index=True)
        df = df.rename(columns={"Unnamed: 0": "frame"})
    else:
        df = pd.read_csv(dataset_path)
    X = df.iloc[:, 1:-1]
    Y_true = df.iloc[:, -1]
    frame_no = df.iloc[:, 0]

    # Normalize data
    X.iloc[:, ::2] = X.iloc[:, ::2] / width_original
    X.iloc[:, 1::2] = X.iloc[:, 1::2] / height_original

    # impute data
    X_np = X.to_numpy()
    del X
    if impute:
        imputer = IterativeImputer(random_state=42)
        X_np = imputer.fit_transform(X_np)

    # Compute box overlap
    comb = np.array(list(combinations([i for i in range(8)], 2)))
    comb = comb[np.where((comb[:, 1] > comb[:, 0]) & (comb[:, 1] > 3) & (comb[:, 0] < 4))]
    X_overlap = np.zeros((X_np.shape[0], len(comb)))
    for i, c in enumerate(comb):
        part_1 = c[0]
        part_2 = c[1]
        X_overlap[:, i] = overlap(X_np[:, part_1 * 4: part_1 * 4 + 4], X_np[:, part_2 * 4: part_2 * 4 + 4])

    # Transform from bounding box to corner points and centerpoints
    X_pts = np.zeros((X_np.shape[0], X_np.shape[1]*2))
    X_centerpts = np.zeros((X_np.shape[0], X_np.shape[1]//2))
    for i in range(X_np.shape[1]//4):
        X_pts[:, i*8:(i+1)*8] = box_to_points(X_np[:, i*4:(i+1)*4])
        X_centerpts[:, i * 2:(i + 1) * 2] = box_to_centerpoint(X_np[:, i * 4:(i + 1) * 4])

    # Calculate velocity features
    X_v = X_pts[1:, :] - X_pts[:-1, :]
    X_v = np.concatenate((np.zeros((1, X_v.shape[1])), X_v), axis=0)

    # Compute pairwise distance
    comb = np.array(list(combinations([i for i in range(X_pts.shape[1]//2)], 2))) #8
    comb = comb[np.where((comb[:, 1] > comb[:, 0]) & (comb[:, 1] > 15) & (comb[:, 0] < 16))]
    X_dist = np.zeros((X_np.shape[0], len(comb)))

    for i, c in enumerate(comb):
        part_1 = c[0]
        part_2 = c[1]
        #X_dist[:, i] = distance(X_centerpts[:, part_1 * 2: part_1 * 2 + 2], X_centerpts[:, part_2 * 2: part_2 * 2 + 2])
        X_dist[:, i] = distance(X_pts[:, part_1 * 2: part_1 * 2 + 2], X_pts[:, part_2 * 2: part_2 * 2 + 2])

    # Create feature sets
    X_np = np.concatenate((X_pts, X_v), axis=1)
    #X_handcraft = np.concatenate((X_overlap, X_centerpts, X_v), axis=1)
    X_handcraft = np.concatenate((X_overlap, X_pts, X_dist, X_v), axis=1)

    # Form sliding windows
    X, Y = sliding_windows(X_np, Y_true.to_numpy().reshape(-1, 1), seq_length)
    X_handcraft, _ = sliding_windows(X_handcraft, Y_true.to_numpy().reshape(-1, 1), seq_length)

    X = X[frame_no.to_numpy()[seq_length:] >= seq_length]
    Y = Y[frame_no.to_numpy()[seq_length:] >= seq_length]
    X_handcraft = X_handcraft[frame_no.to_numpy()[seq_length:] >= seq_length]
    Y = Y.reshape(-1)

    if undersample == "equalize":
        sampler = RandomUnderSampler(random_state=42)
        n, s, f = X.shape
        fhc = X_handcraft.shape[-1]
        X_temp, Y = sampler.fit_resample(np.concatenate((X.reshape(n, -1), X_handcraft.reshape(n, -1)), axis=1), Y)
        X = X_temp[:, :s * f].reshape(-1, s, f)
        X_handcraft = X_temp[:, s * f:].reshape(-1, s, fhc)

    elif undersample == "random":
        # Undersample to fixed number
        keep = np.random.choice(len(Y), sample_amount)
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
        x = torch.unsqueeze(sample[0], 0)
        model.eval()
        with torch.no_grad():
            x = torch.cat((generate_overlap_feature(x[:, 0, :, :64]), x[:, 0, :, :]), dim=2).to(device)
            encoded_x, _ = model(x)
        # Append to list
        encoded_x = encoded_x.squeeze().cpu().numpy()
        encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_x)}
        encoded_samples.append(encoded_sample)
    encoded_samples = pd.DataFrame(encoded_samples)
    return encoded_samples.to_numpy()