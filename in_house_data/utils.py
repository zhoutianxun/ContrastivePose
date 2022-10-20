import numpy as np
import torch
from itertools import combinations


# Util functions
def distance(part_1, part_2):
    return np.sqrt((part_1[:, 0] - part_2[:, 0]) ** 2 + (part_1[:, 1] - part_2[:, 1]) ** 2)


def overlap(box_1, box_2):
    # each box = [left x, top y, width, height]
    left_x = np.maximum(box_1[:, 0], box_2[:, 0])
    top_y = np.maximum(box_1[:, 1], box_2[:, 1])

    right_x = np.minimum(box_1[:, 0] + box_1[:, 2], box_2[:, 0] + box_2[:, 2])
    bot_y = np.minimum(box_1[:, 1] + box_1[:, 3], box_2[:, 1] + box_2[:, 3])

    # compute the area of intersection rectangle
    intersect = np.maximum(np.zeros_like(right_x), right_x - left_x) * np.maximum(np.zeros_like(bot_y), bot_y - top_y)
    total_1 = box_1[:, 2] * box_1[:, 3]
    total_2 = box_2[:, 2] * box_2[:, 3]

    overlap_ratio = intersect / (total_1 + total_2 - intersect)
    # overlap_ratio[np.isnan(overlap_ratio)] = 0
    return overlap_ratio


def overlap_points_ver(box_1, box_2):
    # each box = [top-left x, y, top-right x, y, bot-left x, y, bot-right x, y
    left_x = torch.maximum(box_1[:, :, 0], box_2[:, :, 0])
    top_y = torch.maximum(box_1[:, :, 1], box_2[:, :, 1])
    right_x = torch.minimum(box_1[:, :, 6], box_2[:, :, 6])
    bot_y = torch.minimum(box_1[:, :, 7], box_2[:, :, 7])
    # compute the area of intersection rectangle
    intersect = torch.maximum(torch.zeros_like(right_x), right_x - left_x) * \
                torch.maximum(torch.zeros_like(bot_y), bot_y - top_y)
    total_1 = (box_1[:, :, 6] - box_1[:, :, 0]) * (box_1[:, :, 7] - box_1[:, :, 1])
    total_2 = (box_2[:, :, 6] - box_2[:, :, 0]) * (box_2[:, :, 7] - box_2[:, :, 1])

    overlap_ratio = intersect / (total_1 + total_2 - intersect)
    # overlap_ratio[np.isnan(overlap_ratio)] = 0
    return overlap_ratio


def generate_overlap_feature(X, n=4):
    comb = np.array(list(combinations([i for i in range(n*2)], 2)))
    comb = comb[np.where((comb[:, 1] > comb[:, 0]) & (comb[:, 1] > n-1) & (comb[:, 0] < n))]
    X_overlap = torch.zeros((X.shape[0], X.shape[1], len(comb)))
    for i, c in enumerate(comb):
        part_1 = c[0]
        part_2 = c[1]
        X_overlap[:, :, i] = overlap_points_ver(X[:, :, part_1 * 8: (part_1 + 1) * 8], X[:, :, part_2 * 8: (part_2 + 1) * 8])
    return X_overlap


def sliding_windows(X, Y, seq_length):
    x = np.zeros((len(X) - seq_length, seq_length, X.shape[1]))
    y = np.zeros(len(Y) - seq_length)

    for i in range(len(X) - seq_length):
        _x = X[i:(i + seq_length)]
        x[i] = _x
        y[i] = Y[i + seq_length - 1]

    return x, y


def get_centerpoint(corners):
    # type np array, shape of input: n(n, t, 8), columns: x, y positions of 4 corner points
    min_x = np.min(corners[:, :, ::2], axis=2)
    max_x = np.max(corners[:, :, ::2], axis=2)
    min_y = np.min(corners[:, :, 1::2], axis=2)
    max_y = np.max(corners[:, :, 1::2], axis=2)
    return np.stack(((min_x + max_x) / 2, (min_y + max_y) / 2), axis=2)


def redraw(corners, center_point, old_width, old_height):
    min_x = np.min(corners[:, :, ::2], axis=2)
    max_x = np.max(corners[:, :, ::2], axis=2)
    min_y = np.min(corners[:, :, 1::2], axis=2)
    max_y = np.max(corners[:, :, 1::2], axis=2)
    height2width = (max_y - min_y) / (max_x - min_x + 1e-9)
    area = old_width * old_height
    new_width = np.clip(np.sqrt(area / height2width), 1e-9, 1)
    new_height = area / new_width
    return np.stack(((center_point[:, :, 0] - new_width / 2),
                     (center_point[:, :, 1] - new_height / 2),
                     (center_point[:, :, 0] + new_width / 2),
                     (center_point[:, :, 1] - new_height / 2),
                     (center_point[:, :, 0] - new_width / 2),
                     (center_point[:, :, 1] + new_height / 2),
                     (center_point[:, :, 0] + new_width / 2),
                     (center_point[:, :, 1] + new_height / 2)), axis=2)


'''
def augment_data(x_in, mirror=True, shift=True, rotate=True, velocity=True):
    if velocity:
        features = x_in.shape[-1] // 2
    else:
        features = x_in.shape[-1]

    if type(x_in) == torch.Tensor:
        x = x_in.cpu().numpy().copy()
        x = np.squeeze(x, 1)
    else:
        x = x_in.copy()

    if mirror:
        # flip x
        chosen = np.random.choice(len(x), len(x) // 2, replace=False)
        x[chosen, :, 0:features:2] = 1 - x[chosen, :, 0:features:2]
        x[chosen, :, features::2] = -x[chosen, :, features::2]

        # flip y
        chosen = np.random.choice(len(x), len(x) // 2, replace=False)
        x[chosen, :, 1:features:2] = 1 - x[chosen, :, 1:features:2]
        x[chosen, :, features + 1::2] = -x[chosen, :, features + 1::2]

    if rotate:
        # rotate randomly
        theta = np.random.random(size=(len(x))) * np.pi * 2
        cos = np.cos(theta)
        sin = np.sin(theta)
        rotation_matrix = np.transpose(np.array([[cos, -sin], [sin, cos]]), [2, 1, 0])
        bs, seqlength, feats = x[:, :, 0:features:2].shape
        cg_x = np.sum(x[:, :, 0:features:2]) / (bs * seqlength * feats)
        cg_y = np.sum(x[:, :, 1:features:2]) / (bs * seqlength * feats)
        rotated = np.matmul(rotation_matrix, np.stack(
            (x[:, :, 0:features:2].reshape(bs, -1) - cg_x, x[:, :, 1:features:2].reshape(bs, -1) - cg_y), axis=1))
        rotated_v = np.matmul(rotation_matrix,
                              np.stack((x[:, :, features::2].reshape(bs, -1), x[:, :, features + 1::2].reshape(bs, -1)),
                                       axis=1))
        xp = rotated[:, 0]
        yp = rotated[:, 1]
        xp = xp + cg_x
        yp = yp + cg_y
        xvp = rotated_v[:, 0]
        yvp = rotated_v[:, 1]
        x[:, :, 0:features:2] = xp.reshape(bs, seqlength, feats)
        x[:, :, 1:features:2] = yp.reshape(bs, seqlength, feats)
        x[:, :, features::2] = xvp.reshape(bs, seqlength, feats)
        x[:, :, features + 1::2] = yvp.reshape(bs, seqlength, feats)
        shift = True

    if shift:
        max_x = np.max(x[:, :, 0:features:2], axis=(1, 2))
        min_x = np.min(x[:, :, 0:features:2], axis=(1, 2))
        max_y = np.max(x[:, :, 1:features:2], axis=(1, 2))
        min_y = np.min(x[:, :, 1:features:2], axis=(1, 2))

        x_shift_range = 1 - max_x + min_x
        y_shift_range = 1 - max_y + min_y

        x_shift = np.random.random(size=(len(x))) * x_shift_range - min_x
        y_shift = np.random.random(size=(len(x))) * y_shift_range - min_y
        x[:, :, 0:features:2] = x[:, :, 0:features:2] + x_shift.reshape(-1, 1, 1)
        x[:, :, 1:features:2] = x[:, :, 1:features:2] + y_shift.reshape(-1, 1, 1)

    if type(x_in) == torch.Tensor:
        return torch.unsqueeze(torch.from_numpy(x), 1)
    else:
        return x
'''


def augment_data(x_in, mirror=True, shift=True, rotate=True, velocity=True):
    if velocity:
        features = x_in.shape[-1] // 2
    else:
        features = x_in.shape[-1]

    if type(x_in) == torch.Tensor:
        x = x_in.cpu().numpy().copy()
        x = np.squeeze(x, 1)
    else:
        x = x_in.copy()

    if mirror:
        # flip x
        chosen = np.random.choice(len(x), len(x) // 2, replace=False)
        x[chosen, :, 0:features:2] = 1 - x[chosen, :, 0:features:2]
        x[chosen, :, features::2] = -x[chosen, :, features::2]

        # flip y
        chosen = np.random.choice(len(x), len(x) // 2, replace=False)
        x[chosen, :, 1:features:2] = 1 - x[chosen, :, 1:features:2]
        x[chosen, :, features + 1::2] = -x[chosen, :, features + 1::2]

    if rotate:
        # rotate randomly
        theta = np.random.random(size=(len(x))) * np.pi * 2
        cos = np.cos(theta)
        sin = np.sin(theta)
        rotation_matrix = np.transpose(np.array([[cos, -sin], [sin, cos]]), [2, 1, 0])
        bs, seqlength, feats = x[:, :, 0:features:2].shape
        cg_x = np.sum(x[:, :, 0:features:2]) / (bs * seqlength * feats)
        cg_y = np.sum(x[:, :, 1:features:2]) / (bs * seqlength * feats)
        rotated = np.matmul(rotation_matrix, np.stack(
            (x[:, :, 0:features:2].reshape(bs, -1) - cg_x, x[:, :, 1:features:2].reshape(bs, -1) - cg_y), axis=1))
        rotated_v = np.matmul(rotation_matrix,
                              np.stack((x[:, :, features::2].reshape(bs, -1), x[:, :, features + 1::2].reshape(bs, -1)),
                                       axis=1))
        xp = rotated[:, 0]
        yp = rotated[:, 1]
        xp = xp + cg_x
        yp = yp + cg_y
        xvp = rotated_v[:, 0]
        yvp = rotated_v[:, 1]
        x[:, :, 0:features:2] = xp.reshape(bs, seqlength, feats)
        x[:, :, 1:features:2] = yp.reshape(bs, seqlength, feats)
        x[:, :, features::2] = xvp.reshape(bs, seqlength, feats)
        x[:, :, features + 1::2] = yvp.reshape(bs, seqlength, feats)
        shift = True

    if shift:
        max_x = np.max(x[:, :, 0:features:2], axis=(1, 2))
        min_x = np.min(x[:, :, 0:features:2], axis=(1, 2))
        max_y = np.max(x[:, :, 1:features:2], axis=(1, 2))
        min_y = np.min(x[:, :, 1:features:2], axis=(1, 2))

        x_shift_range = 1 - max_x + min_x
        y_shift_range = 1 - max_y + min_y

        x_shift = np.random.random(size=(len(x))) * x_shift_range - min_x
        y_shift = np.random.random(size=(len(x))) * y_shift_range - min_y
        x[:, :, 0:features:2] = x[:, :, 0:features:2] + x_shift.reshape(-1, 1, 1)
        x[:, :, 1:features:2] = x[:, :, 1:features:2] + y_shift.reshape(-1, 1, 1)

    if rotate:
        center_pts = np.zeros((len(x_in), seqlength, features // 4))
        for i in range(features // 8):
            center_pts[:, :, i * 2:(i + 1) * 2] = get_centerpoint(x[:, :, i * 8: (i + 1) * 8])

        redrawn_pts = np.zeros((len(x_in), seqlength, features))
        for i in range(center_pts.shape[-1] // 2):
            old_width = x_in[:, 0, :, i * 8 + 2] - x_in[:, 0, :, i * 8]
            old_height = x_in[:, 0, :, i * 8 + 5] - x_in[:, 0, :, i * 8 + 1]
            center_point = center_pts[:, :, i * 2: (i + 1) * 2]
            points = x[:, :, i * 8: (i + 1) * 8]
            redrawn_pts[:, :, i * 8: (i + 1) * 8] = redraw(points, center_point, old_width.cpu().numpy(),
                                                           old_height.cpu().numpy())
            x[:, :, :features] = redrawn_pts

    if type(x_in) == torch.Tensor:
        return torch.unsqueeze(torch.from_numpy(x), 1)
    else:
        return x
