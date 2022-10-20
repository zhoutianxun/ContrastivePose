import numpy as np
import torch


# Util functions
def distance(part_1, part_2):
    return np.sqrt((part_1[:, 0] - part_2[:, 0]) ** 2 + (part_1[:, 1] - part_2[:, 1]) ** 2)


def sliding_windows(X, Y, seq_length):
    x = np.zeros((len(X) - seq_length, seq_length, X.shape[1]))
    y = np.zeros(len(Y) - seq_length)

    for i in range(len(X) - seq_length):
        _x = X[i:(i + seq_length)]
        x[i] = _x
        y[i] = Y[i + seq_length - 1]

    return x, y


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