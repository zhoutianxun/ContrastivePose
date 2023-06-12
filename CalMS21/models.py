# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_feature_extractor(nn.Module):
    def __init__(self, seq_length, original_features, extracted_features):
        super(MLP_feature_extractor, self).__init__()
        self.linear1 = nn.Linear(in_features=seq_length * original_features, out_features=extracted_features * 4)
        self.linear2 = nn.Linear(in_features=extracted_features * 4, out_features=extracted_features)

    def forward(self, x):
        x = self.linear1(x.view(x.shape[0], -1))
        x = nn.ReLU(inplace=True)(x)
        x = self.linear2(x)
        return x
    

class TSCNN_feature_extractor(nn.Module):
    def __init__(self, seq_length, original_features, extracted_features):
        super(TSCNN_feature_extractor, self).__init__()
        self.spatial_conv1 = nn.Conv1d(original_features, extracted_features*4, 1)
        self.spatial_conv2 = nn.Conv1d(extracted_features*4, extracted_features, 1)
        if (seq_length+1)%2 == 0:
            w1 = (seq_length+1)/2
            w2 = w1
        else:
            w1 = (seq_length+1)//2
            w2 = seq_length+1-w1
        self.temporal_conv1 = nn.Conv1d(extracted_features, extracted_features, w1)
        self.bn1 = nn.BatchNorm1d(extracted_features)
        self.temporal_conv2 = nn.Conv1d(extracted_features, extracted_features, w2)

    def forward(self, x):
        # x.shape = (batch_size, seq_length, original_features)
        x = x.permute(0, 2, 1)
        x = nn.ReLU()(self.spatial_conv1(x))
        x = nn.ReLU()(self.spatial_conv2(x))
        x = self.temporal_conv1(x)
        x = nn.ReLU()(self.bn1(x))
        x = self.temporal_conv2(x).view(x.shape[0], -1)
        return x


class Projection_head(nn.Module):
    def __init__(self, extracted_features):
        super(Projection_head, self).__init__()
        self.extracted_features = extracted_features
        self.linear1 = nn.Linear(in_features=extracted_features, out_features=extracted_features * 2)
        self.linear2 = nn.Linear(in_features=extracted_features * 2, out_features=extracted_features * 4)

    def forward(self, x):
        x = nn.ReLU(inplace=True)(x)
        x = self.linear1(x.view(-1, self.extracted_features))
        x = nn.ReLU(inplace=True)(x)
        x = self.linear2(x)
        return x


class Contrastive_model(nn.Module):
    def __init__(self, seq_length, original_features, extracted_features):
        super(Contrastive_model, self).__init__()
        self.feature_extractor = MLP_feature_extractor(seq_length, original_features, extracted_features)
        self.projection = Projection_head(extracted_features)

    def forward(self, x):
        embedding = self.feature_extractor(x)
        projection = self.projection(embedding)
        return embedding, projection


class FineTuneModel(nn.Module):
    def __init__(self, base_model, extracted_features, classes):
        super(FineTuneModel, self).__init__()
        self.base_model = base_model
        self.linear = nn.Linear(extracted_features, classes)

    def forward(self, x):
        embedding, _ = self.base_model(x)
        pred = self.linear(nn.ReLU()(embedding))
        return embedding, pred


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class BarlowTwinLoss(nn.Module):
    def __init__(self, batch_size, dimension, off_diag_weight=5e-3):
        super().__init__()
        self.batch_size = batch_size
        self.dim = dimension
        self.off_diag_weight = off_diag_weight
        self.register_buffer("eye", torch.eye(dimension))

    def forward(self, emb_i, emb_j):
        z_i = (emb_i-torch.mean(emb_i, 0))/(torch.std(emb_i, 0) + 10e-9)
        z_j = (emb_j-torch.mean(emb_j, 0))/(torch.std(emb_j, 0) + 10e-9)
        c = torch.mm(z_i.T, z_j)/self.batch_size
        c_diff = (c - self.eye).pow(2)
        c_diff[~self.eye.bool()] = c_diff[~self.eye.bool()] * self.off_diag_weight
        loss = torch.sum(c_diff)
        return loss