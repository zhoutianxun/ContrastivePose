# utility packages
import os
import numpy as np
from load_data import get_dataset, encode_data
from models import Contrastive_model, FineTuneModel
from trainer import train_model, train_finetune_model

#plotting
import plotly.express as px
import plotly.io as pio
from umap import UMAP

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# pytorch
import torch
from torch.utils.data import TensorDataset, DataLoader

# Mode
train = False
test = "contrastive"  # contrastive, finetune, from_scratch
experiment = 1

# Constants
width_original = 1024
height_original = 570
undersample = "random"
seq_length = 30
classes = 4
batch_size = 64
original_features = 56  # 28x2
extracted_features = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pio.renderers.default = 'browser' #browser

behavior_class = {0: "attack",
                  1: "investigation",
                  2: "mount",
                  3: "other"}

"""
Load Data
"""
trainset_path = os.path.join(os.path.getcwd(), "datasets", "trainset")
X, X_handcraft, _, Y = get_dataset(trainset_path,
                                   seq_length,
                                   width_original=width_original,
                                   height_original=height_original,
                                   undersample=undersample,
                                   sample_amount=400000)

# prepare train test split
test_split = 0.10
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_split, random_state=42, stratify=Y)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.50, random_state=42, stratify=y_test)

X_train_hc, X_test_hc, y_train_hc, y_test_hc = train_test_split(X_handcraft, Y, test_size=test_split, random_state=42,
                                                                stratify=Y)
X_valid_hc, X_test_hc, y_valid_hc, y_test_hc = train_test_split(X_test_hc, y_test_hc, test_size=0.50, random_state=42,
                                                                stratify=y_test_hc)

train_data = TensorDataset(X_train, y_train)
valid_data = TensorDataset(X_valid, y_valid)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

# clear memory
del X, X_handcraft, Y
del X_train, X_train_hc, X_test, X_test_hc, y_train, y_train_hc, y_test, y_test_hc

"""
Training of model
"""
if train:
    feature_extractor = Contrastive_model(seq_length, original_features, extracted_features)
    contrastive_model_path = os.path.join(os.getcwd(), "Models", f"contrast_v{experiment}.pt")
    lr = 0.002
    epochs = 60
    patience = 30

    print("Training model in progress...")
    train_model(feature_extractor, contrastive_model_path, train_loader, valid_loader, device, lr, epochs, patience, False)

"""
Fine tune model with validation set
"""
if train:
    basemodel = Contrastive_model(seq_length, original_features, extracted_features)
    basemodel.load_state_dict(torch.load(contrastive_model_path))
    fine_tune_model = FineTuneModel(basemodel, extracted_features, classes)
    fine_tune_model = fine_tune_model.to(device)
    finetune_model_path = os.path.join(os.getcwd(), "Models", f"finetune_v{experiment}.pt")
    lr = 0.001
    epochs = 60
    patience = 60

    print("Fine tuning model in progress...")
    train_finetune_model(fine_tune_model, finetune_model_path, valid_loader, test_loader, device, lr, epochs, patience)

"""
Training from scratch
"""
if train:
    from_scratch_model = FineTuneModel(Contrastive_model(seq_length, original_features, extracted_features),
                                       extracted_features, classes)
    from_scratch_model = from_scratch_model.to(device)
    from_scratch_model_path = os.path.join(os.getcwd(), "Models", f"train_from_scratch_v{experiment}.pt")
    lr = 0.001
    epochs = 100
    patience = 60

    print("Fine tuning model in progress...")
    train_finetune_model(from_scratch_model, from_scratch_model_path, valid_loader, test_loader, device, lr, epochs,
                         patience)

# clear memory
del train_data, train_loader, test_data, test_loader, valid_loader

"""
Test on new dataset
"""
testset_path = os.path.join(os.path.getcwd(), "datasets", "testset")
X_test, X_test_hc, X_test_tensordata, y_test = get_dataset(testset_path,
                                                           seq_length,
                                                           width_original=width_original,
                                                           height_original=height_original,
                                                           undersample=None,
                                                           sample_amount=50000)

assert test in ["contrastive", "finetune", "from_scratch"]
if test == "contrastive":
    test_model = Contrastive_model(seq_length, original_features, extracted_features)
    test_model_path = os.path.join(os.getcwd(), "Models", f"contrast_v{experiment}.pt")
elif test == "finetune":
    # load fine tuned model
    test_model = FineTuneModel(Contrastive_model(seq_length, original_features, extracted_features),
                               extracted_features, classes)
    test_model_path = os.path.join(os.getcwd(), "Models", f"finetune_v{experiment}.pt")
elif test == "from_scratch":
    # load model trained from scratch
    test_model = FineTuneModel(Contrastive_model(seq_length, original_features, extracted_features),
                               extracted_features, classes)
    test_model_path = os.path.join(os.getcwd(), "Models", f"train_from_scratch_v{experiment}.pt")

test_model.load_state_dict(torch.load(test_model_path))
X_train_encoded = encode_data(valid_data, test_model, device)
X_test_encoded = encode_data(X_test_tensordata, test_model, device)

# Train classifier model and check accuracy of classification
y_true = y_test.numpy()
train_size = len(X_valid)
test_size = len(X_test)

# no feature extraction
clf = RandomForestClassifier()
clf.fit(np.squeeze(X_valid.numpy()).reshape(train_size, -1), y_valid.numpy())
y_pred = clf.predict(np.squeeze(X_test.numpy()).reshape(test_size, -1))

print("No Feature Extraction:")
print(classification_report(y_true, y_pred, zero_division=0))

# handcrafted features
clf_hc = RandomForestClassifier()
clf_hc.fit(np.squeeze(X_valid_hc.numpy()).reshape(train_size, -1), y_valid.numpy())
y_pred_hc = clf_hc.predict(X_test_hc.numpy().reshape(test_size, -1))

print("Handcraft Features:")
print(classification_report(y_true, y_pred_hc, zero_division=0))

# learnt features
clf_enc = RandomForestClassifier()
#clf_enc.fit(np.concatenate((X_train_encoded, X_valid_hc.numpy()[:,:,49:].reshape(train_size, -1)), axis=1), y_valid.numpy())
#y_pred_enc = clf_enc.predict(np.concatenate((X_test_encoded, X_test_hc.numpy()[:,:,49:].reshape(test_size, -1)), axis=1))
clf_enc.fit(X_train_encoded, y_valid.numpy())
y_pred_enc = clf_enc.predict(X_test_encoded)

print(f"Learnt Features ({test}):")
print(classification_report(y_true, y_pred_enc, zero_division=0))


# visualize embeddings using UMAP
umap_2d = UMAP(n_components=2, init='random', random_state=42)
viz_results = umap_2d.fit_transform(X_test_encoded)
fig = px.scatter(viz_results, x=0, y=1, color=np.vectorize(behavior_class.__getitem__)(y_test.numpy().astype(int)),
                 labels={'0': 'UMAP 1', '1': 'UMAP 2'}, title="Learnt Feature Representation")
fig.show()

umap_2d = UMAP(n_components=2, init='random', random_state=42)
viz_results = umap_2d.fit_transform(np.squeeze(X_test_hc.numpy()).reshape(test_size, -1))
fig = px.scatter(viz_results, x=0, y=1, color=np.vectorize(behavior_class.__getitem__)(y_test.numpy().astype(int)),
                 labels={'0': 'UMAP 1', '1': 'UMAP 2'}, title="Handcrafted Feature Representation")
fig.show()

umap_2d = UMAP(n_components=2, init='random', random_state=42)
viz_results = umap_2d.fit_transform(np.squeeze(X_test.numpy()).reshape(test_size, -1))
fig = px.scatter(viz_results, x=0, y=1, color=np.vectorize(behavior_class.__getitem__)(y_test.numpy().astype(int)),
                 labels={'0': 'UMAP 1', '1': 'UMAP 2'}, title="Original Feature Representation")
fig.show()