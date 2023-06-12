# utility packages
import os
import numpy as np
from load_data import get_dataset, encode_data, predict_data
from models import Contrastive_model, FineTuneModel
from trainer import train_model, train_finetune_model

#plotting
#import plotly.express as px
#import plotly.io as pio
#from umap import UMAP

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
test = "from_scratch"  # contrastive, finetune, from_scratch
experiment = 15

# Constants
width_original = 1024
height_original = 570
undersample = "random"
seq_length = 30
classes = 4
batch_size = 96
original_features = 56  # 28x2
extracted_features = 105 #64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_seed = experiment #42
#pio.renderers.default = 'browser' #browser
print(f"device engaged: {device}")

behavior_class = {0: "attack",
                  1: "investigation",
                  2: "mount",
                  3: "other"}

"""
Load Data
"""
trainset_path = os.path.join(os.getcwd(), "datasets", "trainset")
X, X_handcraft, _, Y = get_dataset(trainset_path,
                                   seq_length,
                                   width_original=width_original,
                                   height_original=height_original,
                                   undersample=undersample,
                                   sample_amount=400000,
                                   random_seed=random_seed)

# prepare train test split
test_split = 0.02 # control how much finetuning data to use. fintuning amount = 1/2 of test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_split, random_state=random_seed, stratify=Y)

X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.50, random_state=random_seed, stratify=y_test)

X_train_hc, X_test_hc, y_train_hc, y_test_hc = train_test_split(X_handcraft, Y, test_size=test_split, random_state=random_seed,
                                                                stratify=Y)

X_valid_hc, X_test_hc, y_valid_hc, y_test_hc = train_test_split(X_test_hc, y_test_hc, test_size=0.50, random_state=random_seed,
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
    epochs = 15
    patience = 15

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
    epochs = 45
    patience = 10

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
    epochs = 60
    patience = 30

    print("Fine tuning model in progress...")
    train_finetune_model(from_scratch_model, from_scratch_model_path, valid_loader, test_loader, device, lr, epochs,
                         patience)

# clear memory
del train_data, train_loader, test_data, test_loader, valid_loader

"""
Test on new dataset
"""
testset_path = os.path.join(os.getcwd(), "datasets", "testset")
X_test, X_test_hc, X_test_tensordata, y_test = get_dataset(testset_path,
                                                           seq_length,
                                                           width_original=width_original,
                                                           height_original=height_original,
                                                           undersample="random",
                                                           sample_amount=50000,
                                                           random_seed=random_seed)

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
clf = RandomForestClassifier() #RandomForestClassifier()
clf.fit(np.squeeze(X_valid.numpy()).reshape(train_size, -1), y_valid.numpy())
y_pred = clf.predict(np.squeeze(X_test.numpy()).reshape(test_size, -1))

print("No Feature Extraction:")
print(classification_report(y_true, y_pred, zero_division=0))

# handcrafted features set 1
clf_hc = RandomForestClassifier()
clf_hc.fit(np.squeeze(X_valid_hc[:, :, 28:].numpy()).reshape(train_size, -1), y_valid.numpy())
y_pred_hc = clf_hc.predict(X_test_hc[:, :, 28:].numpy().reshape(test_size, -1))

print("Handcraft Features Set 1:")
print(classification_report(y_true, y_pred_hc, zero_division=0))

# handcrafted features set 2
clf_hc = RandomForestClassifier()
clf_hc.fit(np.squeeze(X_valid_hc.numpy()).reshape(train_size, -1), y_valid.numpy())
y_pred_hc = clf_hc.predict(X_test_hc.numpy().reshape(test_size, -1))

print("Handcraft Features Set 2:")
print(classification_report(y_true, y_pred_hc, zero_division=0))

# learnt features
clf_enc = RandomForestClassifier()
clf_enc.fit(X_train_encoded, y_valid.numpy())
y_pred_enc = clf_enc.predict(X_test_encoded)
#y_pred_enc = np.argmax(predict_data(X_test_tensordata, test_model, device), axis=1)

print(f"Learnt Features ({test}):")
print(classification_report(y_true, y_pred_enc, zero_division=0))

# combined features
clf_comb = RandomForestClassifier()
X_combined = np.hstack((np.squeeze(X_valid_hc.numpy()[:,-1,:]).reshape(train_size, -1), X_train_encoded))
clf_comb.fit(X_combined, y_valid.numpy())
X_combined_test = np.hstack((X_test_hc.numpy()[:,-1,:].reshape(test_size, -1), X_test_encoded))
y_pred_comb = clf_comb.predict(X_combined_test)

print(f"Combined Features ({test}):")
print(classification_report(y_true, y_pred_comb, zero_division=0))