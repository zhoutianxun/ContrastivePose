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
test = "finetune"  # contrastive, finetune, from_scratch
experiment = 8

# Constants
width_original = 1920
height_original = 1080
impute = True
seq_length = 30
classes = 10
batch_size = 64
original_features = 144 #128
extracted_features = 80 #64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pio.renderers.default = 'browser'

behavior_class = {0: "nose-nose sniff",
                  1: "mutual circle",
                  2: "anogenital 1",
                  3: "anogenital 2",
                  4: "body sniff 1",
                  5: "body sniff 2",
                  6: "following 1",
                  7: "following 2",
                  8: "affliative",
                  9: "exploration"}

"""
Load Data
"""
if train:
    trainset_path = [os.path.join("datasets", f) for f in os.listdir("datasets") if f[:5] != "A0153"]
    X_train, _, _, y_train = get_dataset(trainset_path,
                                         seq_length,
                                         width_original=width_original,
                                         height_original=height_original,
                                         undersample=None)

    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    del X_train, y_train  # clear memory

trainset_path = os.path.join("datasets", "A0153_1.csv")
X_test, X_test_hc, _, Y_test = get_dataset(trainset_path,
                                              seq_length,
                                              width_original=width_original,
                                              height_original=height_original,
                                              undersample="equalize")

# prepare train test split
test_split = 0.20
X_valid, X_test, y_valid, y_test = train_test_split(X_test, Y_test, test_size=test_split, random_state=42, stratify=Y_test)
X_valid_hc, X_test_hc, y_valid_hc, y_test_hc = train_test_split(X_test_hc, Y_test, test_size=test_split, random_state=42,
                                                                stratify=Y_test)

valid_data = TensorDataset(X_valid, y_valid)
test_data = TensorDataset(X_test, y_test)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

"""
Training of model
"""
if train:
    feature_extractor = Contrastive_model(seq_length, original_features, extracted_features)
    contrastive_model_path = os.path.join(os.getcwd(), "Models", f"contrast_v{experiment}.pt")
    lr = 0.003
    epochs = 100
    patience = 30

    print("Training model in progress...")
    train_model(feature_extractor, contrastive_model_path, train_loader, valid_loader, device, lr, epochs, patience,
                False)

"""
Fine tune model with validation set
"""
if train:
    contrastive_model_path = os.path.join(os.getcwd(), "Models", f"contrast_v{experiment}.pt")
    basemodel = Contrastive_model(seq_length, original_features, extracted_features)
    basemodel.load_state_dict(torch.load(contrastive_model_path))
    fine_tune_model = FineTuneModel(basemodel, extracted_features, classes)
    fine_tune_model = fine_tune_model.to(device)
    finetune_model_path = os.path.join(os.getcwd(), "Models", f"finetune_v{experiment}.pt")
    lr = 0.001
    epochs = 100
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

"""
Test on new dataset
"""
testset_path = os.path.join("datasets", "A0153_2.csv")
X_test, X_test_hc, X_test_tensordata, y_test = get_dataset(testset_path,
                                                           seq_length,
                                                           width_original=width_original,
                                                           height_original=height_original,
                                                           undersample=None)

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
print("Encoding data in progress...")
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
#clf_enc.fit(np.concatenate((X_train_encoded, X_valid_hc.numpy()[:,:,:16].reshape(train_size, -1)), axis=1), y_valid.numpy())
#y_pred_enc = clf_enc.predict(np.concatenate((X_test_encoded, X_test_hc.numpy()[:,:,:16].reshape(test_size, -1)), axis=1))
clf_enc.fit(X_train_encoded, y_valid.numpy())
y_pred_enc = clf_enc.predict(X_test_encoded)

print(f"Learnt Features ({test}):")
print(classification_report(y_true, y_pred_enc, zero_division=0))