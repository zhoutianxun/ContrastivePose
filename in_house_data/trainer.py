import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from utils import augment_data, generate_overlap_feature
from models import ContrastiveLoss, BarlowTwinLoss


### Training function for feature extractor (Contrastive learning)
def train_epoch(model, device, dataloader, optimizer, scheduler, loss_fn):
    # Set train mode for both the encoder and the decoder
    model.train()
    train_loss = 0.0
    total = 0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    train_bar = tqdm(dataloader)
    for data in train_bar:
        x, _ = data
        xp = augment_data(x)
        xa = augment_data(x)

        xp = torch.cat((generate_overlap_feature(xp[:, 0, :, :64]), xp[:, 0, :, :]), dim=2)
        xa = torch.cat((generate_overlap_feature(xa[:, 0, :, :64]), xa[:, 0, :, :]), dim=2)

        xp = xp.to(device).float()
        xa = xa.to(device).float()

        _, project_x = model(xa)
        _, project_xp = model(xp)

        project_x = torch.squeeze(project_x)
        project_xp = torch.squeeze(project_xp)

        # Evaluate loss
        loss = loss_fn(project_x, project_xp)
        train_loss += loss.cpu().item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        total += len(x)
        train_bar.set_description(f"Loss: {train_loss / total:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    return train_loss / total


### Testing function
def test_epoch(model, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, _ in dataloader:
            xp = augment_data(x)
            xa = augment_data(x)

            xp = torch.cat((generate_overlap_feature(xp[:, 0, :, :64]), xp[:, 0, :, :]), dim=2)
            xa = torch.cat((generate_overlap_feature(xa[:, 0, :, :64]), xa[:, 0, :, :]), dim=2)

            xp = xp.to(device).float()
            xa = xa.to(device).float()

            _, project_x = model(xa)
            _, project_xp = model(xp)

            project_x = torch.squeeze(project_x)
            project_xp = torch.squeeze(project_xp)
            loss = loss_fn(project_x, project_xp)
            val_loss += loss.cpu().item()
    return val_loss / len(dataloader.dataset)


def test_predictor(model, device, dataloader):
    # Set evaluation mode for encoder and decoder
    model.eval()
    val_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():  # No need to track the gradients
        for x, y in dataloader:
            x = torch.cat((generate_overlap_feature(x[:, 0, :, :64]), x[:, 0, :, :]), dim=2)
            _, pred = model(x.to(device))
            loss = nn.CrossEntropyLoss()(pred, y.long().to(device))
            val_loss += loss.item()
            total += len(x)
            correct += torch.sum(torch.argmax(pred, axis=1) == y.to(device))
    return val_loss / total, correct / total


def train_model(model, save_path, train_loader, valid_loader, device, lr=0.002, epochs=60, patience=30,
                resume_training=False):
    if resume_training:
        model.load_state_dict(torch.load(save_path))

    model = model.to(device)
    temperature = 0.07
    loss_fn = ContrastiveLoss(train_loader.batch_size, temperature).to(device)
    #loss_fn = BarlowTwinLoss(train_loader.batch_size, 256, off_diag_weight=5e-3).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.002, pct_start=0.15,
                                                    epochs=epochs, steps_per_epoch=len(train_loader))
    min_val_loss = np.inf
    patience_steps = 0
    for epoch in range(epochs):
        train_loss = train_epoch(model, device, train_loader, optimizer, scheduler, loss_fn)
        val_loss = test_epoch(model, device, valid_loader, loss_fn)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            patience_steps = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_steps += 1
        print(
            f"\n EPOCH {epoch + 1}/{epochs} \t lr {optimizer.param_groups[0]['lr']:.5f} \t train loss {train_loss:.6f} \t "
            f"val loss {val_loss:.6f}")
        if patience_steps > patience:
            break


def train_finetune_model(model, save_path, train_loader, valid_loader, device, lr=0.001, epochs=60, patience=60):
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, pct_start=0.15,
                                                    epochs=epochs, steps_per_epoch=len(train_loader))
    min_val_loss = np.inf
    patience_steps = 0

    for epoch in range(epochs):
        train_bar = tqdm(train_loader)
        total = 0
        train_loss = 0
        model.train()
        for sample in train_bar:
            x = sample[0]
            x = torch.cat((generate_overlap_feature(x[:, 0, :, :64]), x[:, 0, :, :]), dim=2).to(device)
            label = sample[1].to(device)
            # Encode input
            _, pred = model(x)
            loss = nn.CrossEntropyLoss()(pred, label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.detach().item()
            total += len(x)
            train_bar.set_description(f"Loss: {train_loss / total:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        val_loss, acc = test_predictor(model, device, valid_loader)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            patience_steps = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_steps += 1
        print(
            f'\n EPOCH {epoch + 1}/{epochs} \t lr {optimizer.param_groups[0]["lr"]:.5f} \t train loss {train_loss / total:.4f} \t val loss {val_loss:.4f} \t accuracy {acc * 100:.3f}')
        if patience_steps > patience:
            break
