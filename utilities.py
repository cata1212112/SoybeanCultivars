import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from torch import device
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from sklearn.metrics import r2_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SoyBeanDataset(Dataset):
    def __init__(self, X, y):
        super(SoyBeanDataset, self).__init__()
        self.X = X.to_numpy()
        self.y = y.to_numpy()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float), torch.tensor(self.y[idx], dtype=torch.float)

class VAEDataset(Dataset):
    def __init__(self, X):
        super(VAEDataset, self).__init__()
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float)

def clip_outliers(dataframe):
    float_columns = dataframe.select_dtypes(include='float64')

    for col in float_columns:
        q1 = np.percentile(dataframe[col], 25)
        q3 = np.percentile(dataframe[col], 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        dataframe[col] = dataframe[col].clip(lower_bound, upper_bound)

    return dataframe


def get_kfold(dataframe):
    kf = KFold(n_splits=10, random_state=7, shuffle=True)

    for i, (train_index, test_index) in enumerate(kf.split(dataframe)):
        yield train_index, test_index


def train(model, EPOCHS, criterion, optimizer, X_train, y_train, X_val, y_val):
    print("OK")
    train_losses = []
    val_losses = []
    train_loader = DataLoader(dataset=SoyBeanDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=SoyBeanDataset(X_val, y_val), batch_size=32, shuffle=False)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_val_loss = 0.0
        train_pred = []
        train_true = []
        val_pred = []
        val_true = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            pred = model(inputs)
            loss = criterion(pred.squeeze(1), labels)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_pred.extend((pred.squeeze(1)).tolist())
            train_true.extend(labels.tolist())

        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                pred = model(inputs)
                loss = criterion(pred.squeeze(1), labels)
                running_val_loss += loss.item()

                val_pred.extend((pred.squeeze(1)).tolist())
                val_true.extend(labels.tolist())

        model.train()

        train_losses.append(running_loss / len(train_loader))
        val_losses.append(running_val_loss / len(test_loader))

    return train_losses, val_losses

def train_vae(model, X, criterion, optimizer, EPOCHS):
    model.train()
    loader = DataLoader(dataset=VAEDataset(X), batch_size=32, shuffle=True)
    all_losses = []
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for input in loader:
            input = input.to(device)
            reconstructed, mu, logvar = model(input)
            loss = criterion(reconstructed, input, mu, logvar)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        all_losses.append(running_loss / len(loader))

    return all_losses

def generate_data(model):
    z = torch.randn(4, 32).to(device)
    return model.decoder(z).cpu().numpy()

def permutation_feature_importance(model, X, y, criterion, n_repeats=100):
    X = X.to_numpy()
    baseline_error = criterion(model(torch.tensor(X, dtype=torch.float).to(device)).cpu().detach().numpy().squeeze(1),
                               y)
    importances = []
    for feature in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, feature] = np.random.permutation(X_permuted[:, feature])
            scores.append(criterion(
                model(torch.tensor(X_permuted, dtype=torch.float).to(device)).cpu().detach().numpy().squeeze(1), y))

        importances.append(np.mean(scores) - baseline_error)

    return importances


def gradient_feature_importance(model, X):
    model.eval()
    output = model(X)
    gradients = torch.autograd.grad(output, X, torch.ones_like(output))[0]

    feature_importance = torch.mean(torch.abs(gradients), dim = 0).cpu().detach().numpy()
    return feature_importance

def weigths_feature_importance(model):
    model.eval()
    weigths = []
    for param in model.parameters():
        weigths.append(param.data.cpu().numpy())

    feature_importance = np.sum(np.abs(weigths[0]), axis = 1)
    return feature_importance

#%%
