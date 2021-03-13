import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def ensure_reproducibility(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

''' obtain jet data from the csv file'''
def parse_csv(filepath):
    
    with open(filepath, "r") as f:
        event_lines = f.readlines()
    
    jets_data = []
    for line in event_lines:
        objects_in_event = line.split(';')[5:-1]
        for obj in objects_in_event:
            if obj[0] == 'j':
                jets_data.append(obj.split(',')[1:])

    jets_df = pd.DataFrame(jets_data, columns=['E', 'pt', 'eta', 'phi'])
    return jets_df.astype('float')

'''plot the distribution of the four variables'''
def plot_dist(df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    cols = df.columns
    sns.histplot(ax=axes[0, 0], data=df[cols[0]])
    sns.histplot(ax=axes[0, 1], data=df[cols[1]])
    sns.histplot(ax=axes[1, 0], data=df[cols[2]])
    sns.histplot(ax=axes[1, 1], data=df[cols[3]])

'''utility to train the model - optimizer, scheduler, loss function hardcoded'''
def train(model, train_dl, val_dl, epochs, verbose=True):
   
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    
    for epoch in tqdm(range(epochs)):
        running_loss = 0
        for batch in train_dl:

            preds = model(batch[0])
            loss = criterion(preds, batch[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss/len(train_dl))

        running_loss = 0
        with torch.no_grad():
            for batch in val_dl:
                preds = model(batch[0])
                loss = criterion(preds, batch[0])

                running_loss += loss.item()
            val_losses.append(running_loss/len(val_dl))

        scheduler.step()

        if verbose:
            print(f"Train loss = {train_losses[-1]}")
            print(f"Val loss = {val_losses[-1]}")
        
    return train_losses, val_losses

'''compare the distributions of the input and output of the autoencoder'''
def compare_input_output(ae_input, ae_output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    sns.histplot(ax=axes[0, 0], data=ae_output[:, 0], color='c', bins=100, label='Output')
    sns.histplot(ax=axes[0, 0], data=ae_input[:, 0], color='orange', bins=100, label='Input')
    axes[0, 0].set_xlabel("E")
    axes[0, 0].set_ylabel("Number of objects")

    sns.histplot(ax=axes[0, 1], data=ae_output[:, 1], color='c', bins=100, label='Output')
    sns.histplot(ax=axes[0, 1], data=ae_input[:, 1], color='orange', bins=100, label='Input')
    axes[0, 1].set_xlabel("pt")
    axes[0, 1].set_ylabel("Number of objects")

    sns.histplot(ax=axes[1, 0], data=ae_output[:, 2], color='c', bins=100, label='Output')
    sns.histplot(ax=axes[1, 0], data=ae_input[:, 2], color='orange', bins=100, label='Input')
    axes[1, 0].set_xlabel("eta")
    axes[1, 0].set_ylabel("Number of objects")

    sns.histplot(ax=axes[1, 1], data=ae_output[:, 3], color='c', bins=100, label='Output')
    sns.histplot(ax=axes[1, 1], data=ae_input[:, 3], color='orange', bins=100, label='Input')
    axes[1, 1].set_xlabel("phi")
    axes[1, 1].set_ylabel("Number of objects")

    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[1, 0].legend()
    axes[1, 1].legend()

'''remove outliers from the residuals'''
def remove_outliers(residuals, num_to_remove=5):
    outliers = []
    outliers.extend(np.argsort(residuals[:, 0])[:num_to_remove])
    outliers.extend(np.argsort(residuals[:, 0])[-num_to_remove:])

    outliers.extend(np.argsort(residuals[:, 1])[:num_to_remove])
    outliers.extend(np.argsort(residuals[:, 1])[-num_to_remove:])

    outliers.extend(np.argsort(residuals[:, 2])[:num_to_remove])
    outliers.extend(np.argsort(residuals[:, 2])[-num_to_remove:])

    outliers.extend(np.argsort(residuals[:, 3])[:num_to_remove])
    outliers.extend(np.argsort(residuals[:, 3])[-num_to_remove:])

    outliers = set(outliers)
    residuals_without_outliers = np.delete(residuals, list(outliers), axis=0)

    return residuals_without_outliers