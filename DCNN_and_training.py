import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import os
import pandas as pd
from torch.utils.data import (Dataset, DataLoader)
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from torch.utils.tensorboard import SummaryWriter

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 1024
learning_rate = 0.00005
batch_size =30
num_epoch = 300

# Load data
class AccDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        data_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        data = pd.read_csv(data_path, header=None)
        data = np.asarray(data)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            data = self.transform(data)

        return (data, y_label)
 
# Several different scenarios are tested as described in the paper.
# You can modify the scenarios below:
datasettrain = AccDataset(csv_file= 'Scenario#1_train.csv', root_dir = 'dataset',
                                   transform = transforms.ToTensor())
datasettest = AccDataset(csv_file= 'Scenario#1_test.csv', root_dir = 'dataset',
                                   transform = transforms.ToTensor())
# define the data loaders for train and tests
train_loader = DataLoader(dataset=datasettrain, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=datasettest, batch_size=batch_size, shuffle=True)

# Model
class CNN(nn.Module):
    def __init__(self, channels_img, features_d):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            # input: Nx1x1024
            nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1), # Nx32x512
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(32, 64, 4, 2, 1), # Nx64x256
            nn.Dropout(p=0.5),
            self._block(64, 128, 4, 2, 1), # Nx128x128
            nn.Dropout(p=0.5),
            self._block(128, 256, 4, 2, 1), # Nx256x64
            nn.Dropout(p=0.5),
            nn.Conv1d(256, 1, kernel_size=64, stride=2, padding=0), # Nx1x1 
            nn.Sigmoid())

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),  
        )
    def forward(self, x):
        return self.cnn(x)

# Initialize network
model = CNN(1, 1024).to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
model.train()
writer = SummaryWriter()

# Train network
for epoch in range (num_epoch):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device, dtype=torch.float)
        targets = targets.to(device=device, dtype=torch.float)
        data = data.squeeze(-1)
        # forward
        train_scores = model(data)
        train_scores = train_scores.squeeze(-1)
        targets = targets.unsqueeze(-1)
        train_loss = criterion(train_scores, targets)
        # backward
        optimizer.zero_grad()
        train_loss.backward()
        # Adamw step
        optimizer.step()
        
        # RMSE, Brier, Logloss
        targets = np.array(targets.detach().cpu())
        train_scores = np.array(train_scores.detach().cpu())
        mse_train = mean_squared_error(targets, train_scores)
        logloss_train = log_loss(targets, train_scores)
        
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("mse_train", mse_train, epoch)
        writer.add_scalar("logloss_train", logloss_train, epoch)

        print(f"Epoch [{epoch}/{num_epoch}] Batch {batch_idx}/{len(train_loader)} \
                  train_loss: {train_loss:.4f}")
        
        model.train()

# Test network
model.eval()

for epoch in range (num_epoch):
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate (test_loader):
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)
            x = x.squeeze(-1)
            test_scores = model(x)
            test_scores = test_scores.squeeze(-1)
            y = y.unsqueeze(-1)
            test_loss = criterion(test_scores, y)
                
            # RMSE, Brier, Logloss
            y = np.array(y.cpu())
            test_scores = np.array(test_scores.cpu())
            mse_test = mean_squared_error(y, test_scores)
            logloss_test = log_loss(y, test_scores)

            writer.add_scalar("test_loss", test_loss, epoch)