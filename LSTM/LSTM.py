# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:03:57 2023

@author: 53055
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first = True)
        self.device = device
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def read_data():
    data = pd.read_csv('AMZN.csv')
    data =data[['Date', 'Close']]
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def data_prepare(df, nsteps):
    df = dc(df)
    df.set_index('Date', inplace = True)
    for i in range(1, nsteps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace = True)
    return df

def train_one_epoch(model, epoch, device, train_loader, loss_function, optimizer):
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0
    
    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_index % 100 == 99:
            avg_loss = running_loss /100
            print('Batch {0}, Loss {1:.3f}'.format(batch_index + 1, avg_loss))
        
            running_loss = 0.0
    print()
        
def validate_one_epoch(model, epoch, device, test_loader, loss_function):
    model.train(False)
    running_loss = 0.0
    
    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        
        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
    avg_loss = running_loss / len(test_loader)
    print('Val Loss {0:.3f}'.format(avg_loss))
    print('************************************')
    print()
        
        
def train(model, device, train_loader, test_loader, learning_rate, loss_function, optimizer):
    
    num_epoch = 10

    for epoch in range(num_epoch):
        train_one_epoch(model, epoch, device, train_loader, loss_function, optimizer)
        validate_one_epoch(model, epoch, device, test_loader, loss_function)
        
if __name__ == "__main__":
    device = 'cuda: 0' if torch.cuda.is_available() else 'cpu'
    data = read_data()
    lookback = 7
    shift_df = data_prepare(data, lookback)
    shift_df_np = shift_df.to_numpy()
    
    scaler = MinMaxScaler(feature_range = (-1, 1))
    shift_df_np = scaler.fit_transform(shift_df_np)
    
    x = shift_df_np[:, 1:]
    y = shift_df_np[:, 0]
    
    x = dc(np.flip(x, axis = 1))
     
    split_index = int(len(x) * 0.95) #Test Index
     
    x_train = x[:split_index]
    x_test = x[split_index:]
    
    y_train = y[: split_index]
    y_test = y[split_index:]
    
    x_train = x_train.reshape((-1, lookback, 1))
    x_test = x_test.reshape((-1, lookback, 1))

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()
    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test).float()
    
    train_dataset = TimeSeriesDataset(x_train, y_train)
    test_dataset = TimeSeriesDataset(x_test, y_test)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)
    
    model = LSTM(1, 4, 1, device)
    model.to(device)
    learning_rate = 0.001
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    train(model, device, train_loader, test_loader, learning_rate, loss_function, optimizer)