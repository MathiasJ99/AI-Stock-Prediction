import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
#import DataCollection
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from copy import deepcopy as dc
from sklearn.model_selection import train_test_split


## this function creates lookback window (close prices of previous days )
def CreateLookbackWindow(df, n_steps):
    df = dc(df)
    df.set_index('Date', inplace=True)
    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)
    return df

def DataPreProcessing(data):
    #data = DataCollection.GetData()
    data = pd.read_excel("MergedDF.xlsx")

    window_size = 7 # days
    processed_df = CreateLookbackWindow(data,window_size)
    X = processed_df.loc[:, processed_df.columns != 'Close'].to_numpy()
    y = processed_df["Close"].to_numpy()

    scaler = MinMaxScaler(feature_range=(-1, 1))

    y = y.reshape(-1,1)
    X, y = scaler.fit_transform(X), scaler.fit_transform(y).flatten()
    X = dc(np.flip(X, axis=1))
    #print(X.shape,y.shape )

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)

    # Split the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=99)  # 20% of train set for validation

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    #reshaping to use lstm with pytorch
    num_features = X.shape[1]

    X_train = X_train.reshape((-1, 1, num_features))
    X_test = X_test.reshape((-1, 1, num_features))

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    #convert to pytorch tensors
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()




    train_dataset = TensorDataset(X_train, y_train)
    validation_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    batch_size = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    for _, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        print(x_batch.shape, y_batch.shape)
        break

    return X_train, X_val, X_test, y_train, y_val, y_test


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def Train():
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

        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index + 1,avg_loss_across_batches))
            running_loss = 0.0

        print()


def Validation():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()


def Plot():
    plt.plot(y_train, label='Actual Close')
    plt.plot(predicted, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()

# Use GPU
device = torch.device("Cuda" if torch.cuda.is_available() else "cpu")
available = torch.cuda.is_available()
count = torch.cuda.device_count()
print(f"Using {device} device, available {available}, count {count}")

#Randomisation
torch.manual_seed(99)

##HYPERPARAMETERS
learning_rate = 0.005
num_epochs = 250
loss_function = nn.MSELoss()

model = LSTM(input_size=num_features, hidden_size=4, num_stacked_layers=1)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

sdafsadf = DataPreProcessing()
for epoch in range(num_epochs):
    Train()
    Validation()
Plot()