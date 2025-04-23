import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import torch.optim as optim

from  Evaluation import evaluate
#import DataCollection
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset, Subset, DataLoader


class AllData(Dataset):
    def __init__(self, sequence_length=20):
        #Data
        # data = DataCollection.GetData()
        #data = pd.read_excel("MergedDF.xlsx")
        data = pd.read_excel("MergedDF - Copy.xlsx")
        X, y , scaler_target = self.DataPreProcessing(data)

        #exclude_columns = ['Close', 'Date']
        #feature_columns = [col for col in data.columns if col not in exclude_columns]

        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

        self.sequence_length = sequence_length
        self.scaler_target = scaler_target

    def __getitem__(self, index):
        x = self.X[index:index + self.sequence_length]
        y = self.y[index + self.sequence_length]  # predict the value right after the sequence
        return x, y

    def __len__(self):
        return len(self.X) - self.sequence_length

    def DataPreProcessing(self, df):
        df = df.copy()
        df.drop(columns=["Date"], inplace=True, errors='ignore')  # drop Date if exists

        target = df["Close"]
        features = df.drop(columns=["Close"])

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = pd.DataFrame(scaler_X.fit_transform(features), columns=features.columns)
        y_scaled = pd.Series(scaler_y.fit_transform(target.values.reshape(-1, 1)).flatten())

        return X_scaled, y_scaled, scaler_y


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size,  output_size, num_layers, dropout):
        super(LSTM,self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout = dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        #out = torch.relu(out)
        return out



if __name__ == "__main__":
    # Use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    available = torch.cuda.is_available()
    count = torch.cuda.device_count()
    print(f"Using {device} device, available {available}, count {count}")

    #Randomisation
    torch.manual_seed(99)
    batch_size = 32
    window_size = 21

    #Creating datasets
    dataset = AllData(sequence_length=window_size)

    train_end = int(len(dataset) * 0.7)
    val_end = int(len(dataset) * 0.85)

    #Create subsets
    train_dataset = Subset(dataset, range(0, train_end))
    val_dataset = Subset(dataset, range(train_end, val_end))
    test_dataset = Subset(dataset, range(val_end, len(dataset)))

    #Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    #Scalar target
    scaler_target = dataset.scaler_target

    ##HYPERPARAMETERS
    input_size = dataset.X.shape[1]
    learning_rate = 0.0003
    num_epochs = 200
    hidden_size = 512
    num_layers = 1
    dropout = 0.3
    output_size = 1

    loss_function = nn.MSELoss()
    criterion = nn.MSELoss()
    model = LSTM(input_size, hidden_size, output_size, num_layers, dropout )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device).float()

            optimizer.zero_grad()
            output = model(X) # forward pass

            loss = criterion(output, y)  #calc loss, gradient and update param
            loss.backward()
            optimizer.step()

            train_loss+=loss.item()

            if (batch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        mean_train_loss = train_loss / len(train_loader)
        train_losses.append(mean_train_loss)

        #  Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device).float()

                val_output = model(X_val)
                val_loss += criterion(val_output, y_val).item()

        mean_val_loss = val_loss / len(val_loader)
        val_losses.append(mean_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {mean_train_loss:.4f}, Val Loss: {mean_val_loss:.4f}")

    print("Training finished!")

    # Test evaluation
    model.eval()
    test_loss = 0.0
    all_preds = [] # predicted
    all_targets = [] # actual

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device).float()

            test_output = model(X_test)
            test_loss += criterion(test_output, y_test).item()

            # store predictions and targets
            all_preds.extend(test_output.cpu().numpy())
            all_targets.extend(y_test.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    print(f"\nFinal Test Loss: {avg_test_loss:.4f}")

    # reshape and inverse transform predictions and targets
    all_preds = np.array(all_preds).reshape(-1, 1)  # Ensure 2D shape for scaler
    all_targets = np.array(all_targets).reshape(-1, 1)

    preds_inverse = scaler_target.inverse_transform(all_preds)
    targets_inverse = scaler_target.inverse_transform(all_targets)

    # flatten  to 1D for plotting
    preds_inverse = preds_inverse.flatten()
    targets_inverse = targets_inverse.flatten()

    # turn into tensors
    preds_tensor = torch.from_numpy(preds_inverse).float()
    targets_tensor = torch.from_numpy(targets_inverse).float()

    ##EVALUATE
    evaluate(preds_tensor, targets_tensor,preds_inverse , targets_inverse, train_losses, val_losses)

    torch.save(model,"LSTM.pth")