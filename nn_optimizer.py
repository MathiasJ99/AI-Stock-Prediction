import optuna
import torch
import torch.nn as nn
import pandas as pd
import torch.utils.data
import torch.optim as optim
from optuna.exceptions import TrialPruned
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Subset, DataLoader
from optuna.storages import RDBStorage
import numpy as np
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from DataCollection import GetData
from  Evaluation import evaluate
import torch.nn.init as init
import time
import numpy as np
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
import os
from fredapi import Fred
from functools import reduce

from torchmetrics import MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError, R2Score

torch.manual_seed(99)
# ------ GPU settings -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
available = torch.cuda.is_available()
count = torch.cuda.device_count()
print(f"Using {device} device, available {available}, count {count}")

class AllData(Dataset):
    def __init__(self, sequence_length=20, data=pd.read_excel("MergedDF_googl.xlsx")):
        self.data = data
        X, y , scaler_target = self.DataPreProcessing(data)

        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

        self.sequence_length = sequence_length
        self.scaler_target = scaler_target # used to revert

    def __getitem__(self, index):
        X = self.X[index:index + self.sequence_length]
        y = self.y[index + self.sequence_length]  # predict the value right after the sequence
        return X, y

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

        if num_layers <= 1:
            dropout = 0

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout = dropout, batch_first=True)
        #self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        out, _ = self.lstm(x)
        #out = self.relu(out)
        out = self.fc(out[:, -1, :])
        return out

class CNN_LSTM(nn.Module):
    def __init__(self,input_size,window_size, num_filters, kernel_size, lstm_hidden_size, num_layers, dropout, output_size):
        super(CNN_LSTM, self).__init__()
        #attributes
        self.input_size = input_size
        self.window_size = window_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        if num_layers <= 1:
            self.dropout = 0
        else:
            self.dropout = dropout


        #1d conv layer
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size)
        # relu activation func
        self.relu = nn.ReLU()

        #LSTM Layer
        self.cnn_output_len = window_size - kernel_size +1
        self.lstm = nn.LSTM(input_size=num_filters * input_size,  # flattened output from CNN
                            hidden_size=lstm_hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)

        #Fc linear layer
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        #apply cnn
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  #
        batch_size, channels, window_size, input_size = x.size()
        #make sure its feature wise processing
        x = x.reshape(batch_size * input_size, channels, window_size)
        x = self.conv1d(x)

        #relu
        x = self.relu(x)

        #change x for lstm input
        x = x.view(batch_size, input_size, self.num_filters, self.cnn_output_len)
        x = x.permute(0, 3, 1, 2).contiguous()  # [batch_size, cnn_output_len, input_size, num_filters]
        x = x.view(batch_size, self.cnn_output_len, input_size * self.num_filters) # [batch size, window, input size]

        # apply lstm layer(s)
        out, _ = self.lstm(x)

        #fc layer
        out = self.fc(out[:, -1, :])
        return out

def train_validate_model(model,train_loader, val_loader, optimizer, criterion, trial, epochs, patience):
    # -------- training -------
    train_losses = []
    val_losses = []
    num_epochs = epochs
    best_val_loss = float("inf")
    patience = patience
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device).float()

            optimizer.zero_grad()
            output = model(X)  # forward pass

            loss = criterion(output, y)  # calc loss, gradient and update param
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        mean_train_loss = train_loss / len(train_loader)
        train_losses.append(mean_train_loss)

        # ----- Validation ----------
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

        if trial is not None:
            print(f"Trial {trial.number} - Epoch {epoch + 1}/{num_epochs} -> Train Loss: {mean_train_loss:.6f}, Val Loss: {mean_val_loss:.6f}")
        else:
            print(f"Epoch {epoch + 1}/{num_epochs} -> Train Loss: {mean_train_loss:.6f}, Val Loss: {mean_val_loss:.6f}")


        ## prevent overfitting
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            patience_counter = 0  # Reset counter when improvement happens
        else:
            patience_counter += 1
            # print(f"No improvement in val loss for {patience_counter} epochs.")
            if patience_counter >= patience:
                print(f"Early stopping triggered val didn't improve for {patience} epochs")
                break  # Stops training

        # ----- prune underperforming models -----
        if trial is not None:
            trial.report(mean_val_loss, epoch)
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch + 1}.")
                raise TrialPruned()

    if trial is not None:
        print(f"Trial {trial.number} finished. Final Validation Loss: {mean_val_loss:.6f}")
    else:
        print(f"Final Validation Loss: {mean_val_loss:.6f}")

    return mean_val_loss, train_losses, val_losses, model

def test_model():
    pass

## optimize hyperparameters
def make_objective(model, data):
    def objective(trial, model=model, data=data):
        print(f"trial {trial.number}")
        # ------- hyperparameters part 1 (common for all models) -------
        window_size = trial.suggest_int("window_size", 5, 25)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        dropout = trial.suggest_float("dropout", 0.1, 0.9)

        # ------ dataset stuff -------
        # Creating datasets
        dataset = AllData(sequence_length=window_size, data=data)

        train_end = int(len(dataset) * 0.7)
        val_end = int(len(dataset) * 0.85)

        # Create subsets
        train_dataset = Subset(dataset, range(0, train_end))
        val_dataset = Subset(dataset, range(train_end, val_end))
        test_dataset = Subset(dataset, range(val_end, len(dataset)))

        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Scalar target
        scaler_target = dataset.scaler_target

        #----- model & optimzer ----
        input_size = dataset.X.shape[1]
        output_size = 1
        # ------- hyperparameters part 2 (different for all models) -------
        if model == "LSTM":
            learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.01, log=True)
            num_layers = trial.suggest_int("num_layers", 1, 5)
            hidden_size = trial.suggest_int("hidden_size", 64, 1024)
            optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])

            model = LSTM(input_size, hidden_size, output_size, num_layers, dropout)

        elif model == "CNN_LSTM":
            learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.01, log=True)
            num_layers = trial.suggest_int("num_layers", 1, 5)
            lstm_hidden_size = trial.suggest_int("lstm_hidden_size", 64, 1024)
            optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])
            num_filters = trial.suggest_int("num_filters", 1, 256)
            kernel_size = trial.suggest_int("kernel_size", 1, 1)

            model = CNN_LSTM(input_size=input_size, window_size=window_size,num_filters=num_filters, kernel_size=kernel_size, lstm_hidden_size=lstm_hidden_size, num_layers=num_layers,dropout=dropout, output_size=output_size)

        model.to(device)

        if optimizer_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        elif optimizer_name == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr = learning_rate)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        criterion = nn.MSELoss()
        criterion.to(device)


        #-------- training & val -------
        mean_val_loss, train_losses, val_losses, best_model= train_validate_model(
            model= model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion= criterion,
            trial = trial,
            epochs = 15,
            patience = 10
        )

        return mean_val_loss
    return objective

def make_test(model, data):
    def test_best_objective(best_trial, model=model, data=data):
        # ---- create best model based on best hyperparams --------
        best_params = best_trial.params
        window_size = best_params["window_size"]

        # Creating datasets
        dataset = AllData(sequence_length=window_size,data=data)
        train_end = int(len(dataset) * 0.7)
        val_end = int(len(dataset) * 0.85)

        # Create subsets
        train_dataset = Subset(dataset, range(0, train_end))
        val_dataset = Subset(dataset, range(train_end, val_end))
        test_dataset = Subset(dataset, range(val_end, len(dataset)))

        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"])
        val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"])
        test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"])

        # Scalar target
        scaler_target = dataset.scaler_target
        input_size = dataset.X.shape[1]
        output_size = 1

        if model == "LSTM":
            best_model = LSTM(input_size,
                         hidden_size=best_params["hidden_size"],
                         output_size=output_size ,
                         num_layers=best_params["num_layers"],
                         dropout=best_params["dropout"])
        elif model == "CNN_LSTM":
            best_model = CNN_LSTM(input_size,
                              num_filters=best_params["num_filters"],
                              output_size=output_size,
                              num_layers=best_params["num_layers"],
                              kernel_size=best_params["kernel_size"],
                              lstm_hidden_size=best_params["lstm_hidden_size"],
                              dropout=best_params["dropout"],
                              window_size = window_size )

        best_model.to(device)




        if best_params["optimizer"] == "AdamW":
            optimizer = optim.AdamW(best_model.parameters(), lr=best_params["learning_rate"])
        elif best_params["optimizer"] == "RMSprop":
            optimizer = optim.RMSprop(best_model.parameters(), lr=best_params["learning_rate"])
        elif best_params["optimizer"] == "SGD":
            optimizer = optim.SGD(best_model.parameters(), lr=best_params["learning_rate"])
        else:  # Adam
            optimizer = optim.Adam(best_model.parameters(), lr=best_params["learning_rate"])

        criterion = nn.MSELoss()
        criterion.to(device)

        #-----  train and val model on best hyperparam -----------
        best_val_loss, train_losses, val_losses, m = train_validate_model(
            model=best_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            trial = None,
            epochs=400,
            patience=15,
        )

        # ---- evaluate best model ---------
        best_model.eval()
        test_loss = 0.0
        all_preds = []  # predicted
        all_targets = []  # actual
        criterion = nn.MSELoss()
        criterion.to(device)


        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test = X_test.to(device)
                y_test = y_test.to(device).float()

                test_output = best_model(X_test)
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
        evaluate(preds_tensor, targets_tensor, preds_inverse, targets_inverse, train_losses, val_losses)

        model_name = "best_" + model +".pth"
        torch.save(best_model, model_name)
    return test_best_objective

if __name__ == "__main__":
    # --- Run the Optimization ---
    study_name = ("cnn-lstm-check-22")
    storage = RDBStorage(
        url=f"sqlite:///{study_name}.db",
        engine_kwargs={"connect_args": {"check_same_thread": False}}
    )

    study = optuna.create_study(
        study_name = study_name,
        storage = storage,
        #load_if_exists = True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=99),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))

    models = ["LSTM", "CNN_LSTM"]
    datasets = [pd.read_excel("MergedDF_googl.xlsx"),pd.read_excel("MergedDF_gold.xlsx")]
    #data = DataCollection.GetData()


    model = models[0]
    data = datasets[0]

    study.optimize(make_objective(model,data),  n_trials=5)

    print("Best trial:")
    print(f"Value (Val Loss): {study.best_trial.value}")
    print(f"best params: {study.best_params}")

    print("----------- training a model using best hyperparameters -----------")
    test_best_objective = make_test(model,data)  # returns a function
    test_best_objective(study.best_trial)


''' 
    # Save the entire best model
    model_filename = "best_LSTM_model_Gold.pth"
    torch.save(best_model, model_filename)
    print(f"Best model (full) saved to {model_filename}")

'''
