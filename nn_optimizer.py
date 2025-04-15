import optuna
import torch
import torch.nn as nn
import pandas as pd
import torch.utils.data
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Subset, DataLoader
from optuna.storages import RDBStorage


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

        if num_layers <= 1:
            dropout = 0

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout = dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        #out = torch.relu(out)
        return out


def objective(trial):
    #------ GPU settings -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    available = torch.cuda.is_available()
    count = torch.cuda.device_count()
    print(f"Using {device} device, available {available}, count {count}")
    print(f"trial {trial.number}")


    #------- hyperparameters -------
    torch.manual_seed(99)

    learning_rate = trial.suggest_float("lr", 0.00001, 0.001, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    hidden_size = trial.suggest_int("hidden_size", 128, 1024)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    window_size = trial.suggest_int("sequence_length", 10, 60)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW","RMSprop" ,"SGD"])

    # ------ dataset stuff -------
    # Creating datasets
    dataset = AllData(sequence_length=window_size)

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
    model = LSTM(input_size, hidden_size, output_size, num_layers, dropout)
    model.to(device)

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    else:  # Adam
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()  # Mean Squared Error is common for price prediction

    #-------- training -------
    train_losses = []
    val_losses = []
    num_epochs = 200

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

            #if (batch + 1) % 10 == 0:
            #    print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        mean_train_loss = train_loss / len(train_loader)
        train_losses.append(mean_train_loss)

        #----- Validation ----------
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

        print(f"Trial {trial.number} - Epoch {epoch+1}/{num_epochs} -> Train Loss: {mean_train_loss:.6f}, Val Loss: {mean_val_loss:.6f}")

        # ----- pruning -----
        trial.report(mean_val_loss, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch + 1}.")

    print(f"Trial {trial.number} finished. Final Validation Loss: {mean_val_loss:.6f}")
    return val_loss

if __name__ == "__main__":
    # --- Run the Optimization ---
    study_name = "lstm-price-optimization-1"
    storage = RDBStorage(
        url=f"sqlite:///{study_name}.db",
        engine_kwargs={"connect_args": {"check_same_thread": False}}
    )

    # Use a sampler like TPE (Bayesian) and a pruner
    study = optuna.create_study(
        study_name = study_name,
        storage = storage,
        #load_if_exists = True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=99),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))

    study.optimize(objective, n_trials=20)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Validation Loss): {trial.value}")
    print(" best params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    #TODO
    # Retrain model with best hyperparameters
    # train on entire dataset
    # save model


''' 
    # Save the entire best model
    model_filename = "best_LSTM_model_Gold.pth"
    torch.save(best_model, model_filename)
    print(f"Best model (full) saved to {model_filename}")

'''