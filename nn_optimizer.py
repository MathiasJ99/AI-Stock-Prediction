import copy

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from optuna.exceptions import TrialPruned
from optuna.storages import RDBStorage
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Subset, DataLoader

import Evaluation

torch.manual_seed(99)
TPE_SEED = 99
# ------ GPU settings -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
available = torch.cuda.is_available()
count = torch.cuda.device_count()
print(f"Using {device} device, available {available}, count {count}")

##
NUM_TRIALS = 3 #250
EPOCHS_PER_TRIAL = 20
EPOCHS_BEST_TRIAL = 200 #1000
TRIALS_BEFORE_PRUNING = 2 #50
EPOCHS_BEFORE_PRUNING = 2  # 25
PATIENCE_TRAINING = 10
PATIENCE_TESTING = 25

TRAINING_SEEDS = [0,1,2]
TEST_SEEDS = [22,44,99]


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

class Baseline_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(Baseline_MLP, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

def train_validate_model(model,train_loader, val_loader, optimizer, criterion, trial, epochs, patience, seed):
    # -------- training -------
    train_losses = []
    val_losses = []
    num_epochs = epochs
    best_val_loss = float("inf")
    patience = patience
    patience_counter = 0
    torch.manual_seed(seed)

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
            print(f"trial {trial.number} - epoch {epoch + 1}/{num_epochs} -> train Loss: {mean_train_loss:.6f}, val Loss: {mean_val_loss:.6f}")
        else:
            print(f"epoch {epoch + 1}/{num_epochs} -> train Loss: {mean_train_loss:.6f}, val Loss: {mean_val_loss:.6f}")


        ## prevent overfitting
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            patience_counter = 0  # Reset counter when model improves
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"early stopping triggered val didn't improve for {patience} epochs")
                break  # Stops training

        # ----- prune underperforming models -----
        if trial is not None:
            trial.report(mean_val_loss, epoch)
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch + 1}.")
                raise TrialPruned()

    if trial is not None:
        print(f"Trial {trial.number} finished Final Validation Loss: {mean_val_loss:.6f}")
    else:
        print(f"Final Validation Loss: {mean_val_loss:.6f}")

    return mean_val_loss, train_losses, val_losses, model

## optimize hyperparameters
def make_objective(model, data):
    def objective(trial, model=model, data=data):
        print(f"trial {trial.number}")
        # ------- hyperparameters part 1 (common for all models) -------
        window_size = trial.suggest_int("window_size", 5, 25)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        if model != "Baseline":
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
            kernel_size = trial.suggest_int("kernel_size", 1, 5)

            model = CNN_LSTM(input_size=input_size, window_size=window_size,num_filters=num_filters, kernel_size=kernel_size, lstm_hidden_size=lstm_hidden_size, num_layers=num_layers,dropout=dropout, output_size=output_size)

        elif model == "Baseline":
            learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.01, log=True)
            num_layers = trial.suggest_int("num_layers", 1, 5)
            hidden_size = trial.suggest_int("hidden_size", 32, 1024)
            optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])

            model = Baseline_MLP(input_size=input_size*window_size, hidden_size=hidden_size, output_size=1, num_layers= num_layers)

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
        seed_val_loss = []
        for seed in TRAINING_SEEDS:
            model_copy = copy.deepcopy(model)
            mean_val_loss, train_losses, val_losses, best_model= train_validate_model(
                model= model_copy,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion= criterion,
                trial = trial,
                epochs = EPOCHS_PER_TRIAL,
                patience = PATIENCE_TRAINING,
                seed = seed,
            )
            seed_val_loss.append(mean_val_loss)

        mean_seed_val_loss = sum(seed_val_loss) / len(seed_val_loss)

        #return mean_val_loss
        return mean_seed_val_loss
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
        elif model == "Baseline":
            best_model =Baseline_MLP(input_size=input_size*window_size,
                                     hidden_size=best_params["hidden_size"],
                                     output_size=1,
                                     num_layers=best_params["num_layers"],
                                     )

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
        all_preds_tensors = []
        all_targets_tensors = []

        for seed in TEST_SEEDS:
            best_model_copy = copy.deepcopy(best_model)
            best_val_loss, train_losses, val_losses, m = train_validate_model(

                model=best_model_copy,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                trial = None,
                epochs=EPOCHS_BEST_TRIAL,
                patience=PATIENCE_TESTING,
                seed = seed,
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

            print(f"seed: {seed}")
            print(f"Final Test Loss: {avg_test_loss:.4f}")

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

            #save for cal average
            all_preds_tensors.append(preds_tensor)
            all_targets_tensors.append(targets_tensor)



            model_name = "seed"+ str(seed) +"_"+ model + "_google_.pth"
            torch.save(best_model, model_name)

            ##displays evaluation metrics for each seed
            Evaluation.evaluate_metrics(preds_tensor, targets_tensor, preds_inverse, targets_inverse)

            print("--------------")
        # ----- calc mean of tensors and display mean graphs ------
        all_preds_tensors = torch.stack(all_preds_tensors)
        all_targets_tensors = torch.stack(all_targets_tensors)

        mean_preds = all_preds_tensors.mean(dim = 0)
        mean_targets = all_targets_tensors.mean(dim=0)

        # Convert to numpy and reshape for scaler
        mean_preds_np = mean_preds.numpy().reshape(-1, 1)
        mean_targets_np = mean_targets.numpy().reshape(-1, 1)

        #flatten for 1d plotting
        mean_preds_inverse = scaler_target.inverse_transform(mean_preds_np).flatten()
        mean_targets_inverse = scaler_target.inverse_transform(mean_targets_np).flatten()

        # displays evalutaion graphs for average of all seeds
        print("---------- Models Average Metrics ----------------")
        Evaluation.evaluate_metrics(mean_preds,mean_targets,mean_preds_inverse, mean_targets_inverse)
        Evaluation.evaluate_graph_mean(mean_preds_inverse,mean_targets_inverse)



    return test_best_objective

if __name__ == "__main__":
    # --- Run the Optimization ---
    study_name = ("baseline-google-test-7")
    storage = RDBStorage(
        url=f"sqlite:///{study_name}.db",
        engine_kwargs={"connect_args": {"check_same_thread": False}}
    )

    study = optuna.create_study(
        study_name = study_name,
        storage = storage,
        #load_if_exists = True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=TPE_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=TRIALS_BEFORE_PRUNING, n_warmup_steps=EPOCHS_BEFORE_PRUNING))
        #pruner=optuna.pruners.HyperbandPruner(min_resource=TRIALS_BEFORE_PRUNING, max_resource=100, reduction_factor=3))

    models = ["LSTM", "CNN_LSTM","Baseline"]
    datasets = [pd.read_excel("MergedDF_googl.xlsx"),pd.read_excel("MergedDF_gold.xlsx")]
    #data = DataCollection.GetData()


    model = models[2]
    data = datasets[0]

    study.optimize(make_objective(model,data),  n_trials=NUM_TRIALS)

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
