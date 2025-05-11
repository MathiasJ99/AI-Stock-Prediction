import copy
import torch.nn.utils as nn_utils  # For gradient clippin
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from optuna.exceptions import TrialPruned
from optuna.storages import RDBStorage
from sklearn.preprocessing import StandardScaler, RobustScaler  # data scaling
from torch.utils.data import Dataset, Subset, DataLoader
import Evaluation

# ------ GPU settings -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
available = torch.cuda.is_available()
count = torch.cuda.device_count()
print(f"Using {device} device, available: {available}, device count: {count}")

# ------ Optuna Config settings-----
NUM_TRIALS = 1000 #25  #200
EPOCHS_PER_TRIAL = 40  # 40
TRIALS_BEFORE_PRUNING = 10  # 10
EPOCHS_BEFORE_PRUNING = 5

EPOCHS_BEST_TRIAL = 300  # 500
PATIENCE_TRAINING = 10  # 10
PATIENCE_TESTING = 25  # 25

TEST_SEEDS = [22, 44, 99]
TPE_SEED = 99


class AllData(Dataset):
    def __init__(self, sequence_length=20, data=pd.read_excel("MergedDF_googl.xlsx"), scaler_X=None, scaler_y=None,fit_scalers=False):
        self.sequence_length = sequence_length
        self.data = data

        df = self.data.copy()
        df.drop(columns=["Date"], inplace=True, errors='ignore')
        df["Close_log"] = np.log(df["Close"])

        features = df.drop(columns=["Close", "Close_log"])
        target = df["Close_log"]

        # Init scalers if not provided
        if scaler_X is None:
            self.scaler_X = RobustScaler()
        else:
            self.scaler_X = scaler_X

        if scaler_y is None:
            self.scaler_y = StandardScaler()
        else:
            self.scaler_y = scaler_y

        # use scaler to scale data
        if fit_scalers:
            X_scaled = self.scaler_X.fit_transform(features)
            y_scaled = self.scaler_y.fit_transform(target.values.reshape(-1, 1))
        else:
            X_scaled = self.scaler_X.transform(features)
            y_scaled = self.scaler_y.transform(target.values.reshape(-1, 1))

        # make scaled data tensors
        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        self.y = torch.tensor(y_scaled, dtype=torch.float32)

    def __getitem__(self, index):
        X = self.X[index: index + self.sequence_length]
        y = self.y[index + self.sequence_length]
        return X, y

    def __len__(self):
        return len(self.X) - self.sequence_length


# --- models ---
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(LSTM, self).__init__()

        if num_layers <= 1:
            dropout = 0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

        self.lstm_norm = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_size, output_size))

    def forward(self, x):
        out, t = self.lstm(x)
        # out = self.relu(out)
        out = self.lstm_norm(out)
        out = self.fc(out[:, -1, :])
        return out


class CNN_LSTM(nn.Module):
    def __init__(self, input_size, window_size, num_filters, kernel_size, lstm_hidden_size, num_layers, dropout,
                 pool_size, output_size):
        super(CNN_LSTM, self).__init__()
        self.input_size = input_size
        self.window_size = window_size  # seq len
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.pool_size = pool_size

        if num_layers <= 1:
            self.dropout = 0
        else:
            self.dropout = dropout

        # 1d conv layer
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size)
        # relu activation func
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

        # cal output len after the conv
        self.cnn_output_len = (window_size - kernel_size + 1) // pool_size


        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size * num_filters, hidden_size=lstm_hidden_size, num_layers=num_layers,
                            dropout=self.dropout, batch_first=True)

        # Fc linear layer
        self.fc = nn.Sequential(nn.Linear(lstm_hidden_size, lstm_hidden_size),nn.ReLU(),nn.Dropout(dropout),nn.Linear(lstm_hidden_size, output_size))

    def forward(self, x):
        batch_size, window_size, num_features = x.size()
        # --- CNN layer ---
        x_permuted = x.permute(0, 2, 1)
        x_reshaped = x_permuted.reshape(batch_size * num_features, window_size)
        x_cnn_input = x_reshaped.unsqueeze(1)

        cnn_out = self.relu(self.conv1d(x_cnn_input))
        cnn_out = self.pool(cnn_out)

        # --- reshape x for lstm input ---
        cnn_out = cnn_out.view(batch_size, num_features, self.num_filters, self.cnn_output_len)
        cnn_out = cnn_out.permute(0, 3, 1, 2).contiguous()
        x_lstm_input = cnn_out.view(batch_size, self.cnn_output_len, num_features * self.num_filters)

        # --- LSTM layer ---
        lstm_out, t = self.lstm(x_lstm_input)
        # lstm_out = self.lstm_norm(lstm_out)

        # --- Fc layer ---
        out = self.fc(lstm_out[:, -1, :])
        return out


class Baseline_MLP(nn.Module):
    def __init__(self, input_size_flat, hidden_size, num_layers, output_size=1):
        super(Baseline_MLP, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size_flat, hidden_size))
        layers.append(nn.ReLU())

        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        return self.model(x_flat)


# --- Train & Val Func ---
def train_validate_model(model, train_loader, val_loader, optimizer, criterion, trial, epochs, patience, seed):
    # ---- set seed -----
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ----Training -----
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    print(f"--- Starting Training Seed: {seed} ---")

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        num_train_batches = 0

        # --- Training Loop ---
        for batch, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device).float()

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)

            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()  # update model weights

            epoch_train_loss += loss.item()
            num_train_batches += 1

        mean_train_loss = epoch_train_loss / num_train_batches
        train_losses.append(mean_train_loss)

        # -------- Validation  -----
        model.eval()
        epoch_val_loss = 0
        num_val_batches = 0

        with torch.no_grad():  # disable grad cal for validation
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device).float()

                val_output = model(X_val)
                vloss = criterion(val_output, y_val)

                epoch_val_loss += vloss.item()
                num_val_batches += 1

        mean_val_loss = epoch_val_loss / num_val_batches
        val_losses.append(mean_val_loss)

        if trial is not None:
            print(f"trial {trial.number} - epoch {epoch + 1}/{epochs} - train Loss: {mean_train_loss:.6f}, val Loss: {mean_val_loss:.6f}")
        else:
            print(f"seed {seed} epoch {epoch + 1}/{epochs} - train Loss: {mean_train_loss:.6f}, val Loss: {mean_val_loss:.6f}")

        # --- prevent overfitting ---
        if torch.isfinite(torch.tensor(mean_val_loss)) and mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered for seed {seed} after {patience} epochs without improvement.")
                break

        # --- Optuna Pruning ---
        if trial is not None:  ## if training models in make objective, not in test object
            trial.report(mean_val_loss, epoch)
            if trial.should_prune():
                print(f"Trial {trial.number} pruned by Optuna at epoch {epoch + 1}.")
                raise TrialPruned()  # stop trial

    print(f"--- Finished Training Seed: {seed}, Best Val Loss: {best_val_loss:.6f} ---")

    return best_val_loss, train_losses, val_losses, model


def make_objective(model_type, data):
    def objective(trial, model_type=model_type, data=data):
        print(f"--- Trial {trial.number} for model type: {model_type} ---")
        try:
            # ------- hyperparameter setup (common) -------
            window_size = trial.suggest_int("window_size", 5, 100)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-3)
            # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD","AdamW", "RMSprop"])
            optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop"])

            if model_type != "Baseline":
                num_layers = trial.suggest_int("num_layers", 1, 4)
                dropout = trial.suggest_float("dropout", 0.1, 0.6)

            # ------- Dataset Setup stuff -------
            train_end = int(len(data) * 0.7)
            val_end = int(len(data) * 0.85)

            # Get training data  to fit scalers on training onlu
            train_df_for_scaling = data.iloc[:train_end].copy()

            # Init and fit scalers ON TRAINING DATA ONLY
            scaler_X = RobustScaler()
            scaler_y = StandardScaler()

            features_to_scale = train_df_for_scaling.drop(columns=["Close", "Date"], errors='ignore')
            scaler_X.fit(features_to_scale)
            scaler_y.fit(np.log(train_df_for_scaling["Close"].values.reshape(-1, 1)))

            dataset = AllData(sequence_length=window_size, data=data, scaler_X=scaler_X, scaler_y=scaler_y,fit_scalers=False)

            n_sequences = len(dataset)
            train_end_seq_idx = max(0, train_end - window_size)
            val_end_seq_idx = max(0, val_end - window_size)
            test_start_seq_idx = max(0, val_end - window_size)

            #print(f"data sections len : Train [0:{train_end_seq_idx}], Val [{train_end_seq_idx}:{val_end_seq_idx}], Test [{test_start_seq_idx}:{n_sequences}]")

            # put data into subsets for training, val, and testing
            train_subset = Subset(dataset, range(0, train_end_seq_idx))
            val_subset = Subset(dataset, range(train_end_seq_idx, val_end_seq_idx))

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            # ----- Model and Hyperparam init  ----
            input_size = dataset.X.shape[1]
            output_size = 1

            if model_type == "LSTM":
                hidden_size = trial.suggest_int("hidden_size", 128, 1024,log=True)
                model = LSTM(input_size, hidden_size, output_size, num_layers, dropout)

            elif model_type == "CNN_LSTM":
                lstm_hidden_size = trial.suggest_int("lstm_hidden_size", 128, 1024, log=True)
                num_filters = trial.suggest_int("num_filters", 32, 128, log=True)
                pool_size = trial.suggest_int("pool_size", 2, 4)

                # Kernel must be <= window_size
                max_kernel_size = window_size
                if max_kernel_size < 2:  # make k min 1
                    kernel_size = 1
                else:
                    kernel_size = trial.suggest_int("kernel_size", 2, min(max_kernel_size, 5))

                if window_size - kernel_size + 1 <= 0:
                    kernel_size = window_size

                model = CNN_LSTM(input_size=input_size, window_size=window_size, num_filters=num_filters,
                                 kernel_size=kernel_size, lstm_hidden_size=lstm_hidden_size, num_layers=num_layers,
                                 dropout=dropout, pool_size=pool_size, output_size=output_size)

            elif model_type == "Baseline":
                num_layers = trial.suggest_int("num_layers", 1, 5)
                hidden_size = trial.suggest_int("hidden_size", 32, 1024, log=True)
                mlp_input_size = input_size * window_size

                model = Baseline_MLP(input_size_flat=mlp_input_size, hidden_size=hidden_size,output_size=output_size, num_layers=num_layers)
            else:
                print(f"Unknown model type: {model_type}")

            # --- Optimisers ------
            if optimizer_name == "AdamW":
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
            elif optimizer_name == "RMSprop":
                optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
            elif optimizer_name == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            else:
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # --- Loss Function ---
            criterion = nn.MSELoss()
            criterion.to(device)
            model.to(device)

            # -------- Training & Validation  -------
            best_val_loss, train_losses, val_losses, model = train_validate_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                trial=trial,
                epochs=EPOCHS_PER_TRIAL,
                patience=PATIENCE_TRAINING,
                seed=0,
            )
            return best_val_loss


        except TrialPruned as e:
            print(f"Trial {trial.number} pruned.")
            raise e

    return objective


# --- Testing best model ---
def make_test(model_type, data):
    def test_best_objective(best_trial, model_type=model_type, data=data):
        print("--- Starting Testing Phase with Best Hyperparameters ---")
        best_params = best_trial.params

        print("Best Hyperparameters found:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        window_size = best_params["window_size"]
        batch_size = best_params["batch_size"]
        
        # ---- Dataset and dataloader stuff ----
        train_end = int(len(data) * 0.7)
        val_end = int(len(data) * 0.85)

        train_df_for_scaling = data.iloc[:train_end].copy()
        scaler_X = RobustScaler()
        scaler_y = StandardScaler()

        features_to_scale = train_df_for_scaling.drop(columns=["Close", "Date"], errors='ignore')
        scaler_X.fit(features_to_scale)
        scaler_y.fit(np.log(train_df_for_scaling["Close"].values.reshape(-1, 1)))

        dataset = AllData(sequence_length=window_size, data=data, scaler_X=scaler_X, scaler_y=scaler_y,fit_scalers=False)  

        n_sequences = len(dataset)
        train_end_seq_idx = max(0, train_end - window_size)
        val_end_seq_idx = max(0, val_end - window_size)
        test_start_seq_idx = max(0, val_end - window_size)

        #print(f"Data splits for testing (sequence indices): Train [0:{train_end_seq_idx}], Val [{train_end_seq_idx}:{val_end_seq_idx}], Test [{test_start_seq_idx}:{n_sequences}]")

        train_subset = Subset(dataset, range(0, train_end_seq_idx))
        val_subset = Subset(dataset, range(train_end_seq_idx, val_end_seq_idx))
        test_subset = Subset(dataset, range(test_start_seq_idx, n_sequences))

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        # ----- make Best Model  ---------
        input_size = dataset.X.shape[1]
        output_size = 1

        if model_type == "LSTM":
            best_model_arch = LSTM(input_size, hidden_size=best_params["hidden_size"], output_size=output_size,
                                   num_layers=best_params["num_layers"], dropout=best_params["dropout"])

        elif model_type == "CNN_LSTM":
            kernel_size = best_params["kernel_size"]
            best_model_arch = CNN_LSTM(input_size=input_size, window_size=window_size,
                                       num_filters=best_params["num_filters"], kernel_size=kernel_size,
                                       pool_size=best_params["pool_size"],
                                       lstm_hidden_size=best_params["lstm_hidden_size"],
                                       num_layers=best_params["num_layers"], dropout=best_params["dropout"],
                                       output_size=output_size)

        elif model_type == "Baseline":
            mlp_input_size = input_size * window_size
            best_model_arch = Baseline_MLP(input_size_flat=mlp_input_size, hidden_size=best_params["hidden_size"],
                                           output_size=output_size, num_layers=best_params["num_layers"])

        else:
            print(f"Unknown model type: {model_type}")

        criterion = nn.MSELoss()
        criterion.to(device)

        # ----- Train and Evaluate on Test Seeds -----------
        all_seed_preds_scaled_list = []
        all_seed_targets_scaled_list = []
        all_seed_preds_inverse = [] 
        all_seed_targets_inverse = []  
        all_seed_test_losses = []  

        for seed in TEST_SEEDS:
            print(f"--- Training and Testing for Seed: {seed} ---")
            model_copy = copy.deepcopy(best_model_arch)
            model_copy.to(device)

            ## reset weights and biases of optims for each seed IMPOETANT
            optimizer_name = best_params["optimizer"]
            learning_rate = best_params["learning_rate"]
            if optimizer_name == "AdamW":
                optimizer_instance = optim.AdamW(model_copy.parameters(), lr=learning_rate)
            elif optimizer_name == "RMSprop":
                optimizer_instance = optim.RMSprop(model_copy.parameters(), lr=learning_rate)
            else:
                optimizer_instance = optim.Adam(model_copy.parameters(), lr=learning_rate)

            # Train the model
            best_val_loss, train_losses, val_losses, best_model = train_validate_model(
                model=model_copy, train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer_instance, criterion=criterion, trial=None,
                epochs=EPOCHS_BEST_TRIAL, patience=PATIENCE_TESTING, seed=seed,
            )

            # ---- Testing Model ----
            model_copy.eval()
            test_loss = 0.0
            current_seed_preds_scaled = []
            current_seed_targets_scaled = []

            with torch.no_grad():
                for X_test, y_test in test_loader:
                    X_test = X_test.to(device)
                    y_test = y_test.to(device).float()

                    test_output = model_copy(X_test)  # scaled log prices
                    loss = criterion(test_output, y_test)
                    test_loss += loss.item()

                    current_seed_preds_scaled.extend(test_output.cpu().numpy())
                    current_seed_targets_scaled.extend(y_test.cpu().numpy())

            avg_test_loss = test_loss / len(test_loader)
            all_seed_test_losses.append(avg_test_loss)
            print(f"Seed {seed} - Final Test Loss (MSE, scaled): {avg_test_loss:.6f}")

            # --- process results ----
            preds_scaled_np = np.array(current_seed_preds_scaled).reshape(-1, 1)
            targets_scaled_np = np.array(current_seed_targets_scaled).reshape(-1, 1)

            # unscale
            preds_log = scaler_y.inverse_transform(preds_scaled_np)
            targets_log = scaler_y.inverse_transform(targets_scaled_np)

            # unlog prices
            preds_prices = np.exp(preds_log).flatten()
            targets_prices = np.exp(targets_log).flatten()

            all_seed_preds_scaled_list.append(preds_scaled_np)
            all_seed_targets_scaled_list.append(targets_scaled_np)
            all_seed_preds_inverse.append(preds_prices)

            if not all_seed_targets_inverse:
                all_seed_targets_inverse.append(targets_prices)

            # --- eval ---
            print(f"---- Seed: {seed} Evaluation metrics & graphs ---")

            Evaluation.evaluate_metrics(predicted_tensor=torch.from_numpy(preds_scaled_np).float(),
                                        targets_tensor=torch.from_numpy(targets_scaled_np).float(),
                                        preds_inverse=preds_prices,
                                        targets_inverse=targets_prices)

            Evaluation.evaluate_graph(preds_inverse=preds_prices,
                                      targets_inverse=targets_prices,
                                      train_losses=train_losses, val_losses=val_losses)


            model_filename = f"{model_type}_seed{seed}_best_trial{best_trial.number}.pth"
            torch.save(model_copy, model_filename)
            print(f"Saved model for seed {seed} to {model_filename}")


        print("--- Averaged Results  ---")
        mean_test_loss = np.mean(all_seed_test_losses)
        std_test_loss = np.std(all_seed_test_losses)
        print(f"Average Test Loss (MSE, scaled): {mean_test_loss:.6f} +/- {std_test_loss:.6f}")

        all_preds_scaled_stacked = np.stack(all_seed_preds_scaled_list,axis=0)
        mean_preds_scaled_np = np.mean(all_preds_scaled_stacked, axis=0)

        mean_preds_scaled_tensor = torch.from_numpy(mean_preds_scaled_np).float()
        mean_targets_scaled_tensor = torch.from_numpy(all_seed_targets_scaled_list[0]).float()

        all_preds_inverse_stacked = np.stack(all_seed_preds_inverse, axis=0)
        mean_preds_inverse = np.mean(all_preds_inverse_stacked, axis=0)
        mean_targets_inverse = all_seed_targets_inverse[0]

        print("--- Average model metrics & graphs ---")
        Evaluation.evaluate_metrics(predicted_tensor=mean_preds_scaled_tensor,
                                    targets_tensor=mean_targets_scaled_tensor, preds_inverse=mean_preds_inverse,
                                    targets_inverse=mean_targets_inverse)
        Evaluation.evaluate_graph_mean(preds_inverse=mean_preds_inverse, targets_inverse=mean_targets_inverse)
        print("---------------------")

    return test_best_objective


if __name__ == "__main__":
    study_name = input("Enter a name for this study:")

    db_filename = f"{study_name}.db"
    storage = RDBStorage( url=f"sqlite:///{db_filename}", engine_kwargs={"connect_args": {"check_same_thread": False}})

    # --- Select Model and Data ---
    models = ["Baseline", "LSTM", "CNN_LSTM"]
    datasets = [pd.read_excel("MergedDF_googl.xlsx"), pd.read_excel("MergedDF_gold.xlsx"),pd.read_excel("MergedDF_amzn.xlsx")]
    model_choice = int(input(f"enter number of model you would like to use: 0:{models[0]}, 1:{models[1]}, 2:{models[2]}"))
    # 1 dataset_choice = input(f"enter file path of dataset to use e.g MergedDF_gold.xlsx ")
    # 1 data = pd.read_excel(dataset_choice)
    dataset_choice = int(input(f"enter number of dataset to use 0: google, 1: gold, 2: amazon"))
    selected_model_type = models[model_choice]
    data = datasets[dataset_choice]
    print(f"using {models[model_choice]} and dataset {dataset_choice}")

    # --- Optuna Study ---
    print(f"--- Starting Study: {study_name} for Model: {selected_model_type} -----")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=TPE_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=TRIALS_BEFORE_PRUNING, n_warmup_steps=EPOCHS_BEFORE_PRUNING ))

    objective_func = make_objective(selected_model_type, data)
    study.optimize(objective_func, n_trials=NUM_TRIALS)

    print("--- Optuna Optimization Finished ---")
    print(f"Study Name: {study.study_name}")
    print(f"Number of finished trials: {len(study.trials)}")
    best_trial = study.best_trial
    print("Best trial found:")
    print(f"Trial Number: {best_trial.number}")
    print(f"Value (Avg Best Val Loss): {best_trial.value:.6f}")

    test_func = make_test(selected_model_type, data)
    test_func(best_trial)

    print("--- END of Program ---")