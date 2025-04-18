import matplotlib as plt
import numpy as np
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError, R2Score
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data


def evaluate(predicted_tensor, targets_tensor,preds_inverse, targets_inverse, train_losses, val_losses ):
    #METRICS
    evaluation_metrics(predicted_tensor, targets_tensor)
    financial_metrics(preds_inverse,targets_inverse )

    ##PLOTS
    train_val_plot(train_losses, val_losses)
    target_predicted_plot(targets_inverse,preds_inverse)
    cumulative_returns_plot(targets_inverse, preds_inverse)
    residual_plots(targets_inverse, preds_inverse)

def evaluation_metrics(predicted_tensor, targets_tensor):
    # ======= Evaluation METRICS =========
    metrics = {
        "MAE": MeanAbsoluteError(),
        "MSE": MeanSquaredError(),
        "RMSE": MeanSquaredError(squared=False),
        "MAPE": MeanAbsolutePercentageError(),
        "R2": R2Score()
    }

    ##Network metrics
    print("\nTest Metrics:")
    for name, metric in metrics.items():
        metric.update(predicted_tensor, targets_tensor)
        print(f"{name}: {metric.compute().item():.4f}")

def financial_metrics(preds_inverse, targets_inverse):
    # Directional accuracy
    direction_true = np.sign(targets_inverse[1:] - targets_inverse[:-1])  # Actual direction (up/down)
    direction_pred = np.sign(preds_inverse[1:] - preds_inverse[:-1])  # Predicted direction (up/down)
    directional_accuracy = np.mean(direction_true == direction_pred) * 100  # Percentage correct
    print('Directional Accuracy:', directional_accuracy, '%')

    # Sharpe ratio
    returns = (targets_inverse[1:] - targets_inverse[:-1]) * direction_pred  # Profit if correct direction
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized Sharpe
    print('Sharpe Ratio (Annualized):', sharpe_ratio)


def train_val_plot(train_losses,val_losses):
    # 1. Plot Training & Validation Loss (unchanged)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs. Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def target_predicted_plot(targets_inverse,preds_inverse):
    #2. Plot Test Predictions vs. Actual Close (INVERSE-TRANSFORMED)
    plt.figure(figsize=(12, 6))
    plt.plot(targets_inverse, label='Actual Close Price', color='blue', alpha=0.7)
    plt.plot(preds_inverse, label='Predicted Close Price', color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Time Steps (Test Set)')
    plt.ylabel('Close Price')
    plt.title('Model Predictions vs. Actual Close (Inverse-Transformed)')
    plt.legend()
    plt.grid(True)
    plt.show()

def cumulative_returns_plot(targets_inverse, preds_inverse):
    returns = (targets_inverse[1:] - targets_inverse[:-1]) * np.sign(preds_inverse[1:] - preds_inverse[:-1])
    cumulative_returns = np.cumsum(returns)
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns, label='Strategy Returns', color='green')
    plt.title("Cumulative Returns from Directional Predictions")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Profit/Loss")
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.grid()
    plt.show()

def residual_plots(targets_inverse,preds_inverse):
    # 4 Residuals plot (prediction errors)
    residuals = targets_inverse - preds_inverse
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(residuals)), residuals, alpha=0.5, color='purple')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Prediction Residuals (Actual - Predicted)")
    plt.xlabel("Time")
    plt.ylabel("Residual Error")
    plt.grid()
    plt.show()
