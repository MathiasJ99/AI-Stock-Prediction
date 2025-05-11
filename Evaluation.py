import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError, R2Score


def evaluate_metrics(predicted_tensor, targets_tensor,preds_inverse, targets_inverse ):
    model_metrics(predicted_tensor, targets_tensor)
    financial_metrics_new(preds_inverse,targets_inverse )

def evaluate_graph(preds_inverse, targets_inverse, train_losses, val_losses ):
    train_val_plot(train_losses, val_losses)
    target_predicted_plot(targets_inverse,preds_inverse)
    cumulative_returns_plot(targets_inverse, preds_inverse)

def evaluate_graph_mean(preds_inverse, targets_inverse):
    target_predicted_plot(targets_inverse,preds_inverse)
    cumulative_returns_plot(targets_inverse, preds_inverse)

def model_metrics(predicted_tensor, targets_tensor):
    metrics = {
        "MAE": MeanAbsoluteError(),
        "MSE": MeanSquaredError(),
        "RMSE": MeanSquaredError(squared=False),
        "MAPE": MeanAbsolutePercentageError(),
        "R2": R2Score()
    }

    print("model Metrics:")
    for name, metric in metrics.items():
        metric.update(predicted_tensor, targets_tensor)
        print(f"{name}: {metric.compute().item():.4f}")

def financial_metrics_new(preds_inverse, targets_inverse):
    preds_inverse = np.asarray(preds_inverse).flatten()
    targets_inverse = np.asarray(targets_inverse).flatten()

    # ----- Buy and Hold Return ---
    buy_hold_return_price = (targets_inverse[-1] - targets_inverse[0])
    buy_hold_return_perc = buy_hold_return_price / targets_inverse[0]
    print(f"Buy and Hold return price: {buy_hold_return_price }: percentage {buy_hold_return_perc:.4%}%")

    # ---- Directional Accuracy ---
    actual_change = targets_inverse[1:] - targets_inverse[:-1]
    predicted_change = preds_inverse[1:] - preds_inverse[:-1]

    direction_true = np.sign(actual_change)
    direction_pred = np.sign(predicted_change)

    directional_accuracy = np.mean(direction_true == direction_pred) * 100
    print(f'Directional Accuracy: {directional_accuracy:.2f}%')


    # --- Strategy Returns ---
    actual_pct_returns = (targets_inverse[1:] - targets_inverse[:-1]) / (targets_inverse[:-1] + 0.0000000000001)
    strategy_position = direction_pred
    actual_price_change = targets_inverse[1:] - targets_inverse[:-1]
    strategy_dollar_returns = actual_price_change * strategy_position
    strategy_pct_returns = actual_pct_returns * strategy_position

    cumulative_strategy_return = np.prod(1 + strategy_pct_returns) - 1
    print(f"Cumulative Strategy Return (Directional % ): {cumulative_strategy_return:.4%}")

    total_strategy_dollar_return = np.sum(strategy_dollar_returns)
    print(f"Total Strategy Return (Directional - $): ${total_strategy_dollar_return:,.2f}")

    initial_capital = 1000
    cumulative_strategy_value = initial_capital * (1 + cumulative_strategy_return)
    print(f"Cumulative Strategy Value (Starting with ${initial_capital}): ${cumulative_strategy_value:,.2f}")

    # --- Sharpe Ratio  ---
    mean_strategy_return = np.mean(strategy_pct_returns)
    std_dev_strategy_return = np.std(strategy_pct_returns)
    if std_dev_strategy_return == 0:
        sharpe_ratio = np.nan
        print(f'Sharpe Ratio : {sharpe_ratio} (Std Dev of strategy returns is zero)')
    else:
        trading_days_per_year = 252
        sharpe_ratio = (mean_strategy_return / std_dev_strategy_return) * np.sqrt(trading_days_per_year)
        print(f'Sharpe Ratio (Annualized): {sharpe_ratio:.4f}')

def train_val_plot(train_losses,val_losses):
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
    strat_daily = (targets_inverse[1:] - targets_inverse[:-1]) * np.sign(preds_inverse[1:] - preds_inverse[:-1])
    strat_cum = np.cumsum(strat_daily)

   # Buy & Hold returns
    bh_daily = targets_inverse[1:] - targets_inverse[:-1]
    bh_cum = np.cumsum(bh_daily)


    plt.figure(figsize=(12, 6))
    plt.plot(strat_cum, label='Model Returns', color='green')
    plt.plot(bh_cum, label='Buy & Hold Returns', color='blue', linestyle='--')
    plt.title("Cumulative Returns: Model Strategy vs. Buy & Hold")
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Profit/Loss")
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()

