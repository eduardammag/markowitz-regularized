import pandas as pd
import numpy as np
from src.utils import sharpe_ratio, max_drawdown

def generate_report(results_dict):

    rows = []

    for name, data in results_dict.items():

        returns = data["returns"]

        row = {
            "Model": name,
            "Sharpe": sharpe_ratio(returns),
            "Drawdown": max_drawdown(returns),
            "Return": np.prod(1 + returns) - 1
        }

        if "mse" in data:
            row["MSE"] = data["mse"]
            row["MAE"] = data["mae"]
            row["Direction"] = data["direction"]

        rows.append(row)

    df = pd.DataFrame(rows)
    return df