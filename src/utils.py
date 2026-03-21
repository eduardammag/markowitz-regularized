import os 
import numpy as np
import matplotlib as plt
from scipy import stats
from config import OUTPUT_DIR

def sharpe_ratio(returns, rf=0.02):
    excess = returns - rf/252
    return np.mean(excess) / np.std(excess)

def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return dd.min()



def compute_drawdown(returns):
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    return (cum - peak) / peak


def save_plot(fig, name, subfolder):
    folder = os.path.join(OUTPUT_DIR, subfolder)
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, name)
    fig.savefig(path)
    plt.close(fig)

def get_model_types(results):
    """
    Extrai automaticamente os tipos de modelo:
    ex: lasso, ridge, elastic
    """
    types = set()

    for name in results.keys():
        if name in ["equal_weight", "ibov"]:
            continue
        types.add(name.split("_")[0])

    return sorted(types)


def filter_models(results, keyword):
    return {k: v for k, v in results.items() if k.startswith(keyword)}




def diebold_mariano(e1, e2, h=1):

    d = (e1 ** 2) - (e2 ** 2)
    mean_d = np.mean(d)

    gamma = [np.cov(d[:-lag], d[lag:])[0,1] for lag in range(1, h)]
    var_d = np.var(d) + 2 * np.sum(gamma)

    DM = mean_d / np.sqrt(var_d / len(d))

    p_value = 2 * (1 - stats.norm.cdf(abs(DM)))

    return DM, p_value