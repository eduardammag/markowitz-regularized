"""
Funcoes auxiliares usadas apenas pela camada de visualizacao.
"""

import os

import matplotlib.pyplot as plt

from config import OUTPUT_DIR


def save_plot(fig, name, subfolder):
    """
    Salva uma figura dentro de output/<subfolder>.
    """

    # Cada familia de grafico fica em uma subpasta propria.
    folder = os.path.join(OUTPUT_DIR, subfolder)
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, name)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_model_types(results):
    """
    Extrai os tipos de modelo a partir dos nomes dos experimentos.

    Exemplo:
    random_forest_g10_l0.1 -> random_forest
    """

    types = set()

    for name in results.keys():
        # Benchmarks nao fazem parte das familias de modelos treinados.
        if name in ["equal_weight", "ibov"]:
            continue

        # Remove o sufixo de hiperparametros sem quebrar nomes com underscore.
        types.add(name.split("_g")[0])

    types_list = sorted(types)
    return types_list


def filter_models(results, keyword):
    """
    Filtra os experimentos pertencentes a uma familia de modelo.
    """

    return {k: v for k, v in results.items() if k.startswith(keyword)}
