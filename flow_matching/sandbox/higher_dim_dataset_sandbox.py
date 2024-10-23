"""
Sandbox script to explore higher dim dataset in

Tensor-Train Density Estimation
https://arxiv.org/pdf/2108.00089 p 8

Masked Autoregressive Flow for Density Estimation
https://arxiv.org/pdf/1705.07057 p 11 sec D.2

Viz for HighDim Data
https://www.geeksforgeeks.org/techniques-for-visualizing-high-dimensional-data/
https://naturalistic-data.org/content/hypertools.html
https://builtin.com/data-science/tsne-python
"""
import pandas as pd
from ucimlrepo import fetch_ucirepo
from loguru import logger


def preprocess_uci_power_dataset(df: pd.DataFrame):
    """
        Quote from Paper "Masked Autoregressive Flow for Density Estimation" p 11 section D.2
        https://arxiv.org/pdf/1705.07057
            POWER. The POWER dataset [1] contains measurements of electric power consumption in a
            household over a period of 47 months. It is actually a time series but was treated as if each example
            were an i.i.d. sample from the marginal distribution. The time feature was turned into an integer for
            the number of minutes in the day, and then uniform random noise was added to it. The date was
            discarded, along with the global reactive power parameter, which seemed to have many values at
            exactly zero, which could have caused arbitrarily large spikes in the learnt distribution. Uniform
            random noise was added to each feature in the interval [0, i
            ], where i
            is large enough to ensure that
            with high probability there are no identical values for the i
            th feature but small enough to not change
            the data values significantly
        """
    pass


def experiment_uci_power_dataset():
    # https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

    # fetch dataset
    individual_household_electric_power_consumption = fetch_ucirepo(id=235)

    # data (as pandas dataframes)
    X = individual_household_electric_power_consumption.data.features
    y = individual_household_electric_power_consumption.data.targets

    # metadata
    print(individual_household_electric_power_consumption.metadata)

    # variable information
    print(individual_household_electric_power_consumption.variables)
    print(X.describe())


if __name__ == "__main__":
    experiment_uci_power_dataset()
