import pandas as pd
import numpy as np
import os

"""
This file contains the functions to

    Generate sample datasets for running unit tests
"""


def generate_dataset_for_testing():
    """
    Generates a sample dataset for testing purposes by loading a CSV file,
    sampling 1000 rows, and introducing NaN values randomly.

    The function performs the following steps:
    1. Loads the dataset from the specified CSV file path.
    2. Samples 1000 rows from the dataset.
    3. Resets the index of the sampled dataset and drops the old index column.
    4. Introduces NaN values in 5% of the data for each column randomly.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the sampled data with
                      random NaN values.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
    """
    csv_path = os.path.join("datasets", "raw", "housing.csv")
    if os.path.exists:
        df = pd.read_csv(csv_path)
        df_sample = df.sample(1000)
        df_sample = df_sample.reset_index().drop("index", axis=1)

        # Introduce NaN values randomly
        nan_count = int(len(df_sample) * 0.05)

        for col in df_sample.columns:
            nan_indices = np.random.choice(
                df_sample.index, size=nan_count, replace=False
            )
            df_sample.loc[nan_indices, col] = np.nan

        return df_sample
    else:
        print("ERROR IN ACCESSING FILES")
