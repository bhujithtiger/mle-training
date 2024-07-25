import argparse
import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
import urllib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import (
    StratifiedShuffleSplit,
)
import logging
import pickle
import sys
import mlflow

remote_server_uri = "http://localhost:8000"
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment("PredictingHousingPrices")

# Ensure the src directory is in the PYTHONPATH for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.module.helper import CombinedAttributesAdder

"""This file contains the functions

    To download datasets and load it in a dataframe

    To preprocess, split the dataset and store it for training purpose
"""

if os.getenv("SPHINX_BUILD") == "true":
    logging.basicConfig(level=logging.DEBUG)
else:
    # Create and configure the logger
    logger = logging.getLogger("mle-training-logger")
    logger.setLevel(logging.DEBUG)

    # Create a file handler
    file_handler = logging.FileHandler("logs\\ingest_data.log")
    file_handler.setLevel(logging.DEBUG)  # Set the file handler logging level

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_path):
    """Fetches housing data from a specified URL, extracts the content of a TAR file,
    and saves it to the given housing path.

    Args:
        housing_path (str): The path where the housing data will be stored.
    """
    os.makedirs(os.path.join(housing_path, "raw"), exist_ok=True)
    tgz_path = os.path.join(housing_path, "raw", "housing.tgz")
    urllib.request.urlretrieve(HOUSING_URL, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=os.path.join(housing_path, "raw"))
    housing_tgz.close()
    if os.path.exists(tgz_path):
        logger.info("TARFILE DOWNLOADED AND CSV FILES EXTRACTED SUCCESSFULLY")
    else:
        logger.info("ERROR WHILE EXECUTING FROM TARFILE")


def load_housing_data(housing_path):
    """Loads the housing data from a CSV file located at the given housing path.

    Args:
        housing_path (str): The path where the housing CSV file is stored.

    Returns:
        pd.DataFrame: A DataFrame containing the housing data.
    """
    try:
        csv_path = os.path.join(housing_path, "raw", "housing.csv")
        logger.info("CSV FILE LOADED SUCCESSFULLY")
        return pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"ERROR WHILE LOADING CSV {str(e)}")


def split_dataset(housing):
    """Splits the housing DataFrame into training and test sets while preserving
    the distribution of the income_cat column.

    Args:
        housing (pd.DataFrame): The housing data to be split.

    Returns:
        tuple: A tuple containing the training and test sets as DataFrames.
    """
    try:
        housing = housing.dropna()

        housing = housing.reset_index().drop("index", axis=1)

        housing["income_cat"] = pd.cut(
            housing["median_income"],
            bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
            labels=[1, 2, 3, 4, 5],
        )

        housing["income_cat"] = (
            housing["income_cat"].cat.add_categories("missing").fillna("missing")
        )

        # Splits the housing DataFrame into training and test sets while preserving the distribution of the income_cat column.
        # income_cat is a categorical variable that we want to maintain the same proportion in both training and test sets.
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        strat_train_set.drop("income_cat", axis=1, inplace=True)
        strat_test_set.drop("income_cat", axis=1, inplace=True)

        # Separate features into numerical and categorical columns
        num_cols = strat_train_set.select_dtypes(include=["number"]).columns
        cat_cols = strat_train_set.select_dtypes(include=["object", "category"]).columns

        # Define the imputer for numerical columns
        num_imputer = SimpleImputer(strategy="median")

        # Define the imputer for categorical columns
        cat_imputer = SimpleImputer(strategy="most_frequent")

        num_imputer.fit(strat_train_set[num_cols])
        cat_imputer.fit(strat_train_set[cat_cols])

        strat_train_set_num_cols = pd.DataFrame(
            num_imputer.transform(strat_train_set[num_cols]), columns=num_cols
        )
        strat_train_set_cat_cols = pd.DataFrame(
            cat_imputer.transform(strat_train_set[cat_cols]), columns=cat_cols
        )

        strat_train_set = pd.concat(
            [strat_train_set_num_cols, strat_train_set_cat_cols], axis=1
        )

        strat_test_set_num_cols = pd.DataFrame(
            num_imputer.transform(strat_test_set[num_cols]), columns=num_cols
        )
        strat_test_set_cat_cols = pd.DataFrame(
            cat_imputer.transform(strat_test_set[cat_cols]), columns=cat_cols
        )

        strat_test_set = pd.concat(
            [strat_test_set_num_cols, strat_test_set_cat_cols], axis=1
        )

        logger.info("DATASET SPLIT SUCCESSFULLY")

        return strat_train_set, strat_test_set

    except Exception as e:
        logger.error(f"ERROR WHILE SPLITTING DATASET {str(e)}")


def get_one_hot_encoder(output_path, col_name, filename):
    """Retrieves or creates a one-hot encoder for the specified column and saves it as a pickle file.

    Args:
        output_path (str): The path to load the housing data.
        col_name (str): The column name for which to create the encoder.
        filename (str): The filename for saving the encoder.

    Returns:
        OneHotEncoder: The one-hot encoder for the specified column.
    """
    os.makedirs(os.path.join("encoders"), exist_ok=True)
    complete_path = os.path.join("encoders", filename)
    try:
        if os.path.exists(complete_path):
            with open(complete_path, "rb") as file:
                ohe = pickle.load(file)
            return ohe
        else:
            ohe = OneHotEncoder(drop="first")
            df = load_housing_data(output_path)
            ohe.fit(df[[col_name]])
            with open(complete_path, "wb") as file:
                pickle.dump(ohe, file)
            return ohe
    except Exception as e:
        logger.error(f"ERROR WHILE LOADING ONE HOT ENCODER {str(e)}")


def preprocess_dataset(set_, output_path):
    """Preprocesses the dataset by creating new features and applying one-hot encoding to categorical columns.

    Args:
        set_ (pd.DataFrame): The dataset to preprocess.
        output_path (str): The path to load the housing data for encoding.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    try:
        attr_adder = CombinedAttributesAdder()
        set_ = attr_adder.transform(set_)

        cat_cols = list(set_.select_dtypes(include=["object", "category"]).columns)

        for col in cat_cols:
            encoder = get_one_hot_encoder(
                output_path, col, f"one_hot_encoder_{col}.pkl"
            )
            tmp = pd.DataFrame(
                encoder.transform(set_[[col]]).toarray(),
                columns=encoder.get_feature_names_out(),
            )
            set_ = pd.concat([set_, tmp], axis=1)
            set_ = set_.drop([col], axis=1)

        logger.info("DATSET PREPROCESSED SUCCESSFULLY")

        return set_

    except Exception as e:
        logger.error(f"ERROR WHILE PROCESSING DATASET {str(e)}")


def create_datasets(df, output_path, filename):
    """Saves the processed dataset to a specified file in the output path.

    Args:
        df (pd.DataFrame): The dataset to be saved.
        output_path (str): The path where the processed dataset will be saved.
        filename (str): The name of the file to save the dataset.

    Returns:
        None
    """
    os.makedirs(os.path.join(output_path, "processed"), exist_ok=True)

    file_path = os.path.join(output_path, "processed", filename)

    df.to_csv(file_path, index=False)

    if os.path.exists(file_path):
        logger.info(f"DATASET SPLIT, PROCESSED AND STORED AT {file_path}")

        with mlflow.start_run(
            run_name=f"Data Ingestion {filename}", nested=True
        ) as run:
            mlflow.log_artifact(file_path)

        run_id = run.info.run_id

        mlflow.end_run()
        print(f"ingest data {run_id}")
        return run_id

    else:
        logger.info("ERROR WHILE STORING FILE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data and create datasets")
    parser.add_argument(
        "--output_path", type=str, required=True, help="Output folder path for datasets"
    )
    args = parser.parse_args()

    fetch_housing_data(args.output_path)

    housing_dataset = load_housing_data(args.output_path)

    train_set, test_set = split_dataset(housing_dataset)

    train_set_processed = preprocess_dataset(train_set, args.output_path)

    test_set_processed = preprocess_dataset(test_set, args.output_path)

    mlflow_training_set_run_id = create_datasets(
        train_set_processed, args.output_path, "train.csv"
    )

    mlflow_testing_set_run_id = create_datasets(
        test_set_processed, args.output_path, "test.csv"
    )
