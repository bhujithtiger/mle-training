import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import randint
import pickle
import logging

"""
This file contains the functions to

    Load the datasets from the specified location

    Build and train machine learning models using the best hyperparameters

    Save the trained model for later use
"""

if os.getenv("SPHINX_BUILD") == "true":
    logging.basicConfig(level=logging.DEBUG)
else:
    # Create and configure the logger
    logger = logging.getLogger("mle-training-logger")
    logger.setLevel(logging.DEBUG)

    # Create a file handler
    file_handler = logging.FileHandler("logs\\train.log")
    file_handler.setLevel(logging.DEBUG)  # Set the file handler logging level

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)


def save_model(output_path, model, filename="model.pkl"):
    """
    Saves a given model to the specified output path.

    Args:
        output_path (str): The path to save the model.
        model (object): The model to be saved.
        filename (str): The filename for the saved model (default is "model.pkl").
    """
    filepath = os.path.join(output_path, filename)
    os.makedirs(output_path, exist_ok=True)
    try:
        with open(filepath, "wb") as file:
            pickle.dump(
                {"model": model, "columns": list(model.feature_names_in_)}, file
            )
        logger.info(f"SAVED MODEL SUCCESSFULLY AT {filepath}")
    except Exception as e:
        logger.error(f"ERROR WHILE SAVING MODEL {str(e)}")


def load_datasets_and_prepare_for_training(datasets_location):
    """
    Loads the training and test datasets from the specified location and prepares them for training.

    Args:
        datasets_location (str): The location where the datasets are stored.

    Returns:
        tuple: A tuple containing the features and target variables for both training and test sets.
    """
    try:
        train_file_path = os.path.join(datasets_location, "processed", "train.csv")
        test_file_path = os.path.join(datasets_location, "processed", "test.csv")

        train = pd.read_csv(train_file_path)
        test = pd.read_csv(test_file_path)

        x_train = train.drop("median_house_value", axis=1)
        y_train = train[["median_house_value"]]

        x_test = test.drop("median_house_value", axis=1)
        y_test = test[["median_house_value"]]

        logger.info("SPLIT DATASET SUCESSFULLY")

        return x_train, y_train, x_test, y_test

    except Exception as e:
        logger.error(f"ERROR WHILE SPLITTING DATASET {str(e)}")


def find_best_params(x, y, no_of_iterations=5):
    """
    Performs hyperparameter tuning to find the best parameters for a RandomForestRegressor.

    Args:
        x (pd.DataFrame): The feature variables for training.
        y (pd.DataFrame): The target variable for training.
        no_of_iterations (int): The number of iterations for randomized search (default is 5).

    Returns:
        dict: The best hyperparameters found during the tuning process.
    """
    logger.info("STARTED HYPERPARAMETER TUNING")

    try:
        param_distribs = {
            "n_estimators": randint(low=1, high=200),
            "max_features": randint(low=1, high=8),
        }

        forest_reg = RandomForestRegressor(random_state=42)

        rnd_search = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_distribs,
            n_iter=no_of_iterations,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        rnd_search.fit(x, y)
        cvres = rnd_search.cv_results_
        params_with_lowest_rmse = sorted(
            dict(zip(np.sqrt(-(cvres["mean_test_score"])), cvres["params"])).items(),
            key=lambda x: x[0],
        )[0]

        logger.info(
            f"Parameters with lowest RMSE {params_with_lowest_rmse[1]} RMSE SCORE {params_with_lowest_rmse[0]}"
        )

        # params_with_lowest_rmse[0] - root_mean_squared_error
        # params_with_lowest_rmse[1] - parameters

        return params_with_lowest_rmse[1]

    except Exception as e:
        logger.error(f"ERROR WHILE HYPERPARAMETER TUNING {str(e)}")


def build_and_train_model(datasets_location, model_output_location):
    """
    Builds, trains, and saves a RandomForestRegressor model using the training dataset.

    Args:
        datasets_location (str): The location where the datasets are stored.
        model_output_location (str): The location to save the trained model.
    """
    x_train, y_train, x_test, y_test = load_datasets_and_prepare_for_training(
        datasets_location
    )

    params_with_lowest_rmse = find_best_params(x_train, y_train)

    logger.info("STARTED MODEL TRAINING")

    try:
        randomForestRegressor = RandomForestRegressor(
            **params_with_lowest_rmse, random_state=42
        )
        randomForestRegressor.fit(x_train, y_train)

        pred_train = randomForestRegressor.predict(x_train)
        pred_test = randomForestRegressor.predict(x_test)

        logger.info(
            f"Training set RMSE {np.sqrt(mean_squared_error(y_train, pred_train))}"
        )
        logger.info(
            f"Testing set RMSE {np.sqrt(mean_squared_error(y_test, pred_test))}"
        )

        save_model(
            model_output_location,
            randomForestRegressor,
            filename="random_forest_regressor.pkl",
        )

    except Exception as e:
        logger.error(f"ERROR WHILE BUILDING AND TRAINING MODEL {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build, train and save models")
    parser.add_argument(
        "--datasets_location", type=str, required=True, help="Folder path for datasets"
    )
    parser.add_argument(
        "--model_location",
        type=str,
        required=True,
        help="Output folder path to save models",
    )
    args = parser.parse_args()

    build_and_train_model(args.datasets_location, args.model_location)
