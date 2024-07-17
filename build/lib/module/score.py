import os
import sys

# Ensure the src directory is in the PYTHONPATH for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import pickle
from datetime import datetime
import pandas as pd
import logging
from src.module.ingest_data import get_one_hot_encoder, preprocess_dataset


"""
This file contains the functions to
    Load the saved models

    Make predictions on datasets using the model
"""

if os.getenv("SPHINX_BUILD") == "true":
    logging.basicConfig(level=logging.DEBUG)
else:
    # Create and configure the logger
    logger = logging.getLogger("mle-training-logger")
    logger.setLevel(logging.DEBUG)

    # Create a file handler
    file_handler = logging.FileHandler("logs\\score.log")
    file_handler.setLevel(logging.DEBUG)  # Set the file handler logging level

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)


def load_model(model_location):
    """
    Loads a model from the specified file location.

    Args:
        model_location (str): The file path to the model.

    Returns:
        object: The loaded model.
        column_names: The columns that need to be present for the model to make prediction
    """
    try:
        with open(model_location, "rb") as file:
            data = pickle.load(file)
            model = data.get("model")
            column_names = data.get("columns")
        if model and len(column_names) > 0:
            logger.info("MODEL LOADED SUCCESSFULLY")
        return model, column_names
    except Exception as e:
        logger.info(f"ERROR WHILE LOADING MODEL {str(e)}")


def make_predictions(datasets_path, model_location, output_location="results"):
    """
    Makes predictions using a pre-trained model and saves the results to a specified output location.

    Args:
        datasets_path (str): The path to the dataset on which to make predictions.
        model_location (str): The path to the pre-trained model.
        output_location (str): The path to save the prediction results (default is "results").

    Returns:
        None
    """
    os.makedirs(output_location, exist_ok=True)
    output_filepath = os.path.join(
        output_location, f"predictions_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )

    model, column_names = load_model(model_location)

    if os.path.exists(datasets_path):
        df = pd.read_csv(datasets_path)
        logger.info("FILE FOUND FOR MAKING PREDICTIONS")

        for col in column_names:
            if col not in df.columns:
                raise ValueError(f"{col} not found in dataframe")

            if "ocean_proximity" in df.columns:
                ohe = get_one_hot_encoder(
                    datasets_path,
                    "ocean_proximity",
                    "one_hot_encoder_ocean_proximity.pkl",
                )
                tmp = pd.DataFrame(
                    ohe.transform(df[["ocean_proximity"]]).toarray(),
                    columns=ohe.get_feature_names_out(),
                )
                df = pd.concat([df, tmp], axis=1)
                df = df.drop(["ocean_proximity"], axis=1)
    else:
        logger.info(
            "NO VALID FILE FOUND FOR MAKING PREDICTIONS. MAKING PREDICTIONS ON TESTING SET"
        )
        df = pd.read_csv("datasets\\processed\\test.csv")

    if "median_house_value" in df.columns:
        df = df.drop("median_house_value", axis=1)

    predictions = model.predict(df)
    predictions_df = pd.DataFrame(predictions, columns=["predictions"])
    predictions_df.to_csv(output_filepath, index=False)
    if os.path.exists(output_filepath):
        logger.info("PREDICTIONS CSV FILE SAVED SUCCESSFULLY")
    else:
        logger.info("FAILURE IN SAVING PREDICTIONS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using models")
    parser.add_argument(
        "--datasets_path",
        type=str,
        required=True,
        help="Complete filepath for the dataset to make prediction",
    )
    parser.add_argument(
        "--model_location",
        type=str,
        required=True,
        help="Complete filepath for the saved model",
    )
    parser.add_argument(
        "--output_location",
        type=str,
        required=False,
        help="Folder for storing the predictions",
    )
    args = parser.parse_args()

    if args.output_location:
        output_location = args.output_location
    else:
        output_location = "results"

    make_predictions(args.datasets_path, args.model_location, output_location)
