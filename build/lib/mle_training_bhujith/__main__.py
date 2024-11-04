import mlflow
import os
from .ingest_data import (
    fetch_housing_data,
    preprocess_dataset,
    load_housing_data,
    split_dataset,
    create_datasets,
)

from .train import build_and_train_model
from .score import make_predictions

datasets_output_path = "datasets"
model_path = "models"
datasets_path_for_prediction = "datasets\\processed\\test.csv"
model_path_for_prediction = "models\\random_forest_regressor.pkl"
predictions_output_path = "results"


def main():

    with mlflow.start_run(run_name="Predict Housing Prices") as parent_run:

        fetch_housing_data(datasets_output_path)

        housing_dataset = load_housing_data(datasets_output_path)

        train_set, test_set = split_dataset(housing_dataset)

        train_set_processed = preprocess_dataset(train_set, datasets_output_path)
        test_set_processed = preprocess_dataset(test_set, datasets_output_path)

        mlflow_training_set_run_id = create_datasets(
            train_set_processed, datasets_output_path, "train.csv"
        )
        mlflow_testing_set_run_id = create_datasets(
            test_set_processed, datasets_output_path, "test.csv"
        )

        with mlflow.start_run(run_id=mlflow_training_set_run_id, nested=True):
            mlflow.log_param("parent_run_id", parent_run.info.run_id)

        model_artifact_run_id, model_params_and_metrics_run_id = build_and_train_model(
            datasets_output_path, model_path
        )

        with mlflow.start_run(run_id=model_artifact_run_id, nested=True):
            mlflow.log_param("parent_run_id", parent_run.info.run_id)

        with mlflow.start_run(run_id=model_params_and_metrics_run_id, nested=True):
            mlflow.log_param("parent_run_id", parent_run.info.run_id)

        predictions_run_id = make_predictions(
            datasets_path_for_prediction,
            model_path_for_prediction,
            predictions_output_path,
        )

        with mlflow.start_run(run_id=predictions_run_id, nested=True):
            mlflow.log_param("parent_run_id", parent_run.info.run_id)


if __name__ == "__main__":
    main()
