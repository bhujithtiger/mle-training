import pytest
import numpy as np
import sys
import os

# Ensure the src directory is in the PYTHONPATH for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from module.ingest_data import split_dataset, preprocess_dataset
from module.helper import generate_dataset_for_testing

"""
This file contains the functions to

    Build unit tests to test certain functions in the module
"""


# Generate 3 tests
@pytest.mark.parametrize("run", range(3))
def test_split_dataset(run):
    """
    Description:
    Tests the split_dataset function to ensure that the dataset is properly split into training and testing sets, and that the resulting datasets do not contain the "income_cat" column.

    Parameters:
    run (int): The current test run index, used to repeat the test multiple times.

    Test Scenarios:
    Ensure the training set is not empty.
    Ensure the testing set is not empty.
    Ensure the "income_cat" column is not present in both the training and testing sets.
    Ensure both the training and testing sets have the same columns.
    """
    housing = generate_dataset_for_testing()
    train_set, test_set = split_dataset(housing)
    assert not train_set.empty
    assert not test_set.empty
    assert "income_cat" not in train_set.columns
    assert "income_cat" not in test_set.columns
    assert set(train_set.columns) == set(test_set.columns)


@pytest.mark.parametrize("run", range(3))
def test_preprocess_dataset(run):
    """
    Description:
    Tests the preprocess_dataset function to ensure that the dataset is correctly preprocessed by adding new features and removing the "ocean_proximity" column.

    Parameters:
    run (int): The current test run index, used to repeat the test multiple times.

    Test Scenarios:
    Ensure the "rooms_per_household" feature is present in the preprocessed dataset.
    Ensure the "bedrooms_per_room" feature is present in the preprocessed dataset.
    Ensure the "population_per_household" feature is present in the preprocessed dataset.
    Ensure the "ocean_proximity" column is not present in the preprocessed dataset.
    """
    housing = generate_dataset_for_testing()

    set, _ = split_dataset(housing)

    set = preprocess_dataset(set, "datasets")

    assert "rooms_per_household" in set.columns
    assert "bedrooms_per_room" in set.columns
    assert "population_per_household" in set.columns
    assert "ocean_proximity" not in set.columns
