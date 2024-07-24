# Median housing value prediction

Objective - We will be predicting the housing prices using machine learning models

## Datasets

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

## About the files and folders

    - src
        - This package contains the files to help us with our core objective of predicting housing prices

        - module

            - ingest_data.py
                - Run this file to download the tar file from the internet, extract the csv file and load it in the dataframe followed by preprocessing and splitting it into training and testing sets

            - train.py
                - Run this file to build and train the Random Forest Regressor model followed by saving the model.

            - score.py
                - Run this file to make predictions using the model saved during the previous step.

        - tests

            - tests.py
                - Run this file to conduct unit tests to validate the functionalities of some functions in the src package.

    - logs
        - This folder contains the logs generated as part of achieving our objective of predicting housing prices.

    - models
        - This folder contains the models generated during the model building process saved as pickle files

    - encoders
        - This folder contains the one hot encoders used for transforming the categorical columns

    - results
        - This folder contains the predictions generated as part of the scoring process and saved as csv files.

## To run the script
 - Install Miniconda
 - Create a conda environment
 - Activate the environment
 - Install the required packages
 - Create env.yml file
 - Execute the python script

## Corresponding commands

```shell
conda create --name mle-dev

conda activate mle-dev

conda install <package-names>

conda env export > env.yml

python <script-name>
```

## To create docs

**sphinx-quickstart** command creates the setup files for generation

**sphinx-apidoc -o docs src** command creates the rst files in docs folder, for packages in the src folder. It is important that each module within the src package has **__init__.py** file.

**src\make.bat html** command generates the html files for the packages

```shell
mkdir docs

pip install sphinx sphinx-autobuild sphinx_rtd_theme

cd docs

sphinx-quickstart

sphinx-apidoc -o docs src

docs\make.bat html
```

## To convert into package

Create setup.py file and run the following commands

twine package is used to verify the package installation and upload to PyPI

```shell
pip install wheel twine

python setup.py sdist

python setup.py bdist_wheel

twine check dist/*

twine upload dist/*
```







