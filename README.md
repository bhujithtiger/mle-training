```markdown
# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techniques are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To run the script
 - Install Miniconda
 - Create a conda environment
 - Activate the environment 
 - Install the required packages
 - Create env.yml file 
 - Execute the python script

## Corresponding commands
    
```shell
conda create --name mle-dev python=3.9

conda activate mle-dev

conda install <package-names>

conda env export > env.yml

python <script-name>
