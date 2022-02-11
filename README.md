# Student Data Classification

The dataset used for this project is the [Student Performance Data Set](https://archive.ics.uci.edu/ml/datasets/Student+Performance) obtained from UCI Machine Learning Repository.  
There are two datasets provided, one for students performance on mathematics and another for students performance on Portugese.  

For both datasets, there are 32 independent variables and 1 target variable. The target variable is the final year grade for the corresponding subject, and the independent variables are the student grades, demographic, social and school related features collected using school reports and questionnaires.  

For this project, we will only be using the dataset corresponding to the students performance on mathematics (395 records). However, the project can be easily modified to train on the Portugese dataset by modifying the configuration files.

In the [original paper](http://www3.dsi.uminho.pt/pcortez/student.pdf), the authors trained 4 algorithms on 3 different independent variable setups and 3 different target variable setups for each dataset:

**Algorithms:**
1. Neural Networks
2. `Support Vector Machines`
3. Decision Tree
4. Random Forest

**Independent Variables Setup**
1. `With all 32 independent variables included`
2. Excluding the second period grade (31 independent variables)
3. Excluding the first and second period grade (30 independent variables)

**Target Variable Setup**
1. `Binary classification (pass-fail)`
2. Five level classification
3. Regression

This means that for each dataset a total of 3x3x4=36 models has been trained for evaluation.  
Considering the time constrain for assignment submission, we will only be training against the `Support Vector Machine` algorithm, with all `32 independent variables included` and using a `binary classification`.  

However, the project has been designed with enough flexibility to train on the other setups with minimal configurations.  

## Installation  

The project assumes that you are running on a Linux system.
If you are running on a Windows OS, proceed to [download WSL for windows](https://docs.microsoft.com/en-us/windows/wsl/install) before continuing.

Navigate into the working directory and clone the repository using

```shell
git clone https://github.com/weishengtoh/machinelearning_assignment.git
```

Once that is completed, navigate to the folder and create the conda environment using

```shell
cd machinelearning_assignment
```

```shell
conda env create -f environment.yml
```

Activate the conda environment that was created using

```shell
conda activate ml_assignment
```

## Pipeline Structure

Before running the project, it might be helpful to understand the individual pipeline components as well as their inputs, operations and artifacts.


| Pipeline   | Inputs | Operations | Artifacts |
|:-----------|:------|:-----------|:----------|
| dataloader | URI to the raw dataset zip file | Download, extract and save a copy of the raw dataset  | `student-mat.csv` |
| preprocess | `student-mat.csv` | Perform data cleaning and bins the target into binary classes  | `student-maths-clean.csv`  |
| dataval    | `student-maths-clean.csv` | Perform data validation on the cleaned data  | None |
| segregate  | `student-maths-clean.csv` | Splits the clean and validated data into train-test-splits according to a predefined split and 'stratify'  | `student_maths_train.csv`, `student_maths_val.csv`, `student_maths_test.csv`  |
| inference  | `student_maths_train.csv`, `student_maths_val.csv` | Run hyperparameter sweeps, generate model analysis and save the model outputs  | `svm_model`  |
| score      | `student_maths_test.csv`, `svm_model` | Performs a final evaluation of the trained model on the test set  | None  |  

Each of the pipeline component is defined in a seperate folder containing at least:
1. a `MLProject` file defining the entry points
2. a `conda.yml` file specifing the conda environment for the pipeline 
3. a `python script` performing the operations of the pipeline

For ease of convenience, the configurations for each of the pipeline component is defined in a yaml file in the folder `configs`.  

The main python script used to connect the pipelines are contained in `executor`, which also contains its own `MLProject` and `conda.yml` file

## Usage  
  
Navigate to the scripts folder with the command

```shell
cd ./scripts
```

Run the script that executes all the pipelines with the command  

```shell
bash execute_pipelines.sh
```

The report that is used for EDA is generated in the same folder as `train_data_profile.html`  
This can be opened using a browser to perform the EDA workflow.  

There will be a prompt from `wandb` *Weights and Biases* asking if you would like to:
1. Create an account
2. Login from an existing account
3. Run offline  

If you do not already have a `wandb` account, select the option to create an account and follow the instructions.  
This is required as the project will be uploading/retrieving all the artifacts to/from `wandb`.  

The visualisations will also be available on the dashboard at `wandb`.  

## Summary

Training metrics (also appears in the terminal)
```shell
Run Summary:
    classifier__C: 2
    classifier__gamma: 0.00195
    train_accuracy: 0.93651
    val_accuracy: 0.85
    val_roc_auc_score: 0.94987
```
The hyperparameters C and gamma were obtained via grid search C = [0.0001, 2, 4, 6, 8] and gamma = [2^-9, 2^-7, 2^-5, 2^-3, 2^-1]  
The summary shows the best hyper parameters obtained, and the corresponding test and train summary metrics.  

Test metrics (also appears in the terminal)
``shell
Run Summary:
  test_accuracy: 0.925
  test_roc_auc_score: 0.98997
```

The test perfomance appears to be better than the validation data, but this is due to the randomized nature of the train/validation/test data split.  

The visualisations of the validation and test runs are also available in the Weights & Biases dashboard.  

## Key Tools Used

- `pandas-profiling` is used to generate report on the train dataset for EDA. 
- `MLflow Projects` is used to chain together the individual workflow components.
- `conda` is used to define the environment for each individual workflow component. An alternative is to define the environment using docker containers.
- `hydra` is used for configuration files management and to reduce the clutter from overuse of `argparse`
- `Weights & Biases` is used to track and store the artifacts generated by the workflow components.
- `scikit-learn` is used to run the grid search and to define the model pipelines. This is different from the ML pipelines containing the workflow components.
- `pytest` is used to perform data validation. This approach is admittedly code-heavy and clunky however. 
- `logging` is used to generate helpful logs tracking the program while it runs.

## Next Steps

- Only deterministics tests on the data has been performed. Non-deterministic tests has not been implemented yet.
- Feature engineering has not been explored yet and feature selection only uses a simple approach by automatically removing values with 0 variance.
- Comparison of the performance against other models has not been performed yet.
- Model explainability via feature_importances or permutation_importances has not been implemented yet.
- Deployment of the model as an API using `FastAPI` or `Flask` should also be considered if the model is to be pushed to production
- Optimally, there should be another component to compare the current best model with the previous best model, and to serve the new model only if it outperforms the current best model
- Data validation is performed using `pytest` which is code-heavy and inelegant. To explore other promising tools with higher learning curve such as `great_expectations`
- To consider implementing a component to monitor the model in production (i.e. check for model drift).
