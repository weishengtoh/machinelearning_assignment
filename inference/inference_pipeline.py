'''
Define the pipeline that is used to run hyperparameter search and training of the model.


This pipeline takes in the student_maths_train.csv train data artifact and 
student_maths_val.csv validation artifact from Weights & Biases. 

Grid search will be performed according to the parameters defined in the inference_config
file, and the best performing model will be saved into Weights & Biases.

The training artifacts (ROC curve, confusion matrix, summary metrics) will be generated
and stored in Weights & Biases for tracking.

The best performing model will be used by the scoring_pipeline to perform a final evaluation 
on the test data.
'''

import logging
import os
import tempfile
from tokenize import group

import hydra
import mlflow.sklearn
import pandas as pd
import wandb
from mlflow.models import infer_signature
from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils import estimator_html_repr

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@hydra.main(config_path=os.path.join(os.pardir, 'configs'),
            config_name="inference_config")
def start_pipeline(model_config: DictConfig):

    model_config = OmegaConf.to_container(model_config)

    # Load the model configs
    model_params = model_config['inference_pipeline']['model_params']
    data_artifacts = model_config['data']
    model_features = model_config['inference_pipeline']['features']
    grid_params = model_config['inference_pipeline']['grid_search']

    # Initialise the W&B run
    run = wandb.init(project=model_config['main']['project_name'],
                     group=model_config['main']['experiment_name'],
                     job_type=model_config['main']['job_type'])

    # Download train artifact from W&B and split into X and y
    logger.info('Download train Artifact from W&B')
    train_data_path = run.use_artifact(data_artifacts['train_data']).file()

    train_df = pd.read_csv(train_data_path, header=0)
    train_X = train_df.copy()
    train_y = train_X.pop(model_features['target_var'])

    # Download validation artifact from W&B and split into X and y
    logger.info('Download val Artifact from W&B')
    val_data_path = run.use_artifact(data_artifacts['val_data']).file()

    val_df = pd.read_csv(val_data_path, header=0)
    val_X = val_df.copy()
    val_y = val_X.pop(model_features['target_var'])

    # Generate the sklearn pipeline
    logger.info('Setting up inference pipeline')
    wandb.config.update(model_config, allow_val_change=True)
    pipeline = get_inference_pipeline(model_features, model_params)

    target_labels = model_features['target_bins_labels']

    # Run GridSearch on pipeline and obtain the best model
    logger.info('Performing GridSearch')
    grid_search = GridSearchCV(pipeline, grid_params)
    grid_search.fit(train_X, train_y)

    model = grid_search.best_estimator_

    # Fit and evaluate the model
    logger.info('Fitting the model')
    model.fit(train_X, train_y)

    logger.info('Evaluating the model')
    val_pred = model.predict(val_X)
    val_probas = model.predict_proba(val_X)

    train_pred = model.predict(train_X)

    # Log the summary metrics and the parameters obtained from grid search
    run.summary['train_accuracy'] = accuracy_score(train_y, train_pred)
    run.summary['val_accuracy'] = accuracy_score(val_y, val_pred)
    run.summary['val_roc_auc_score'] = roc_auc_score(
        val_y,
        val_probas[:, 1],
        multi_class=model_params['decision_function_shape'])

    for params in grid_params:
        run.summary[params] = grid_search.best_params_[params]

    # Export model into W&B
    if model_config['model_name'] != 'None':
        export_model(model, run, val_X, val_pred,
                     model_config['artifact_export'],
                     model_config['model_name'])

    cwd_path = hydra.utils.get_original_cwd(
    )  # Need to use this command to retrieve cwd when using hydra

    # Save a visualisation of the pipeline as html file
    with open(os.path.join(cwd_path, 'pipeline_fig.html'), 'w') as file:
        file.write(estimator_html_repr(model))

    # Plot the ROC curve, precision-recall curve and confusion matrix in W&B
    wandb.sklearn.plot_confusion_matrix(val_y, val_pred, target_labels)
    wandb.sklearn.plot_precision_recall(val_y, val_probas, target_labels)
    wandb.sklearn.plot_roc(val_y, val_probas, target_labels)

    # Log the classification report
    run.summary['classification_report'] = classification_report(
        val_y, val_pred, target_names=target_labels, output_dict=True)

    # Log into W&B the training pipeline figure
    run.log({
        'training_pipeline':
            wandb.Html(open(os.path.join(cwd_path, 'pipeline_fig.html')))
    })

    # Finish the wandb run
    wandb.finish()


def get_inference_pipeline(model_features, model_params) -> Pipeline:

    # Obtain the list of numerical, nominal and ordinal features
    numerical_var = model_features['numerical_var']
    nominal_var = model_features['nominal_var']
    ordinal_var = model_features['ordinal_var']

    # Preprocessing pipeline for numerical data
    numerical_prepro = Pipeline(
        steps=[('imputer',
                SimpleImputer(strategy='median')), ('scaler',
                                                    StandardScaler())])

    # Preprocessing pipeline for nominal data
    nominal_prepro = OneHotEncoder(drop='first')

    # Preprocessing pipeline for ordinal data
    ordinal_prepro = OrdinalEncoder()

    # Combine preprocessing pipelines
    prepro_pipeline = ColumnTransformer(
        transformers=[('numerical_transform', numerical_prepro, numerical_var),
                      ('nominal_transform', nominal_prepro, nominal_var),
                      ('ordinal_transform', ordinal_prepro, ordinal_var)],
        remainder='drop'
    )  # Drop all the variables that are not defined in the config file

    # Remove variables based on variance threshold
    var_thresholder = VarianceThreshold(model_features['variance_thresh'])

    # Define the model using parameters in the config file
    clf = SVC(**model_params)

    # Combine the pipelines, variance threshold and classifier into a single pipeline
    inference_pipe = Pipeline(
        steps=[('preprocessing',
                prepro_pipeline), ('variance_thresholder',
                                   var_thresholder), ('classifier', clf)])

    return inference_pipe


def export_model(pipe, run, val_X, val_pred, artifact, model_name):

    signature = infer_signature(val_X, val_pred)
    cwd_path = hydra.utils.get_original_cwd()
    export_path = os.path.join(*[cwd_path, os.path.pardir, 'models'])

    with tempfile.TemporaryDirectory() as temp:

        export_path = os.path.join(temp, model_name)

        # Save the entire pipeline as a model, minimizing risk of training-serving skew
        mlflow.sklearn.save_model(
            pipe,  # sklearn pipeline
            export_path,  # path to store the pipeline
            signature=signature,  # the schema for input and output
            serialization_format=mlflow.sklearn.
            SERIALIZATION_FORMAT_CLOUDPICKLE,
            # include a few examples of valid input as reference
            input_example=val_X.iloc[:5])

        # Upload the saved model as an artifact to W&B
        artifact = wandb.Artifact(artifact,
                                  type='model',
                                  description='Model pipeline')

        artifact.add_dir(export_path)
        run.log_artifact(artifact)

        artifact.wait()


if __name__ == "__main__":

    start_pipeline()
