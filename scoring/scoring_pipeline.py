'''
Define the pipeline that is used to perform a final evaluation of the trained model.

This pipeline takes in the svm_model model artifact from Weights & Biases
and computes the metrics on the test data 

The test artifacts (ROC curve, confusion matrix, summary metrics) will be generated
and stored in Weights & Biases for tracking.

'''

import os
import logging
import pandas as pd
import wandb
import mlflow.sklearn
from omegaconf import DictConfig, OmegaConf
import hydra

from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score

# Standard logging configuration in all pipelines
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@hydra.main(config_path=os.path.join(os.pardir, 'configs'),
            config_name="scoring_config")
def start_pipeline(scoring_config: DictConfig):

    scoring_config = OmegaConf.to_container(scoring_config)
    model_features = scoring_config['scoring_pipeline']['features']

    # Initialise the W&B run
    run = wandb.init(project=scoring_config['main']['project_name'],
                     group=scoring_config['main']['experiment_name'],
                     job_type=scoring_config['main']['job_type'])

    # Download and load the pipeline from W&B
    model_export_path = run.use_artifact(
        scoring_config['scoring_pipeline']['scoring_model']).download()
    pipeline = mlflow.sklearn.load_model(model_export_path)

    # Download test artifact from W&B and split into X and y
    logger.info('Download test Artifact from W&B')
    test_data_path = run.use_artifact(
        scoring_config['data']['test_data']).file()

    test_df = pd.read_csv(test_data_path)
    test_X = test_df.copy()
    test_y = test_X.pop(model_features['target_var'])

    # Place the target variable into bins for categorization
    target_labels = model_features['target_bins_labels']
    num_labels = len(target_labels)

    test_y = pd.cut(test_y,
                    bins=num_labels,
                    labels=target_labels,
                    ordered=False)

    test_y = label_binarize(test_y, classes=target_labels).ravel()

    # Etestuate the model
    logger.info('Testing the model')
    test_pred = pipeline.predict(test_X)
    test_probas = pipeline.predict_proba(test_X)

    test_accuracy = accuracy_score(test_y, test_pred)
    run.summary['test_accuracy'] = test_accuracy
    run.summary['test_roc_auc_score'] = roc_auc_score(test_y,
                                                      test_probas[:, 1],
                                                      multi_class='ovr')

    # Plot the ROC curve, precision-recall curve and confusion matrix in W&B
    wandb.sklearn.plot_confusion_matrix(test_y, test_pred, target_labels)
    wandb.sklearn.plot_precision_recall(test_y, test_probas, target_labels)
    wandb.sklearn.plot_roc(test_y, test_probas, target_labels)

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":

    start_pipeline()
