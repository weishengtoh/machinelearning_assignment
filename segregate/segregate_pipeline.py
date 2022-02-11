'''
Define the pipeline that is used to segregate the data into train, validation 
and test splits. 

This pipeline component is meant to only run when the data validation component
succeeds. 

This pipeline taks in the student-maths-clean.csv data artifact from Weights & Biases
and splits the data according to the seed and stratify defined in the config file.

The train, val and test data artifacts generated from this pipeline component is 
to be used by the inference_pipeline component for training and by scoring_pipeline
for evaluation of the final model.
'''

import logging
import os

import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

# Basic setup of the logging utility
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@hydra.main(config_path=os.path.join(os.pardir, 'configs'),
            config_name="segregate_config")
def start_pipeline(segregate_config: DictConfig):

    segregate_config = OmegaConf.to_container(segregate_config)
    params = segregate_config['parameters']

    logger.info('Start a Weights and Biases run')
    run = wandb.init(job_type='split_data')

    logger.info('Download data Artifact')
    data_artifact = run.use_artifact(params['input_artifact'])
    DATA_ARTIFACT_PATH = data_artifact.file()

    df = pd.read_csv(DATA_ARTIFACT_PATH)

    logger.info('Split data into train, validation and test sets')
    train, test = train_test_split(df,
                                   test_size=params['test_size'],
                                   random_state=params['random_state'],
                                   stratify=df[params['stratify']]
                                   if params['stratify'] != 'none' else None)

    train, val = train_test_split(
        train,
        # This is just to express the val size as the train data computed previously
        test_size=params['val_size'] / (1 - params['test_size']),
        random_state=params['random_state'],
        stratify=train[params['stratify']]
        if params['stratify'] != 'None' else None)

    split_data = {'train': train, 'val': val, 'test': test}

    logger.info('Create a folder to store the segregated data')
    CWD_PATH = hydra.utils.get_original_cwd()
    ROOT_PATH = os.path.join(CWD_PATH, os.pardir)
    OUTPUT_FOLDER = os.path.join(*[ROOT_PATH, 'data', params['output_folder']])

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # save the train, validation and test data to Weights and Biases
    for split, df in split_data.items():

        artifact_name = f'{params["artifact_rootname"]}_{split}.csv'

        logger.info(f'Upload {split} data to {artifact_name}')

        df.to_csv(os.path.join(OUTPUT_FOLDER, artifact_name), index=False)

        artifact = wandb.Artifact(
            name=artifact_name,
            type=params['artifact_type'],
            description=f'{split} split of dataset {params["input_artifact"]}')

        logger.info(f'Log {split} Artifact - {artifact_name}')
        artifact.add_file(os.path.join(OUTPUT_FOLDER, artifact_name))
        run.log_artifact(artifact)


if __name__ == '__main__':

    start_pipeline()
