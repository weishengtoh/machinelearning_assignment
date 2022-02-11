'''
Define the pipeline component to extract and unzip the raw data from a specified URI

Raw data is saved as student-mat.csv locally in the directory /data/raw and also 
as an artifact in Weights & Biases.

The artifact generated from this pipeline component is to be used by the 
preprocess_pipeline component.
'''

import logging
import os
import urllib.request
from datetime import datetime
from zipfile import ZipFile

import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@hydra.main(config_path=os.path.join(os.pardir, 'configs'),
            config_name="dataloader_config")
def start_pipeline(dataloader_config: DictConfig):

    dataloader_config = OmegaConf.to_container(dataloader_config)

    params = dataloader_config['parameters']

    cwd_path = hydra.utils.get_original_cwd()
    root_path = os.path.join(cwd_path, os.pardir)
    output_folder = os.path.join(*[root_path, 'data', params['output_folder']])
    file_path = os.path.join(output_folder, params["artifact_name"])

    logger.info(f'Download the dataset from {params["file_url"]}')
    data_ingestion_time = download_unzip(params["file_url"], output_folder,
                                         params["artifact_name"])

    # Initialise the W&B run
    with wandb.init(project=dataloader_config['main']['project_name'],
                    group=dataloader_config['main']['experiment_name'],
                    job_type=dataloader_config['main']['job_type']) as run:
        logger.info('Create a Weights and Biases Artifact')
        artifact = wandb.Artifact(name=params["artifact_name"],
                                  type=params["artifact_type"],
                                  description=params["artifact_description"],
                                  metadata={
                                      'file_url': params["file_url"],
                                      'data_ingestion_time': data_ingestion_time
                                  })

        logger.info(f'Save file locally to {file_path}')
        df = pd.read_csv(
            file_path, sep=';'
        )  # Original file is seperated by ";" - will be changed to comma sep file after saving
        df.to_csv(file_path, index=False)

        logger.info('Create and add artifact to be logged in W&B')
        artifact.add_file(file_path, name=params["artifact_name"])
        run.log_artifact(artifact)

    # Finish the wandb run
    wandb.finish()


def download_unzip(url: str,
                   output_folder: str,
                   file_name,
                   delete_zip=True) -> str:

    os.makedirs(
        output_folder,
        exist_ok=True)  # Create the output folder if not already created

    # Attempts to download file
    RAW_file_path = os.path.join(output_folder, 'raw.zip')
    urllib.request.urlretrieve(url, RAW_file_path)

    # Track the time that the file was downloaded
    datetime_now = datetime.now()
    data_ingestion_time = f'{datetime_now.year}_{datetime_now.month}_{datetime_now.day}'

    # Attempts to unzip file to output folder
    with ZipFile(RAW_file_path, 'r') as zip:
        zip.extract(member=file_name, path=output_folder)

    if delete_zip:
        os.remove(RAW_file_path)

    return data_ingestion_time


if __name__ == '__main__':

    start_pipeline()
