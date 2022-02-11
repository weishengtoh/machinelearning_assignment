import os

import pandas as pd
import pytest
import yaml

import wandb

run = wandb.init(project='RP_NVIDIA_Machine_Learning',
                 job_type='data_validation')


@pytest.fixture(scope='session')
def data():

    config_path = os.path.join(os.pardir, 'configs')

    with open(os.path.join(config_path, 'dataval_config.yaml'), 'r') as file:
        config_name = yaml.safe_load(file)

    data_artifact = config_name['parameters']['artifact_name']

    if data_artifact is None:
        pytest.fail('missing --data_artifact argument')

    data_path = run.use_artifact(data_artifact).file()
    df = pd.read_csv(data_path)

    return df
