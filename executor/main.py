import argparse
import os

import mlflow
import yaml


def start_pipelines(args):

    # Get the configuration as a yaml file
    config = os.path.join(*[os.pardir, 'configs', args.config_name])

    with open(config) as file:
        config = yaml.safe_load(file)

    # Define the weights and biases project name and group for tracking
    os.environ['WANDB_PROJECT'] = config['main']['project_name']
    os.environ['WANDB_RUN_GROUP'] = config['main']['experiment_name']

    execution_steps = config['execute']

    # Run pipeline to download and extract data
    if 'dataloader' in execution_steps:
        mlflow.run(uri=os.path.join(os.pardir, 'dataloader'),
                   entry_point='main')

    # Run preprocessing pipeline
    if 'preprocess' in execution_steps:
        mlflow.run(uri=os.path.join(os.pardir, 'preprocess'),
                   entry_point='main')

    # Run data validation pipeline
    if 'dataval' in execution_steps:
        mlflow.run(uri=os.path.join(os.pardir, 'dataval'), entry_point='main')

    # Run pipeline to split data into train, val, test split
    if 'segregate' in execution_steps:
        mlflow.run(uri=os.path.join(os.pardir, 'segregate'), entry_point='main')

    # Run inference pipeline
    if 'inference' in execution_steps:
        mlflow.run(uri=os.path.join(os.pardir, 'inference'), entry_point='main')

    # Run scoring pipeline
    if 'scoring' in execution_steps:
        mlflow.run(uri=os.path.join(os.pardir, 'scoring'), entry_point='main')

    # Run pipeline to push model to registry for serving
    # mlflow.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Main entrypoint to run the MLflow pipelines',
        fromfile_prefix_chars='@')

    parser.add_argument('--config_name',
                        type=str,
                        help='Name of config file to run',
                        required=True)

    args = parser.parse_args()
    start_pipelines(args)
