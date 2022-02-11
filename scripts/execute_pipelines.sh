#!/bin/sh

echo 'Running yapf with google style'
yapf -ir .. --style google -vv

echo 'Reformatting completed'

echo 'Retrieving, preprocessing and segregating data...'
mlflow run ../executor/. -P config_name=eda_config.yaml

echo 'Data pipeline executed'

echo 'Generating pandas-profiling report for train dataset...'
python data_profiler.py --input_file=../data/segregated/student_maths_train.csv --output_file=./train_data_profile.html

echo 'pandas-profiling report generated.'

echo 'Training Model...'
mlflow run ../inference/. 

echo 'Model training completed.'

echo 'Evaluating the model...'
mlflow run ../scoring/. 

echo 'Model scoring completed'