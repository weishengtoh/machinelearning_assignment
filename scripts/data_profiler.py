import pandas as pd
import os
import argparse

from pandas_profiling import ProfileReport


def generate_eda(args):

    file_path = args.input_file
    train_df = pd.read_csv(os.path.join(file_path))

    profile = ProfileReport(train_df,
                            title="Pandas Profiling Report",
                            explorative=True,
                            progress_bar=True)

    profile.to_file(args.output_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Generate EDA output from pandas-profiling',
        fromfile_prefix_chars='@')

    parser.add_argument('--output_file',
                        type=str,
                        help='Name of output file to run',
                        required=True)

    parser.add_argument('--input_file',
                        type=str,
                        help='Name of input file to run',
                        required=True)

    args = parser.parse_args()
    generate_eda(args)
