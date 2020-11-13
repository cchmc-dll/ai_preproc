import os
import pandas as pd


def merge_csv(input_files, key='ID', input_dir="preprocessed", output_dir="output", output_file="merged_data.csv"):
    if not os.path.isabs(input_dir):
        input_dir = os.path.abspath(input_dir)

    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)

    input_list = list()
    for i in input_files:
        input_list.append(os.path.join(input_dir, i))

    df_base = []
    for index, in_file in enumerate(input_list):
        # Read INPUT
        df_in = pd.read_csv(in_file, index_col=False)
        df_in.set_index(key, inplace=True)
        if index > 0:
            df_base = pd.merge(df_base, df_in, how='left', on=[key])
        else:
            df_base = df_in.copy()

    df_base.to_csv(os.path.join(output_dir, output_file), encoding='utf-8')
