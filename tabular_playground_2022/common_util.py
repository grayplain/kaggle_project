import pandas as pd
import numpy as np
import os
import random

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# 提出ファイル作成
def output_submit(test_data, estimator):
    sample_submit = read_pd_data('sample_submit.csv', header=None)
    pred = estimator.predict(test_data)
    sample_submit[1] = pred
    sample_submit.to_csv('submit_tabular.csv', header=None, sep=',', index=False)

def read_pd_data(file_name, header="infer", nrows=None):
    file_name_path = '/' + file_name
    # return pd.read_csv(os.getcwd() + '/datas/' + file_name_path, header=header, na_filter=False)
    return pd.read_csv(os.getcwd() + '/datas/' + file_name_path, header=header, nrows=nrows)


def output_submit(predict_pd_data: pd.DataFrame):
    predict_pd_data.to_csv('submit_tabular.csv', header=True, sep=',', index=False)

def output_sample_pd_data(pd_data: pd.DataFrame):
    sample_pd_data = pd_data.sample(n=2000)
    sample_pd_data.to_csv('random_sampling_tablular.csv', header=True, sep=',')

