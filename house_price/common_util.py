import pandas as pd
import numpy as np
import os


# 提出ファイル作成
def output_submit(test_data, estimator):
    sample_submit = read_pd_data('sample_submit.csv', header=None)
    pred = estimator.predict(test_data)
    sample_submit[1] = pred
    sample_submit.to_csv('submit_minting.csv', header=None, sep=',', index=False)

def read_pd_data(file_name, header="infer"):
    file_name_path = '/' + file_name
    return pd.read_csv(os.getcwd() + '/datas/' + file_name_path, header=header, na_filter=False)
    # return pd.read_csv(os.getcwd() + '/datas/' + file_name_path, header=header)


def output_submit(predict_value: np.ndarray):
    sample_submit = read_pd_data('sample_submission.csv')
    sample_submit['SalePrice'] = predict_value
    sample_submit.to_csv('submit_house_price.csv', header=True, sep=',', index=False)
