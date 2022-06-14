import common_util
import numpy as np
import pandas as pd
import feature_engineering as fe
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

def main_tabular():
    sample_pd_data = common_util.read_pd_data('random_sampling_tablular.csv')
    primary_pd = common_util.read_pd_data('head_3000_submission.csv')
    use_train=False

    target_name = 'F_1_0'
    train_X, test_X, train_y, test_y = generate_target_data_set(sample_pd_data, target_name=target_name, use_train=use_train)

    # 目的変数のインデックス番号
    target_index = np.array(np.array(test_X[:, 0], dtype=int), dtype=str)
    target_row_name = np.array(np.full(target_index.size, '-'+target_name), dtype=object)
    target_row = target_index + target_row_name

    model = Ridge()
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)

    if use_train:
        rmse = np.sqrt(mean_squared_error(test_y, pred_y))
        print(rmse)

    pred_pd = pd.DataFrame(data={'row-col': target_row, 'value': pred_y})
    merge_pd_data(primary_pd, pred_pd)

    print("end.")

#
def merge_pd_data(primary_pd: pd.DataFrame, pred_pd: pd.DataFrame):
    result_pd = pd.concat([primary_pd, pred_pd]).groupby('row-col').last().reset_index()
    # print(result_pd[result_pd['row-col'] == '21-F_1_0'])
    return result_pd


# 指定した特徴量を目的変数にして返却する。
# use_train=True モデル作成用。
# use_train=False 実際用。
def generate_target_data_set(pd_data, target_name, use_train=True):
    train_pd_data = pd_data[pd_data[target_name].notnull()]
    test_pd_data = pd_data[pd_data[target_name].isnull()]
    # 欠損値を埋める。テスト用の pd はターゲット
    train_pd_data = fe.inspect_data().fill_na(train_pd_data)
    test_pd_data = fe.inspect_data().fill_na(test_pd_data, target_name)

    if use_train:
        return train_test_split(train_pd_data.drop(target_name, axis=1).values,
                                train_pd_data[target_name].values)
    else:
        return train_pd_data.drop(target_name, axis=1).values,\
               test_pd_data.drop(target_name, axis=1).values, \
               train_pd_data[target_name].values, \
               test_pd_data.drop(target_name, axis=1).values


main_tabular()