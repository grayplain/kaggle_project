import common_util
import numpy as np
import pandas as pd
import feature_engineering as fe
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

def main_tabular():
    data_set_pd_data = common_util.read_pd_data('random_sampling_tablular.csv')
    submit_pd = common_util.read_pd_data('head_3000_submission.csv')
    data_set_pd_data = common_util.read_pd_data('data.csv')
    submit_pd = common_util.read_pd_data('sample_submission.csv')
    submit_pd['index'] = submit_pd.index
    use_train=False

    targets = data_set_pd_data.drop(['row_id'], axis=1).columns.values
    for target_name in targets:
        print('target_name={}、start.'.format(target_name))

        if data_set_pd_data[data_set_pd_data[target_name].isnull()].count(axis=1).size <= 0:
            print('target_name={}、not exist.'.format(target_name))
            continue

        train_X, test_X, train_y, test_y = generate_target_data_set(data_set_pd_data,
                                                                    target_name=target_name,
                                                                    use_train=use_train)

        # 目的変数のインデックス番号
        target_index = np.array(np.array(test_X[:, 0], dtype=int), dtype=str)
        target_row_name = np.array(np.full(target_index.size, '-' + target_name), dtype=object)
        target_row = target_index + target_row_name

        model = Ridge()
        model.fit(np.delete(train_X, [0], axis=1), train_y)
        pred_y = model.predict(np.delete(test_X, [0], axis=1))
        # model.fit(train_X, train_y)
        # pred_y = model.predict(test_X)

        if use_train:
            rmse = np.sqrt(mean_squared_error(test_y, pred_y))
            print('target_name={}, rmse={}'.format(target_name, rmse))
        else:
            print('target_name={}、done'.format(target_name))
            pred_pd = pd.DataFrame(data={'row-col': target_row, 'value': pred_y})
            submit_pd = merge_pd_data(submit_pd, pred_pd)

    submit_pd = trim_index_pd_data(submit_pd)
    common_util.output_submit(submit_pd)
    print("end.")

def trim_index_pd_data(pd_data: pd.DataFrame, column_name='index'):
    ret_value = pd_data.sort_values(column_name)
    ret_value = ret_value.drop(column_name, axis=1)
    return ret_value

#
def merge_pd_data(primary_pd: pd.DataFrame, pred_pd: pd.DataFrame):
    result_pd2 = pd.concat([primary_pd, pred_pd]).groupby('row-col', as_index=False).last()

    # result_pd = pd.concat([primary_pd, pred_pd]).groupby('row-col').last().reset_index()
    # print(result_pd[result_pd['row-col'] == '21-F_1_0'])
    return result_pd2


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