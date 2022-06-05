from tabular_playground_2022.datas import common_util
import pandas as pd
import feature_engineering
import numpy as np
import matplotlib.pyplot as plt

def main_method():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 70)

    house_price_pd = common_util.read_pd_data("train.csv")
    test_house_price_pd = common_util.read_pd_data("test.csv")

    house_price_pd, test_house_price_pd = feature_engineering.fill_in_categories(house_price_pd, test_house_price_pd)

    house_price_pd = feature_engineering.drop_outliers(house_price_pd)

    x_pd = house_price_pd.drop('SalePrice', axis=1)
    x_pd = feature_engineering.execute_feature_engineering(x_pd)


    target_pd = feature_engineering.execute_feature_engineering(test_house_price_pd)

    aaa = x_pd.columns.values
    bbb = target_pd.columns.values
    aaa_bbb = np.setdiff1d(aaa, bbb)
    bbb_aaa = np.setdiff1d(bbb, aaa)
    print(aaa_bbb)
    print(bbb_aaa)

# main_method()

def inspect_from_graph():
    train = common_util.read_pd_data("train.csv")

    # data = train['SalePrice'].to_numpy()

    high_data = train[train['SalePrice'] >= 300000]['SalePrice'].to_numpy()
    low_data = train[train['SalePrice'] < 300000]['SalePrice'].to_numpy()
    # data = train['SalePrice'].to_numpy()
    unk = [high_data, low_data]

    fig, axes = plt.subplots(1, 2)

    for data, ax in zip(unk, axes):
        ax.boxplot(data)

    # plt.boxplot(data)

    plt.show()


inspect_from_graph()