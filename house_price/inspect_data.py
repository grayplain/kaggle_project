import common_util
import pandas as pd
import feature_engineering
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main_method():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 70)

    house_price_pd = common_util.read_pd_data("train.csv")
    house_price_pd = feature_engineering.drop_outliers(house_price_pd)

    x_pd = house_price_pd.drop('SalePrice', axis=1)
    x_pd = feature_engineering.execute_feature_engineering(x_pd)

    test_house_price_pd = common_util.read_pd_data("test.csv")
    target_pd = feature_engineering.execute_feature_engineering(test_house_price_pd)
    target_pd = feature_engineering.supplement_column_to_test(target_pd)

    aaa = x_pd.columns.values
    bbb = target_pd.columns.values
    aaa_bbb = np.setdiff1d(aaa, bbb)
    bbb_aaa = np.setdiff1d(bbb, aaa)
    print(aaa_bbb)
    print(bbb_aaa)

main_method()

def inspect_from_graph():
    train = common_util.read_pd_data("train.csv")
    train = train.sort_values('SalePrice', ascending=False)

    # plt.hist(train['SalePrice'], bins=30)
    plt.scatter(train['SalePrice'], train['YrSold'])

    # pd_data = pd_data.drop('YrSold', axis=1)
    # pd_data = pd_data.drop('MoSold', axis=1)
    plt.show()

# inspect_from_graph()