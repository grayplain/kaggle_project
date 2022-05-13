import common_util
import pandas as pd
import feature_engineering
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main_method():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 70)

    train_house_price_pd = common_util.read_pd_data("train.csv")
    train_house_price_pd = train_house_price_pd.drop('SalePrice', axis=1)
    train_house_price_pd = feature_engineering.execute_feature_engineering(train_house_price_pd)

    test_house_price_pd = common_util.read_pd_data("test.csv")
    target_pd = feature_engineering.execute_feature_engineering(test_house_price_pd)

    target_pd = feature_engineering.supplement_column_to_test(target_pd)


    # print(train_house_price_pd)
    aaa = target_pd.columns.values
    bbb = train_house_price_pd.columns.values

    aaa_bbb = np.setdiff1d(aaa, bbb)
    bbb_aaa = np.setdiff1d(bbb, aaa)

    print(aaa_bbb)
    print(bbb_aaa)

main_method()



def seaborn_test():
    train_house_price_pd = common_util.read_pd_data("train.csv")
    train_house_price_pd = train_house_price_pd[train_house_price_pd['SalePrice'] <= 200000]

    house_price_corr = train_house_price_pd.corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(house_price_corr, square=True, vmax=1, vmin=-1, center=0)

    plt.show()

# seaborn_test()