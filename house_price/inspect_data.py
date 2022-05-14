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

    print(train_house_price_pd.isnull().sum().sort_values())
    # train_house_price_pd = train_house_price_pd.drop('SalePrice', axis=1)

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