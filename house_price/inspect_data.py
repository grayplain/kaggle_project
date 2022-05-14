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


main_method()



def seaborn_test():
    train = common_util.read_pd_data("train.csv")
    train = train.sort_values('SalePrice', ascending=False)

    plt.hist(train['SalePrice'], bins=30)
    plt.show()


# seaborn_test()