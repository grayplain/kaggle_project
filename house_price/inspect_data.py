import common_util
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import numpy as np

def main_method():
    pd.set_option('display.max_rows', 500)

    train_house_price_pd = common_util.read_pd_data("train.csv")
    train_house_price_pd = train_house_price_pd.drop('Id', axis=1)

    target_column_name = 'SaleCondition'

    filtered_price_pd = train_house_price_pd.loc[:, ['SalePrice', target_column_name]]

    # exist_pool_pd = filtered_price_pd.query('PoolQC == PoolQC')
    x_dummied_pd = common_util.to_numeric_dummies(filtered_price_pd, [target_column_name])

    # print(filtered_price_pd.corr())
    print(x_dummied_pd.corr())

    # x_dummied_pd = common_util.to_dummies(train_house_price_pd).fillna(0)


main_method()



# def seaborn_test():
#     train_house_price_pd = common_util.read_pd_data("train.csv")
#
#     house_price_corr = train_house_price_pd.corr()
#     fig, ax = plt.subplots(figsize=(12, 9))
#     sns.heatmap(house_price_corr, square=True, vmax=1, vmin=-1, center=0)
#
#     plt.show()

# seaborn_test()