import common_util
import pandas as pd

def main_method():
    pd.set_option('display.max_rows', 500)

    train_house_price_pd = common_util.read_pd_data("train.csv")
    test_house_price_pd = common_util.read_pd_data("test.csv")

    train_house_price_pd = train_house_price_pd.drop('SalePrice', axis=1)

    x_train_dummied_pd = common_util.to_dummies(train_house_price_pd)
    x_test_dummied_pd = common_util.to_dummies(test_house_price_pd)

    print(x_train_dummied_pd.values.shape)
    print(x_test_dummied_pd.values.shape)

    merged_pd = pd.merge(x_test_dummied_pd, x_train_dummied_pd,  how='left')

    print(merged_pd.shape)
    print(merged_pd["LotFrontage"][1:10])
    print(merged_pd.fillna(merged_pd.mean())["LotFrontage"][1:10])

main_method()

