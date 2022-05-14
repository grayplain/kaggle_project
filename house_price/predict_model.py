import common_util
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

import feature_engineering

def pipe_model(model):
    return make_pipeline(RobustScaler(), model)

def rmse_score(y_true, y_pred):
    """RMSE (Root Mean Square Error: 平均二乗誤差平方根) を計算する関数"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return rmse

def fit_model(x_train: np.ndarray, y_train: np.ndarray):
    # model = pipe_model(GradientBoostingRegressor())
    model = GradientBoostingRegressor()

    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(x_train)
    rmse = np.sqrt(-cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_log_error", cv=kf))

    print("\nscore: {:.4f} ({:.4f})\n".format(rmse.mean(), rmse.std()))

    fitted_model = model.fit(x_train, y_train)
    return fitted_model

def score_model(fitted_model, x_test: np.ndarray, y_test: np.ndarray):
    result = fitted_model.predict(x_test)
    print(np.sqrt(mean_squared_log_error(y_test, result)))

def predict_model(fitted_model, x_test: np.ndarray):
    result = fitted_model.predict(x_test)
    print(result)
    return result

def main_method():
    house_price_pd = common_util.read_pd_data("train.csv")

    house_price_pd = feature_engineering.drop_outliers(house_price_pd)

    x_pd = house_price_pd.drop('SalePrice', axis=1)
    y_pd = house_price_pd['SalePrice']

    x_pd = feature_engineering.execute_feature_engineering(x_pd)
    x_pd = feature_engineering.extract_pd_data(x_pd)

    fitted_model = fit_model(x_train=x_pd.to_numpy(), y_train=y_pd.to_numpy())

    test_house_price_pd = common_util.read_pd_data("test.csv")
    target_pd = feature_engineering.execute_feature_engineering(test_house_price_pd)
    target_pd = feature_engineering.extract_pd_data(target_pd)

    result_value = predict_model(fitted_model, RobustScaler().fit_transform(target_pd.to_numpy()))
    common_util.output_submit(result_value)

main_method()


# def hoge_method():
#     test_house_price_pd = common_util.read_pd_data("test.csv")
#
#     print(test_house_price_pd["MSSubClass"])
#     return
#     x_test_dummied_pd = common_util.to_dummies(test_house_price_pd)
#     target_x = x_test_dummied_pd.to_numpy()
#
# hoge_method()