import math
import string
import pandas as pd
import numpy as np

def delete_feature_engineering(pd_data: pd.DataFrame):
    # 0.1 未満
    pd_data = pd_data.drop('Id', axis=1)
    pd_data = pd_data.drop('Utilities', axis=1)

    pd_data = pd_data.drop('MiscVal', axis=1)
    pd_data = pd_data.drop('PoolArea', axis=1)
    pd_data = pd_data.drop('PoolQC', axis=1)
    pd_data = pd_data.drop('Street', axis=1)
    pd_data = pd_data.drop('LandSlope', axis=1)
    pd_data = pd_data.drop('Condition2', axis=1)
    pd_data = pd_data.drop('HouseStyle', axis=1)
    pd_data = pd_data.drop('Heating', axis=1)
    pd_data = pd_data.drop('MiscFeature', axis=1)

    # 全体の10% 以上欠損値があり、各カテゴリが住宅価格と相関係数が0.1 未満の場合、消す
    # pd_data = pd_data.drop('LotFrontage', axis=1)
    # pd_data = pd_data.drop('FireplaceQu', axis=1)
    # pd_data = pd_data.drop('Fence', axis=1)
    # pd_data = pd_data.drop('Alley', axis=1)

    # 0.1ちょっと
    # pd_data = pd_data.drop('LandContour', axis=1)
    # pd_data = pd_data.drop('LotConfig', axis=1)
    # pd_data = pd_data.drop('Condition1', axis=1)
    # pd_data = pd_data.drop('BldgType', axis=1)
    # pd_data = pd_data.drop('RoofMatl', axis=1)
    # pd_data = pd_data.drop('BsmtCond', axis=1)
    # pd_data = pd_data.drop('Functional', axis=1)

    return pd_data

def fill_na_value(pd_data: pd.DataFrame, target_column: string):
    # 欠損値を埋めるのは中央値で良さそう
    fill_na_val = math.floor(pd_data[target_column].mean())
    result_pd = pd_data.fillna({target_column: fill_na_val})
    return result_pd

def to_dummies(pd_data: pd.DataFrame):
    # return pandas.get_dummies(pd_data, dummy_na=True, drop_first=True)
    return pd.get_dummies(pd_data, dummy_na=True)

def to_numeric_dummies(pd_data: pd.DataFrame, target_column: list):
    # return pandas.get_dummies(pd_data, dummy_na=True, drop_first=True)
    return pd.get_dummies(pd_data, dummy_na=False, columns=target_column)
    # return pd.get_dummies(pd_data, dummy_na=True, columns=target_column)

def fill_na_features(pd_data: pd.DataFrame):
    result_pd = to_numeric_dummies(pd_data, ['MSSubClass'])
    result_pd = to_numeric_dummies(result_pd, ['YearBuilt'])
    result_pd = to_numeric_dummies(result_pd, ['YearRemodAdd'])
    result_pd = to_numeric_dummies(result_pd, ['YrSold', 'MoSold'])
    result_pd = fill_na_value(result_pd, 'LotFrontage')
    result_pd = fill_na_value(result_pd, 'LotArea')
    result_pd = fill_na_value(result_pd, 'MasVnrArea')
    result_pd = fill_na_value(result_pd, 'BsmtFinSF1')
    result_pd = fill_na_value(result_pd, 'BsmtFinSF2')
    result_pd = fill_na_value(result_pd, 'BsmtUnfSF')
    result_pd = fill_na_value(result_pd, 'TotalBsmtSF')
    result_pd = fill_na_value(result_pd, '1stFlrSF')
    result_pd = fill_na_value(result_pd, '2ndFlrSF')
    result_pd = fill_na_value(result_pd, 'GrLivArea')
    result_pd = fill_na_value(result_pd, 'GarageArea')
    result_pd = fill_na_value(result_pd, 'WoodDeckSF')
    result_pd = fill_na_value(result_pd, 'OpenPorchSF')
    result_pd = fill_na_value(result_pd, 'EnclosedPorch')
    result_pd = fill_na_value(result_pd, '3SsnPorch')
    result_pd = fill_na_value(result_pd, 'ScreenPorch')
    return result_pd

def execute_feature_engineering(pd_data: pd.DataFrame):
    result_pd = delete_feature_engineering(pd_data)
    result_pd = fill_na_features(result_pd)
    result_pd = to_dummies(result_pd)
    # どうしても欠損値がある場合は0埋めにする。
    result_pd = result_pd.fillna(0)
    return result_pd

# 外れ値消す。SalesPrice で明らかに高いやつ。
def drop_outliers(train_pd_data: pd.DataFrame):
    result_pd = train_pd_data.drop(train_pd_data[train_pd_data['Id'] == 185].index)
    result_pd = result_pd.drop(result_pd[result_pd['Id'] == 523].index)
    result_pd = result_pd.drop(result_pd[result_pd['Id'] == 691].index)
    result_pd = result_pd.drop(result_pd[result_pd['Id'] == 803].index)
    result_pd = result_pd.drop(result_pd[result_pd['Id'] == 898].index)
    result_pd = result_pd.drop(result_pd[result_pd['Id'] == 1182].index)
    result_pd = result_pd.drop(result_pd[result_pd['Id'] == 1298].index)
    return result_pd


# 推定器が重要だと判断して特徴量だけを抜き出したバージョンだが、正直あかんかった。
def extract_pd_data(pd_data: pd.DataFrame):
    result_pd = pd.DataFrame(data={'LotArea': pd_data['LotArea'],
                       'OverallQual': pd_data['OverallQual'],
                       'BsmtFinSF1': pd_data['BsmtFinSF1'],
                       'TotalBsmtSF': pd_data['TotalBsmtSF'],
                       '1stFlrSF': pd_data['1stFlrSF'],
                       '2ndFlrSF': pd_data['2ndFlrSF'],
                       'GrLivArea': pd_data['GrLivArea'],
                       'FullBath': pd_data['FullBath'],
                       'GarageCars': pd_data['GarageCars'],
                       'BsmtQual_Ex': pd_data['BsmtQual_Ex']})

    return result_pd


def extract_ym_pd_data(pd_data: pd.DataFrame):
    result_pd = pd.DataFrame(data={'YrSold': pd_data['YrSold'],
                       'MoSold': pd_data['MoSold'],
                       'SalePrice': pd_data['SalePrice']})

    return  result_pd


def fill_in_categories(pd_data_1: pd.DataFrame, pd_data_2: pd.DataFrame):
    for feature_name in ['YearBuilt', 'RoofMatl', 'Electrical', 'Exterior1st', 'Exterior2nd', 'GarageQual', 'MSSubClass']:
        categories = set(pd_data_1[feature_name].unique().tolist() + pd_data_1[feature_name].unique().tolist())
        categories.discard(np.nan)
        pd_data_1[feature_name] = pd.Categorical(pd_data_1[feature_name], categories=categories)
        pd_data_2[feature_name] = pd.Categorical(pd_data_2[feature_name], categories=categories)

    return pd_data_1, pd_data_2