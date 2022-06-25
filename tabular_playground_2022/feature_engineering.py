import pandas
import numpy as np
import pandas as pd
import string
from sklearn.preprocessing import PolynomialFeatures

class featureing_data:
    def fill_na(self, pd_data: pandas.DataFrame, target_feature=None):
        return_value = pd_data.fillna(pd_data.median())
        if target_feature is not None:
            return_value[target_feature].replace(0, np.nan, inplace=True)

        return return_value

    def poly_feature(self, pd_data: pandas.DataFrame):
        list = ["F_4_0", "F_4_1", "F_4_2", "F_4_3", "F_4_4", "F_4_5", "F_4_6", "F_4_7", "F_4_8", "F_4_9", "F_4_10",
                "F_4_11", "F_4_12", "F_4_13", "F_4_14"]
        poly_data = PolynomialFeatures(degree=2, include_bias=False).fit_transform(pd_data[list])
        poly_pd_data = pd.DataFrame(poly_data)
        result_pd = pd_data.join(poly_pd_data)

        return result_pd.fillna(result_pd.median())