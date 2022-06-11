import pandas
import numpy as np
import pandas as pd


class inspect_data:
    def fill_na(self, pd_data: pandas.DataFrame, target_feature=None):
        return_value = pd_data.fillna(0)
        if target_feature is not None:
            return_value[target_feature].replace(0, np.nan, inplace=True)

        return return_value