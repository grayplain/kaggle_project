import common_util
import feature_engineering as fe

sample_pd_data = common_util.read_pd_data('random_sampling_tablular.csv')

train_pd_data = sample_pd_data[sample_pd_data['F_1_0'].notnull()]
test_pd_data = sample_pd_data[sample_pd_data['F_1_0'].isnull ()]

# 欠損値を埋める。テスト用の pd はターゲット
train_pd_data = fe.inspect_data().fill_na(train_pd_data)
test_pd_data = fe.inspect_data().fill_na(test_pd_data, 'F_1_0')

