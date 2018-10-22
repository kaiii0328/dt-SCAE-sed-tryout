import numpy as np


class GlobalMeanStdScaler(object):
    def __init__(self, feature_range=(0, 1)):
        self.data_mean = 0    #mean_value
        self.data_std = 0     #standard deviation
        self.feature_range = feature_range

        if self.feature_range[0] >= self.feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(self.feature_range))

    def fit(self, data: object) -> object:
        self.data_mean = data.mean()
        self.data_std = data.std()

    def transform(self, data):
        # data_scaled = np.array(data, copy=True)
        data_scaled = np.array(data, copy=False)
        data_scaled = (data_scaled - self.data_mean) / (self.data_std)

        return data_scaled

    def fit_transform(self, data):
        self.fit(data)

        return self.transform(data)
