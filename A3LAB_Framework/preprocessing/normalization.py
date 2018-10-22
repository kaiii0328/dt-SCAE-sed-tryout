import numpy as np


class GlobalMinMaxScaler(object):
    def __init__(self, feature_range=(0, 1)):
        self.data_min = 0
        self.data_max = 0
        self.feature_range = feature_range

        if self.feature_range[0] >= self.feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(self.feature_range))

    def fit(self, data: object) -> object:
        self.data_min = data.min()
        self.data_max = data.max()

    def transform(self, data):
        data_scaled = np.array(data, copy=True)
        data_scaled = (data_scaled - self.data_min) / (self.data_max - self.data_min)
        data_scaled = data_scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

        return data_scaled

    def fit_transform(self, data):
        self.fit(data)

        return self.transform(data)
