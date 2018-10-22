import numpy as np
import logging
from sklearn import preprocessing
import os
from A3LAB_Framework.utility.utility import makedir


class DataPreprocessing:
    def __init__(self, data_dims_num=None, path=None):
        """
        initialization preprocessing class
        """
        # logging.info("Preprocessing data")
        if data_dims_num is None:
            raise "Specify number of data dimensions"
        self.data_dims_num = data_dims_num
        self.data_shape = list()
        self._scaler = None
        self.mean = np.zeros(0)
        self.std = np.zeros(0)
        self.path = path

    def get_max_data_dimensions(self, data, concatenate=False):
        """
        :param data list of samples
        :param data_shape
        :returns maximum data dimensions in the set
        """

        self.data_shape = [len(data)]
        data_shapes = np.empty((len(data), self.data_dims_num))

        for i in range(0, len(data)):
            shape = data[i].shape
            data_shapes[i] = shape
        if concatenate:
            max_dimensions = np.sum(data_shapes, axis=0).astype(np.int)
            max_dimensions[1] = shape[1]
        else:
            max_dimensions = np.amax(data_shapes, axis=0).astype(np.int)
        self.data_shape.append(max_dimensions.tolist())
        return max_dimensions

    def zero_padding_set(self, data, max_shape=None, features_dimension=None, axis_to_pad=0):
        """
        padding set of data with zeros to reach the specified max_shape
        by default the features dimension is the second dimension of the data, otherwise specify
        the axis and the dimension of padding
        :param data list of samples
        :param max_shape(tuple) desired shape to obtain after pad
        :param features_dimension dimension of features vector (fixed)
        :param axis_to_pad axis of samples to pad
        :returns padded_data list of padded samples

        """
        if max_shape is None:
            max_shape = DataPreprocessing.get_max_data_dimensions(data)
        if features_dimension is None:
            features_dimension = max_shape[1]
        padded_data = np.empty((len(data), max_shape[0], max_shape[1]))
        for i in range(len(data)):
            # zero pad samples
            sample = np.asarray(data[i], dtype=np.float32)
            pad = max_shape[axis_to_pad] - sample.shape[axis_to_pad]
            zero_matrix = np.zeros((pad, features_dimension))
            sample = np.concatenate((sample, zero_matrix), axis=axis_to_pad)
            padded_data[i] = sample
        return padded_data

    def concatenate_data(self, data, axis_to_concatenate=0, shapes=None):
        """
        concatenate samples along an axis
        :param data:
        :param axis_to_concatenate
        :return concatenated_data
        """
        if shapes is None:
            concatenated_data = np.zeros((self.data_shape[0] * self.data_shape[1][axis_to_concatenate],
                                         self.data_shape[1][1]), dtype=np.float32)
            for i in range(len(data)):
                concatenated_data[i * self.data_shape[1][axis_to_concatenate]:
                                  (i + 1) * self.data_shape[1][axis_to_concatenate], :] = data[i]
        else:
            concatenated_data = np.zeros((shapes[0], shapes[1]), dtype=np.float32)
            start_idx = 0
            for i in range(len(data)):
                concatenated_data[start_idx: start_idx+data[i].shape[0], :] = data[i]
                start_idx += data[i].shape[0]
        return concatenated_data

    def standardize_data(self, data, scaler=None, save_stats=False):
        """
        transform the data to center it by removing the mean value of each feature,
        then scale it by dividing non-constant features by their standard deviation.

        :param data:
        :param scaler: eventually load from disk for future experiments
        :param save_stats: flag, save scaler on disk
        :return: data standardized
        """
        logging.info("Standardize Data")
        if scaler is None:
            if self._scaler is None:
                logging.info("Computing new scaler")
                scaler = preprocessing.StandardScaler(copy=False, with_mean=True, with_std=True).fit(data)
                self._scaler = scaler
            else:
                logging.info("Using previous computed scaler")
                scaler = self._scaler
            data_std = scaler.transform(data)
            if save_stats:
                makedir(os.path.join(self.path, 'stats'))
                scaler_file = os.path.join(self.path, 'stats', 'scaler.npy')
                np.save(scaler_file, scaler)
        else:
            logging.info("Using scaler loaded from disk")
            data_std = scaler.transform(data)

        return data_std

    def standardize_data_numpy(self, data, mean=0, std=0, save_stats=False):
        '''
        normalizza media e varianza del dataset passato
        se mean e variance = 0 essi vengono calcolati in place sui data
        '''
        logging.info("Standardize Data")
        data = np.asarray(data, dtype=np.float32)

        if mean and std:
            logging.info("Using mean and stdev loaded from disk")
            self.mean = np.load(mean)
            self.std = np.load(std)

        if np.all(self.mean == 0) and np.all(self.std == 0):  # compute mean and variance of the passed data
            logging.info("Computing new mean and stdev")
            self.mean = np.mean(data, axis=(0, 1))
            self.std = np.std(data, axis=(0, 1))
        data = (data - self.mean) / self.std

        if save_stats:
            makedir(os.path.join(self.path, 'stats'))
            mean_file = os.path.join(self.path, 'stats', 'mean.npy')
            std_file = os.path.join(self.path, 'stats', 'std.npy')
            np.save(mean_file, mean)
            np.save(std_file, std)
        return data

    def min_max_scale_data(self, data, range=(-1, 1), scaler=None):
        """
        Transforms features by scaling each feature to a given range.
        :param data:
        :param range (tuple)
        :param scaler: eventually load from disk for future experiments
        :return: data scaled
        """
        if self._scaler is None:
            scaler = preprocessing.MinMaxScaler(feature_range=range).fit(data)
            data_std = scaler.transform(data)
            self._scaler = scaler
            # TODO save scaler to disk
        else:
            data_std = self._scaler.transform(data)

        return data_std

    @staticmethod
    def divide_in_sequences(data, labels, seq_len):
        total_length = data.shape[0]
        num_sequences = total_length // seq_len
        splitted_data = np.empty((num_sequences, seq_len, data.shape[1]), dtype=np.float32)
        splitted_labels = np.empty((num_sequences, seq_len, 1), dtype=np.int)

        for i in range(0, num_sequences):
            splitted_data[i, :] = data[i * seq_len:(i + 1) * seq_len, :]
            splitted_labels[i, :] = labels[i * seq_len:(i + 1) * seq_len, :]

        return splitted_data, splitted_labels



