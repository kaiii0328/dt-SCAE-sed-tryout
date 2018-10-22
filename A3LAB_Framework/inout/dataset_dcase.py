#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:36:52 2017

@author: fabiov89
"""
from A3LAB_Framework.inout.dataset import *
import os
import math
from numpy.lib.stride_tricks import as_strided as ast


class DatasetDCASE2017(Dataset):
    def __init__(self, class_labels):
        logging.info("Loading Dataset")
        self.folds = []
        self.dataset_path = None
        self.dataset_structure = pd.DataFrame
        self.year = 2017
        if class_labels == 'Home':
            self.class_labels = ['(object) rustling', '(object) snapping', 'cupboard', 'cutlery', 'dishes', 'drawer',
                                 'glass jingling', 'object impact', 'people walking', 'washing dishes', 'water tap running']
            self.year = 2016
        elif class_labels == 'Residential':
            self.class_labels = ['(object) banging','bird singing','car passing by','children shouting',
                                 'people speaking','people walking','wind blowing']
            self.year = 2016
        else:
            self.class_labels = ['brakes squeaking', 'car', 'children', 'large vehicle', 'people speaking', 'people walking']

    def load_dataset_metadata(self, dataset_list=None,
                              metadata_fields=["filename"],
                              field_sep='\t', header='infer'):
        """"
        loading file with metadata description and metadata
        :param dataset_list: list of paths of csv files which describes each dataset sample
        :param metadata_fields: columns of the csv file
        :param field_sep: row-element separator
        :param header (T/F)

        """
        if len(dataset_list) > 1:
            dataset_structure = dict(trainset=None, validset=None, testset=None)
            dataset_structure['trainset'] = pd.read_csv(dataset_list[0], sep=field_sep, names=metadata_fields, header=header)
            dataset_structure['validset'] = pd.read_csv(dataset_list[1], sep=field_sep, names=metadata_fields, header=header)
            dataset_structure['testset'] = pd.read_csv(dataset_list[2], sep=field_sep, names=metadata_fields, header=header)
            self.dataset_structure = dataset_structure
        else:
            dataset_structure = dict(testset=None)
            dataset_structure['testset'] = pd.read_csv(dataset_list[0], sep=field_sep, names=metadata_fields, header=header)
            self.dataset_structure = dataset_structure

    def get_data_and_labels_dcase(self, data_path, features, fold_name, filetype='.npy'):
        """
        :param data_path: absolute path of the folder where the features are stored
        :param features: features type, contained in subfolder of data_path
        :param fold_num: index of the fold to load
        :param filetype: features file extension (e.g. htk, npy, ecc.)
        :param one_hot_vect: bool, if to return numerical labels or one hot vectors
        :return data: list of records [filename , matrix ]
        NOTE: data must have dimension ordering (Time, Coefficients)
        """

        fold_list = self.dataset_structure[fold_name]

        for file in fold_list.iterrows():
            filename = file[1]['filename']
            metadata_file = os.path.join(data_path, os.path.splitext(filename.replace('audio', 'meta'))[0] + '.ann')

            filename = os.path.basename(filename)
            filename = os.path.splitext(filename)[0]
            record = np.load(os.path.join(data_path, 'features', features, filename + filetype))
            label_matrix = self.get_event_labels(metadata_file, length=record.shape[1], dataset=self.year)

            num_axis = len(record.shape)
            if num_axis == 2:
                record = record.T
            elif num_axis == 3:
                record = np.swapaxes(record, 0, 1)

            if 'data' in locals():
                data = np.concatenate((data, record), axis=0)
            else:
                data = record

            if 'labels' in locals():
                labels = np.concatenate((labels, label_matrix), axis=0)
            else:
                labels = label_matrix

        return data, labels

    def get_event_labels(self, metadata_filepath,
                         time_resolution=0.02,
                         length=1501,
                         frame_context=1,
                         dataset=2017):

        if dataset == 2017:
            meta_names = ["filename", "background", "event_onset", "event_offset", "event_label", "mixture", "bf"]
        elif dataset == 2016:
            meta_names = ["event_onset", "event_offset", "event_label"]

        """Event roll

         Event roll is binary matrix indicating event activity withing time segment defined by time_resolution.

         Parameters
         ----------
         fold_number : Dataframe containing metadata
             Meta data
         class_labels : list
             List of labels in correct order
         time_resolution : float > 0.0
             Time resolution used when converting event into event roll.
             Default value "0.01"
         label : str
             Meta data field used to create event roll
             Default value "event_label"
         length : int, optional
             length of event roll, if none given max offset of the meta data is used.
             Default value "None"

         only_onset: bool, optional

            if True genera un vettore invece che una matrice, senza distinguere le classi

         return_sequences: bool, if False time resolution is scaled by frame context
         frame_context: int, time resolution scaling factor

         """

        dataset_list = pd.read_csv(metadata_filepath,
                                   names=meta_names,
                                   sep='\t',
                                   header='infer')
        max_offset_value = DatasetDCASE2017.max_event_offset(dataset_list['event_offset'])
        indices = list(dataset_list.index)
        # Initialize event roll
        event_roll = np.zeros(
            (int(math.ceil(max_offset_value * 1.0 / (time_resolution * frame_context)) + 1),
             len(self.class_labels)))
        event_roll = DatasetDCASE2017.pad(event_roll, length=length)

        for item in indices:
            # Fill-in event_roll
            event_onset = dataset_list.ix[item]['event_onset']
            event_offset = dataset_list.ix[item]['event_offset']
            event_label = dataset_list.ix[item]['event_label']
            if event_onset is not np.nan and event_offset is not np.nan:
                if event_label is not np.nan:
                    pos = self.class_labels.index(event_label)
                    onset = int(np.floor(event_onset * 1.0 / (time_resolution * frame_context)))
                    offset = int(np.ceil(event_offset * 1.0 / (time_resolution * frame_context)))

                    if offset > event_roll.shape[0]:
                        # we have event which continues beyond max_offset_value
                        offset = event_roll.shape[0]

                    if onset <= event_roll.shape[0]:
                        # We have event inside roll
                        event_roll[onset:offset, pos] = 1

        event_roll_bkg = np.logical_not(np.sum(event_roll[:, :], axis=1)).astype(int)
        event_roll_bkg = np.reshape(event_roll_bkg, (event_roll_bkg.shape[0],1))
        target_matrix = np.concatenate((event_roll_bkg, event_roll), axis=1)
        target_matrix = target_matrix.astype(np.int8)
        return target_matrix

    @staticmethod
    def max_event_offset(event_list):
        """Find the offset (end-time) of last event

        Parameters
        ----------
        event_list : list, shape=(n,)
            A list containing event dicts

        Returns
        -------
        max_offset: float > 0
            maximum offset

        """

        max_offset = 0
        for event in event_list:
            if event is not np.nan:
                if event > max_offset:
                    max_offset = event
        return max_offset

    @staticmethod
    def pad(sequence, length):
        """Pad event roll's length to given length

        Parameters
        ----------
        sequence
        length : int
            Length to be padded

        Returns
        -------
         sequence: np.ndarray, shape=(m,k)
            Padded event roll

        """

        if length > sequence.shape[0]:
            padding = np.zeros((length - sequence.shape[0], sequence.shape[1]))
            sequence = np.vstack((sequence, padding))

        elif length < sequence.shape[0]:
            sequence = sequence[0:length, :]

        return sequence

    @staticmethod
    def chunk_data(data, window_size, overlap_size=0, flatten_inside_window=True):
        assert data.ndim == 1 or data.ndim == 2
        if data.ndim == 1:
            data = data.reshape((-1, 1))

        # get the number of overlapping windows that fit into the data
        num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
        overhang = data.shape[0] - (num_windows * window_size - (num_windows - 1) * overlap_size)

        # if there's overhang, need an extra window and a zero pad on the data
        # (numpy 1.7 has a nice pad function I'm not using here)
        if overhang != 0:
            num_windows += 1
            newdata = np.zeros((num_windows * window_size - (num_windows - 1) * overlap_size, data.shape[1]))
            newdata[:data.shape[0]] = data
            data = newdata

        sz = data.dtype.itemsize
        ret = ast(
            data,
            shape=(num_windows, window_size * data.shape[1]),
            strides=((window_size - overlap_size) * data.shape[1] * sz, sz)
        )

        if flatten_inside_window:
            return ret
        else:
            return ret.reshape((num_windows, -1, data.shape[1]))


class DatasetDCASE2017Rare(Dataset):
    def __init__(self, class_labels=['background', 'babycry', 'glassbreak', 'gunshot']):
        logging.info("Loading Dataset")
        self.folds = []
        self.dataset_path = None
        self.dataset_structure = pd.DataFrame
        self.class_labels = class_labels

    def load_dataset_metadata(self, dataset_list=None,
                              metadata_fields=["filename", "event_onset", "event_offset", "event_label"],
                              field_sep='\t', header='infer'):
        """"
        loading file with metadata description and metadata
        :param dataset_list: list of paths of csv files which describes each dataset sample
        :param metadata_fields: columns of the csv file
        :param field_sep: row-element separator
        :param header (T/F)

        """
        if len(dataset_list) > 1:
            dataset_structure = dict(trainset=None, validset=None, testset=None)
            dataset_structure['trainset'] = pd.read_csv(dataset_list[0], sep=field_sep, names=metadata_fields, header=header)
            dataset_structure['validset'] = pd.read_csv(dataset_list[1], sep=field_sep, names=metadata_fields, header=header)
            dataset_structure['testset'] = pd.read_csv(dataset_list[2], sep=field_sep, names=metadata_fields, header=header)
            self.dataset_structure = dataset_structure
        else:
            dataset_structure = dict(testset=None)
            dataset_structure['testset'] = pd.read_csv(dataset_list[0], sep=field_sep, names=metadata_fields, header=header)
            self.dataset_structure = dataset_structure

    def get_data_and_labels_dcase(self, data_path, features, fold_name, filetype='.npy'):
        """
        :param data_path: absolute path of the folder where the features are stored
        :param features: features type, contained in subfolder of data_path
        :param fold_num: index of the fold to load
        :param filetype: features file extension (e.g. htk, npy, ecc.)
        :param one_hot_vect: bool, if to return numerical labels or one hot vectors
        :return data: list of records [filename , matrix ]
        NOTE: data must have dimension ordering (Time, Coefficients)
        """

        fold_list = self.dataset_structure[fold_name]

        for file in fold_list.iterrows():
            filename = file[1]['filename']
            filename = os.path.splitext(filename)[0]
            record = np.load(os.path.join(data_path, 'features', features, filename + filetype))

            if 'data' in locals():
                data = np.concatenate((data, record.T), axis=0)
            else:
                data = record.T

        label_matrix = self.get_event_labels(fold_list, length=record.shape[1], only_onset=True)

        return data, label_matrix

    def get_event_labels(self, dataset_list, time_resolution=0.02, length=1501, only_onset=False, single_out=False):

        """Event roll

         Event roll is binary matrix indicating event activity withing time segment defined by time_resolution.

         Parameters
         ----------
         """

        max_offset_value = DatasetDCASE2017Rare.max_event_offset(dataset_list['event_offset'])
        indices = list(dataset_list.index)
        target_matrix = []
        for item in indices:
            # Create event roll
            # Initialize event roll
            event_roll = np.zeros(
                (int(math.ceil(max_offset_value * 1.0 / time_resolution) + 1),
                 len(self.class_labels)))

            # Fill-in event_roll
            event_onset = dataset_list.ix[item]['event_onset']
            event_offset = dataset_list.ix[item]['event_offset']
            event_label = dataset_list.ix[item]['event_label']
            if event_onset is not np.nan and event_offset is not np.nan:
                if event_label is not np.nan:
                    pos = self.class_labels.index(event_label)
                    onset = int(np.floor(event_onset * 1.0 / time_resolution))
                    offset = int(np.ceil(event_offset * 1.0 / time_resolution))

                    if offset > event_roll.shape[0]:
                        # we have event which continues beyond max_offset_value
                        offset = event_roll.shape[0]

                    if onset <= event_roll.shape[0]:
                        # We have event inside roll
                        event_roll[onset:offset, pos] = 1

                        # TODO Pad event roll to full length of the signal
            # Create annotations for background class
            event_roll = DatasetDCASE2017Rare.pad(event_roll, length=length)
            event_roll[:, 0] = np.logical_not(np.sum(event_roll[:, 1:], axis=1)).astype(int)
            if only_onset:
                event_roll_onset = np.sum(event_roll[:, 1:], axis=1)
                if single_out:
                    target_matrix.append(event_roll_onset)
                else:
                    target_matrix.append(np.stack((event_roll[:, 0], event_roll_onset), axis=1))
            else:
                target_matrix.append(event_roll)
        target_matrix = np.asarray(target_matrix, dtype=np.uint8)
        return target_matrix

    @staticmethod
    def max_event_offset(event_list):
        """Find the offset (end-time) of last event

        Parameters
        ----------
        event_list : list, shape=(n,)
            A list containing event dicts

        Returns
        -------
        max_offset: float > 0
            maximum offset

        """

        max_offset = 0
        for event in event_list:
            if event is not np.nan:
                if event > max_offset:
                    max_offset = event
        return max_offset

    @staticmethod
    def pad(sequence, length):
        """Pad event roll's length to given length

        Parameters
        ----------
        sequence
        length : int
            Length to be padded

        Returns
        -------
         sequence: np.ndarray, shape=(m,k)
            Padded event roll

        """

        if length > sequence.shape[0]:
            padding = np.zeros((length - sequence.shape[0], sequence.shape[1]))
            sequence = np.vstack((sequence, padding))

        elif length < sequence.shape[0]:
            sequence = sequence[0:length, :]

        return sequence

    @staticmethod
    def chunk_data(data, window_size, overlap_size=0, flatten_inside_window=True):
        assert data.ndim == 1 or data.ndim == 2
        if data.ndim == 1:
            data = data.reshape((-1, 1))

        # get the number of overlapping windows that fit into the data
        num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
        overhang = data.shape[0] - (num_windows * window_size - (num_windows - 1) * overlap_size)

        # if there's overhang, need an extra window and a zero pad on the data
        # (numpy 1.7 has a nice pad function I'm not using here)
        if overhang != 0:
            num_windows += 1
            newdata = np.zeros((num_windows * window_size - (num_windows - 1) * overlap_size, data.shape[1]))
            newdata[:data.shape[0]] = data
            data = newdata

        sz = data.dtype.itemsize
        ret = ast(
            data,
            shape=(num_windows, window_size * data.shape[1]),
            strides=((window_size - overlap_size) * data.shape[1] * sz, sz)
        )

        if flatten_inside_window:
            return ret
        else:
            return ret.reshape((num_windows, -1, data.shape[1]))
