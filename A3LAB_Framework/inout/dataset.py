#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:36:52 2017

@author: fabiov89
"""
import logging
import numpy as np
from tensorflow.python.keras.utils import to_categorical
from sklearn.model_selection import KFold
import pandas as pd


class Dataset:
    def __init__(self):
        """
        initialization dataset class
        """
        logging.info("Loading Dataset")
        self.folds = []
        self.dataset_path = None
        self.dataset_structure = pd.DataFrame

    def load_dataset_metadata(self):
        pass

    def get_data(self, data_path, features, filetype, fold_list, transpose=False):
        pass

    def dataset_kfold_split(self, kfolds):
        """"
        split dataset structure into k-folds for cross validation
        """
        kf = KFold(n_splits=kfolds)
        dataset_indices = list(self.dataset_structure.index)
        indices = np.asarray(dataset_indices)
        fold_list = list(train_index for train_index in kf.split(indices))
        cv_folds = []
        for fold in fold_list:
            trainset_list = self.dataset_structure.ix[fold[0]]
            develset_list = self.dataset_structure.ix[fold[1]]
            cv_folds.append([trainset_list, develset_list])
            self.folds = cv_folds
        return cv_folds

    @staticmethod
    def get_categorical_labels(dataset_list, class_labels=None, one_hot_vect=True):
        """
        :param dataset_list: list of records
        :param class_labels: list of class labels
        :param one_hot_vect (bool): choose if targets are a matrix of one hot vector or a list of class labels index
        :return: targets: list of labels for respective list of records
        """
        indices = list(dataset_list.index)
        targets = []
        for item in indices:
            event_label = dataset_list.ix[item]['class']
            pos = class_labels.index(event_label)
            targets.append(pos)
        if one_hot_vect:
            targets = to_categorical(targets, num_classes=len(class_labels))
        return targets

    @staticmethod
    def get_binary_labels(dataset_list, single_class=None, class_labels=None, one_hot_vect=True):
        """
        :param dataset_list: list of records
        :param single_class: (char) label of
        :param class_labels: list of class labels ONE vs rest class
        :param one_hot_vect (bool): choose if targets are a matrix of one hot vector or a list of class labels index
        :return: targets: list of labels for respective list of records
        """
        indices = list(dataset_list.index)
        targets = []
        for item in indices:
            event_label = dataset_list.ix[item]['class']
            if event_label == single_class:
                pos = 1
            else:
                pos = 0
            targets.append(pos)
        if one_hot_vect:
            targets = to_categorical(targets, num_classes=2)
        return targets


