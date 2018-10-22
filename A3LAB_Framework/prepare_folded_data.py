#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:11:09 2017

@author: fabiov89
"""
import numpy as np
np.random.seed(888)  # for experiment repetibility: this goes here, before importing keras. It works?
import os
import datetime
import sys
import logging
import json
import inout.dataset as io
from inout.standardization import *
from utility.initialization import *
from utility.custom_parser import args
from utility.gpu_utils import get_gpu
sys.setrecursionlimit(10000)  # for deepcopy net model


def main():
    done = False  # status flag for this process
    try:  # catch all unhandled error for evaluation experiment failure
        experiment_start_time = datetime.datetime.now()
        logging.info("Experiment {}.{} starts".format(args.confID, args.expID))

        if sys.platform == 'linux':
            gpuID = get_gpu()
            logging.info("GPU ID:" + str(gpuID))

        #########################################SAVING EXPERIMENT PARAMETERS###########################################
        # # todo implementare salvataggio su db, ora fa solo il dump su json
        if not args.db_save:
            json_args = json.dumps(args.__dict__, indent=4)
            with open(os.path.join(EXP_FOLDER, 'args' + STRING_ID + '.json'), 'w') as args_file:
                args_file.write(json_args)
        ################################################################################################################
        # MANAGE DATASET
        # Initialize dataset class
        dataset = io.DatasetCry()

        logging.info("------------------------DATA SET LOAD AND FOLDED DATA GENERATION------------------------------")

        snr = args.snr
        scenario = args.scenario
        working_path = os.path.join(args.dataset_path, snr, scenario)

        clean_path = [os.path.join(args.dataset_path, "pianti")]
        files_path = []

        if not args.use_clean:
            dir_list = os.listdir(working_path)
            for directory in dir_list:
                if directory == "pianti":
                    continue
                files_path += [os.path.join(working_path, directory)]

        if not args.use_clean:
            curr_path = files_path
        else:
            curr_path = clean_path

        if curr_path == clean_path:
            dataset.load_dataset_metadata(None, None, args.kfold_list_filepath, scenario='pianti')
        else:
            dataset.load_dataset_metadata(None, None, args.kfold_list_filepath, scenario=scenario)

        fold_list = list(range(0, args.kfold))

        temp_data_path = os.path.join(args.dataset_path, 'folded_data')   # temporary folder to store completed data set folds
        if not os.path.exists(temp_data_path):
            os.makedirs(temp_data_path)

        if args.fit_model:
            for fold in fold_list:
                np.save(os.path.join(temp_data_path, 'fold_' + str(fold) + '.npy'), dataset.get_data(curr_path, 'mel', [args.cxt, args.str], dataset.folds[fold]))
                np.save(os.path.join(temp_data_path, 'label_' + str(fold) + '.npy'), dataset.get_labels(curr_path, 'mel', None, dataset.folds[fold]))


        ################################################################################################################
        logging.info("------------------------FOLDED DATA SAVED-------------------------------")
        done = True
        logging.info('DONE')
        logging.info("Fold creation time (DAYS:HOURS:MIN:SEC): " + utils.GetTime(datetime.datetime.now() - experiment_start_time))
        logging.info("Folded data and labels saved in : " + temp_data_path)
        utils.logcleaner()  # remove garbage character from log file

    except Exception:
        logging.exception("Unhandled exception:")

    finally:
        if not done:
            # if experiment is broken, trace on log file the index
            with open(os.path.join(ROOT_PATH, 'Status_Processes_Report.txt'), 'a') as statusFile:
                statusFile.write('\nprocess_' + STRING_ID + ' ERROR\n')


if __name__ == '__main__':
    main()
