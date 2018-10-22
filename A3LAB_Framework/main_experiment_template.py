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
import model.neural_network as nnet
import inout.dataset as io
from inout.standardization import *
import inout.scores as score
from utility.initialization import *
from utility.gpu_utils import get_gpu
from sklearn.metrics import average_precision_score
sys.setrecursionlimit(10000)  # for deepcopy net model


def main():
    done = False  # status flag for this process
    base_paths, args = initialization()
    try:  # catch all unhandled error for evaluation experiment failure
        experiment_start_time = datetime.datetime.now()
        if args.expID != "":
            logging.info("Experiment {}.{} starts".format(args.confID, args.expID))
        else:
            logging.info("Experiment {} starts".format(args.confID))

        if sys.platform == 'linux':
            gpuID = get_gpu()
            logging.info("GPU ID:" + str(gpuID))

        temp_data_path = os.path.join(args.dataset_path, 'folded_data')   # temporary folder to store completed data set folds
        fold_list = list(range(0, args.kfold))

        if not args.evaluation_only:

            logging.info("------------------------BEGINNING CROSS VALIDATION------------------------------")

            session_results = {'fold_' + str(key): {'fold_' + str(key2): {'best_score': 0, 'best_epoch': 0} for key2 in fold_list} for key in fold_list}

            for test_fold in fold_list:
            # test_fold = 0
            # if test_fold == 0:

                logging.info('-----------------------------TEST FOLD {}------------------------------'.format(test_fold))
                fold_number = test_fold
                validation_list = list(set(fold_list) - set([test_fold]))
                # x_test = np.load(os.path.join(temp_data_path, 'fold_' + str(test_fold) + '.npy'))
                # y_test = np.load(os.path.join(temp_data_path, 'label_' + str(test_fold) + '.npy'))

                for dev_fold in validation_list:

                    logging.info('-----------------------------VALIDATION FOLD {}------------------------------'.format(dev_fold))
                    # timestamp info
                    fold_sub_number = dev_fold
                    fold_start_time = datetime.datetime.now()
                    logging.info("Fold {} timestamp: ".format(dev_fold) + fold_start_time.strftime('%Y-%m-%d %H:%M:%S'))
                    ########################################################################################################

                    training_list = list(set(validation_list) - set([dev_fold]))
                    x_dev = np.load(os.path.join(temp_data_path, 'fold_' + str(dev_fold) + '.npy'))
                    y_dev = np.load(os.path.join(temp_data_path, 'label_' + str(dev_fold) + '.npy'))

                    if 'x_train' in locals():
                        del x_train, y_train

                    for fold in training_list:

                        if 'x_train' in locals():
                            x_train = np.concatenate((x_train, np.load(os.path.join(temp_data_path, 'fold_' + str(fold) + '.npy'))), axis=0)
                            y_train = np.concatenate((y_train, np.load(os.path.join(temp_data_path, 'label_' + str(fold) + '.npy'))), axis=0)
                        else:
                            x_train = np.load(os.path.join(temp_data_path, 'fold_' + str(fold) + '.npy'))
                            y_train = np.load(os.path.join(temp_data_path, 'label_' + str(fold) + '.npy'))

                    ########################################################################################################
                    ### data standardization ###
                    # scaler = GlobalMinMaxScaler()
                    scaler = GlobalMeanStdScaler()
                    scaler.fit(x_train)
                    x_train = scaler.transform(x_train)
                    x_dev = scaler.transform(x_dev)
                    logging.info('x_train_shape = ' + str(x_train.shape))
                    logging.info('y_train_shape = ' + str(y_train.shape))
                    ########################################################################################################
                    # Initialize network object
                    if 'fold_sub_number' in locals() and isinstance(fold_sub_number, int):
                        fold_name = str(fold_number) + '.' + str(fold_sub_number)
                    else:
                        fold_name = str(fold_number)

                    logging.info('Processing configuration: ' + fold_name)

                    net = nnet.NeuralNetwork(params=args, fold_id=fold_name)
                    if args.fit_model:
                        # Build network architecture
                        # net.define_cnn_static(args)
                        if args.setup == "baseline":
                            net.define_cnn_static()
                        elif args.setup == "lavner":
                            net.define_cnn_lavner()
                        elif args.setup == "torres":
                            net.define_cnn_torres()
                        # net.name = 'fold_' + str(fold_number)
                        # net.model_compile(args)
                        net.model_compile()
                        # Network training
                        net.model_fit(x_train, y_train, x_dev=x_dev, y_dev=y_dev)
                        net.best_epoch = net._network.best_epoch + 1
                        session_results['fold_' + str(fold_number)]['fold_' + str(fold_sub_number)]['best_epoch'] = net.best_epoch
                        # TODO save best epoch value on disk

                        logging.info('Saving epoch: ' + str(net.best_epoch))
                        logging.info("Fold training time (DAYS:HOURS:MIN:SEC):" + utils.GetTime(datetime.datetime.now() - fold_start_time))

                    if not args.skip_predict:
                        logging.info('Loading Trained Model with best epoch from disk')
                        # Generates output predictions for input samples. If skip_predict = True, only model training
                        # TODO define param
                        test_batch_size = 1
                        # TODO ATTENTION: in case of early stop specify if load last epoch model or best
                        if args.load_model:
                            net.load_model(os.path.join(base_paths['root_path'], args.load_model + net.name + '_best.h5'))
                        else:
                            net.load_model(os.path.join(base_paths['experiment_folder'], 'model_trained', net.name + '_best.h5'))

                        net.name = 'fold_' + fold_name
                        # net.model_predict(x_dev, batch_size=test_batch_size, predict_classes=True)
                        predictions = net.model_predict(x_dev, batch_size=test_batch_size)
                        np.save(os.path.join(base_paths['experiment_folder'], 'network_outputs', 'labels_' + fold_name + '.npy'), y_dev)

                        # TODO load best epoch value from disk
                        score_value = average_precision_score(y_dev, predictions, average='macro')
                        session_results['fold_' + str(fold_number)]['fold_' + str(fold_sub_number)]['best_score'] = score_value
                        logging.info('Saving score: ' + str(score_value))

            score.log_validation_scores(session_results, fold_list)
        ready_for_evaluation = True
        if args.skip_predict:
            ready_for_evaluation = False

        ################################################################################################################
        # EVALUATION
        if ready_for_evaluation:
            logging.info("------------------------EVALUATION PHASE-------------------------")
            # init structures  containing the scores for each fold
            # key_idx = range(1, len(dataset.folds) + 1)
            key_idx = fold_list
            results = {'fold_' + str(key): {'metrics': None, 'data': None} for key in key_idx}
            results.update({'average': {'overall': None, 'fold_based': None}})

            fold_data = {
                'predictions': [],
                'labels': []
            }
            fold_result = {
                'Score': []
            }
            mean_result = {
                'Score': []
            }
            overall_result = {
                'Score': []
            }

            for test_fold in fold_list:
            # test_fold = 0
            # if test_fold == 0:
                validation_list = list(set(fold_list) - set([test_fold]))
                logging.info('-----------------------------TEST FOLD {}------------------------------'.format(test_fold))
                fold_result['Score'] = 0
                results['fold_' + str(test_fold)]['metrics'] = fold_result.copy()
                for dev_fold in validation_list:
                    fold_name = str(test_fold) + '.' + str(dev_fold)
                    logging.info('-----------------------------VALIDATION FOLD {}------------------------------'.format(dev_fold))
                    # load predictions from disk
                    predictions_folder = os.path.join(base_paths['experiment_folder'], 'network_outputs')  # network output folder
                    predictions = np.load(os.path.join(predictions_folder, 'preds_fold_' + fold_name + '.npy'))
                    predictions = predictions.reshape([1, len(predictions)]).tolist()[0]
                    # get labels
                    labels = np.load(os.path.join(predictions_folder, 'labels_' + fold_name + '.npy'))[0:, 0].tolist()
                    # compute scores and store into results structure

                    # TODO load best epoch value from disk

                    fold_result['Score'] = average_precision_score(labels, predictions, average='macro')

                    # store fold data and labels into results structure
                    fold_data['predictions'] = predictions
                    fold_data['labels'] = labels

                    if fold_result['Score'] > results['fold_' + str(test_fold)]['metrics']['Score']:
                        results['fold_' + str(test_fold)]['metrics'] = fold_result.copy()
                        results['fold_' + str(test_fold)]['data'] = fold_data
                        logging.info('Fold {fold} new best Average Prediction Score: {score}'.format(fold=test_fold, score=fold_result['Score']))

                    # TODO Write partial results on DISK - DataBase - TEXT FILE

            logging.info("------------------------END EVALUATION PHASE--------------------------")
            ############################################################################################################
            # Compute averaged metrics -- #TODO choose strategy
            # fold based

            ap_score = []
            for key in key_idx:
                if results['fold_' + str(key)]['metrics']:
                    ap_score.append(results['fold_' + str(key)]['metrics']['Score'])

            mean_result['Score'] = np.mean(ap_score)
            results['average']['fold_based'] = mean_result.copy()
            logging.info('Fold Mean Average Prediction Score: {score}'.format(score=mean_result))

            # Compute averaged metrics
            # overall
            total_preds = []
            total_labels = []
            for key in key_idx:
                if results['fold_' + str(key)]['data']:
                    total_preds.extend(results['fold_' + str(key)]['data']['predictions'])
                    total_labels.extend(results['fold_' + str(key)]['data']['labels'])
            overall_result['Score'] = average_precision_score(total_labels, total_preds, average='macro')
            results['average']['overall'] = overall_result
            logging.info('Overall Average Prediction Score: {score}'.format(score=overall_result))

            # Finally
            experiment_time = utils.GetTime(datetime.datetime.now() - experiment_start_time)
            # save most interesting metrics on Scores_Report.csv
            score.print_AP_scores(results['average']['fold_based']['Score'], results['average']['overall']['Score'], experiment_time)


        ################################################################################################################
        logging.info("------------------------END EXPERIMENT-------------------------------")
        done = True
        logging.info('DONE')
        logging.info("Experiment time (DAYS:HOURS:MIN:SEC):" + utils.GetTime(datetime.datetime.now() - experiment_start_time))
        utils.logcleaner()  # remove garbage character from log file

    except Exception:
        logging.exception("Unhandled exception:")

    finally:
        if not done:
            # if experiment is broken, trace on log file the index
            with open(os.path.join(base_paths['root_path'], 'Status_Processes_Report.txt'), 'a') as statusFile:
                statusFile.write('\nprocess_' + base_paths['string_id'] + ' ERROR\n')


if __name__ == '__main__':
    main()
