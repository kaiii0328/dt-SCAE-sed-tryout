#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:11:09 2017

@author: fabiov89
"""
import random

random.seed(888)
import numpy as np

np.random.seed(888)
import tensorflow as tf

tf.set_random_seed(888)
import os
import datetime
import sys
import logging
from capsule.capsulenet import *
import A3LAB_Framework.model.neural_network as nnet
import A3LAB_Framework.inout.dataset_dcase as io
import A3LAB_Framework.preprocessing.preprocessing as preproc
from A3LAB_Framework.utility.initialization import *
from A3LAB_Framework.utility.gpu_utils import get_gpu
from evaluation.scores import *

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

        if sys.platform == 'linux' and args.use_gpu:
            gpuID = get_gpu()
            logging.info("GPU ID:" + str(gpuID))

        fold_list = list(range(1, args.kfold + 1))
        # temporary folder to store completed data set fold
        temp_data_path = os.path.join(base_paths['root_path'], 'folded_data_' + args.class_labels
                                      + '_' + args.input_type)
        os.makedirs(temp_data_path, exist_ok=True)
        for fold in fold_list:
            dataset_list = [args.dataset_list + str(fold) + '_train.txt',
                            args.dataset_list + str(fold) + '_evaluate.txt',
                            args.dataset_list + str(fold) + '_test.txt']
            dataset = io.DatasetDCASE2017(args.class_labels)
            dataset.load_dataset_metadata(dataset_list=dataset_list)

            tmp_data_name_train = os.path.join(temp_data_path, 'fold_' + str(fold) + '_train.npy')
            tmp_label_name_train = os.path.join(temp_data_path, 'label_' + str(fold) + '_train.npy')
            tmp_data_name_eval = os.path.join(temp_data_path, 'fold_' + str(fold) + '_eval.npy')
            tmp_label_name_eval = os.path.join(temp_data_path, 'label_' + str(fold) + '_eval.npy')
            tmp_data_name_test = os.path.join(temp_data_path, 'fold_' + str(fold) + '_test.npy')
            tmp_label_name_test = os.path.join(temp_data_path, 'label_' + str(fold) + '_test.npy')
            if not (os.path.exists(tmp_data_name_train) and os.path.exists(tmp_label_name_train)):
                # TRAIN ###
                fold_data_train, fold_labels_train = dataset.get_data_and_labels_dcase(args.dataset_path,
                                                                                       features=args.input_type,
                                                                                       fold_name='trainset')
                np.save(tmp_data_name_train, fold_data_train)
                np.save(tmp_label_name_train, fold_labels_train)
                del fold_data_train, fold_labels_train
                # DEV ###
                fold_data_eval, fold_labels_eval = dataset.get_data_and_labels_dcase(args.dataset_path,
                                                                                     features=args.input_type,
                                                                                     fold_name='validset')
                np.save(tmp_data_name_eval, fold_data_eval)
                np.save(tmp_label_name_eval, fold_labels_eval)
                del fold_data_eval, fold_labels_eval

                # TEST ###
                fold_data_test, fold_labels_test = dataset.get_data_and_labels_dcase(args.dataset_path,
                                                                                     features=args.input_type,
                                                                                     fold_name='testset')
                np.save(tmp_data_name_test, fold_data_test)
                np.save(tmp_label_name_test, fold_labels_test)
                del fold_data_test, fold_labels_test

            else:
                logging.info('Temporary data files already exists!')

        results_dict = {'conf_id': base_paths['string_id'],
                        'valid_score': 0,
                        'test_score': 0,
                        'exp_time': 0}

        logging.info("------------------------BEGINNING CROSS VALIDATION------------------------------")
        validation_results = {'fold_' + str(key): {'best_score': 0, 'best_epoch': 0} for key in fold_list}
        average_validation_score = []
        if not args.evaluation_only:
            for test_fold in fold_list:
                logging.info(
                    '-----------------------------TEST FOLD {}------------------------------'.format(test_fold))
                fold_start_time = datetime.datetime.now()
                logging.info("Fold {} timestamp: ".format(test_fold) + fold_start_time.strftime('%Y-%m-%d %H:%M:%S'))
                ####################################################################################################
                if 'x_test' in locals():
                    del x_test, y_test

                x_test = np.load(os.path.join(temp_data_path, 'fold_' + str(test_fold) + '_test.npy'))
                y_test = np.load(os.path.join(temp_data_path, 'label_' + str(test_fold) + '_test.npy'))

                if 'x_dev' in locals():
                    del x_dev, y_dev

                x_dev = np.load(os.path.join(temp_data_path, 'fold_' + str(test_fold) + '_eval.npy'))
                y_dev = np.load(os.path.join(temp_data_path, 'label_' + str(test_fold) + '_eval.npy'))

                if 'x_train' in locals():
                    del x_train, y_train

                x_train = np.load(os.path.join(temp_data_path, 'fold_' + str(fold) + '_train.npy'))
                y_train = np.load(os.path.join(temp_data_path, 'label_' + str(fold) + '_train.npy'))
                ####################################################################################################
                ### data standardization ###
                if args.standardize_data:
                    if len(x_train.shape) == 2:
                        scaler = preproc.DataPreprocessing(data_dims_num=2)
                        train_dims = scaler.get_max_data_dimensions(x_train)

                        x_train = scaler.standardize_data(x_train)
                        x_dev = scaler.standardize_data(x_dev)
                        x_test = scaler.standardize_data(x_test)
                    else:
                        for channel in range(0, x_train.shape[2]):
                            scaler = preproc.DataPreprocessing(data_dims_num=2)
                            train_dims = scaler.get_max_data_dimensions(x_train[:, :, channel])

                            x_train[:, :, channel] = scaler.standardize_data(x_train[:, :, channel])
                            x_dev[:, :, channel] = scaler.standardize_data(x_dev[:, :, channel])
                            x_test[:, :, channel] = scaler.standardize_data(x_test[:, :, channel])

                # reshape into 3D data for CNN
                if len(x_train.shape) == 2:
                    x_train = np.expand_dims(
                        dataset.chunk_data(x_train, window_size=args.window_size, overlap_size=0,
                                           flatten_inside_window=False), axis=4)
                    x_dev = np.expand_dims(
                        dataset.chunk_data(x_dev, window_size=args.window_size, overlap_size=0,
                                           flatten_inside_window=False), axis=4)
                    x_test = np.expand_dims(
                        dataset.chunk_data(x_test, window_size=args.window_size, overlap_size=0,
                                           flatten_inside_window=False), axis=4)
                else:
                    for channel in range(0, x_train.shape[2]):
                        if 'x_train_chunked' in locals():
                            x_train_chunked = np.stack((x_train_chunked, dataset.chunk_data(x_train[:, :, channel],
                                                                                            window_size=args.window_size,
                                                                                            overlap_size=0,
                                                                                            flatten_inside_window=False)),
                                                       axis=3)
                        else:
                            x_train_chunked = dataset.chunk_data(x_train[:, :, channel], window_size=args.window_size,
                                                                 overlap_size=0, flatten_inside_window=False)

                        if 'x_dev_chunked' in locals():
                            x_dev_chunked = np.stack((x_dev_chunked, dataset.chunk_data(x_dev[:, :, channel],
                                                                                        window_size=args.window_size,
                                                                                        overlap_size=0,
                                                                                        flatten_inside_window=False)),
                                                     axis=3)
                        else:
                            x_dev_chunked = dataset.chunk_data(x_dev[:, :, channel], window_size=args.window_size,
                                                               overlap_size=0, flatten_inside_window=False)

                        if 'x_test_chunked' in locals():
                            x_test_chunked = np.stack((x_test_chunked, dataset.chunk_data(x_test[:, :, channel],
                                                                                          window_size=args.window_size,
                                                                                          overlap_size=0,
                                                                                          flatten_inside_window=False)),
                                                      axis=3)
                        else:
                            x_test_chunked = dataset.chunk_data(x_test[:, :, channel], window_size=args.window_size,
                                                                overlap_size=0, flatten_inside_window=False)

                    x_train = x_train_chunked
                    del x_train_chunked
                    x_dev = x_dev_chunked
                    del x_dev_chunked
                    x_test = x_test_chunked
                    del x_test_chunked

                args.n_classes = len(np.unique(np.argmax(y_train, 1)))
                y_train = dataset.chunk_data(y_train, window_size=args.window_size, overlap_size=0,
                                             flatten_inside_window=False)
                y_dev = dataset.chunk_data(y_dev, window_size=args.window_size, overlap_size=0,
                                           flatten_inside_window=False)
                y_test = dataset.chunk_data(y_test, window_size=args.window_size, overlap_size=0,
                                            flatten_inside_window=False)

                f_in_s = 50

                logging.info('x_train_shape = ' + str(x_train.shape))
                logging.info('y_train_shape = ' + str(y_train.shape))
                ####################################################################################################
                # Initialize network object
                fold_name = str(test_fold)
                logging.info('Processing configuration: ' + fold_name)
                save_dir = os.path.join(base_paths['experiment_folder'], 'fold_' + fold_name)
                # Build network architecture
                if not args.standard_CNN:
                    net = CapsuleNeuralNetwork(params=args, fold_id=fold_name, paths=base_paths)
                    model, eval_model = net.CapsNet(input_shape=x_train.shape[1:],
                                                    n_class=args.n_classes)
                    if args.fit_model:
                        if args.weights is not None:  # init the model weights with provided one
                            logging.info('Loading Pre-Trained Model from disk')
                            model.load_weights(args.weights)
                        net.train(model=model, train_data=(x_train, y_train), validation_data=(x_dev, y_dev),
                                  list_of_callback=args.callbacks)
                        if args.early_stopping:
                            validation_results['fold_' + str(test_fold)]['best_epoch'] = model.best_epoch + 1
                            logging.info('Saving epoch: ' + str(model.best_epoch))
                        logging.info("Fold training time (DAYS:HOURS:MIN:SEC):" + utils.GetTime(
                            datetime.datetime.now() - fold_start_time))
                        if not args.skip_predict:
                            logging.info('Loading Trained Model with best epoch from disk')
                            if args.early_stopping:
                                eval_model.load_weights(os.path.join(save_dir, 'trained_best_model.h5'))
                            else:
                                eval_model.load_weights(os.path.join(save_dir, 'trained_model.h5'))
                            predictions = net.test(model=eval_model, data=(x_test, y_test))
                            err_rate = sed_eval_metric(predictions, y_test)
                            logging.info('Devel Error Rate: ' + "{0:.2f}".format(err_rate['er_overall_1sec']))
                            logging.info('Devel F1 Score: ' + "{0:.2f}".format(err_rate['f1_overall_1sec'] * 100))
                            validation_results['fold_' + str(test_fold)]['best_score'] = err_rate
                            print_scores(validation_results, path=base_paths['experiment_folder'])
                    elif args.load_model:
                        logging.info('Loading Pre-Trained Model from disk')
                        if args.early_stopping:
                            eval_model.load_weights(os.path.join(save_dir, 'trained_best_model.h5'))
                        else:
                            eval_model.load_weights(os.path.join(save_dir, 'trained_model.h5'))
                        if not args.skip_predict:
                            predictions = net.test(model=eval_model, data=(x_test, y_test))
                            err_rate = sed_eval_metric(predictions, y_test)
                            logging.info('Devel Error Rate: ' + "{0:.2f}".format(err_rate['er_overall_1sec'] * 100))
                            logging.info('Devel F1 Score: ' + "{0:.2f}".format(err_rate['f1_overall_1sec'] * 100))
                            validation_results['fold_' + str(test_fold)]['best_score'] = err_rate
                            print_scores(validation_results, path=base_paths['experiment_folder'])
                else:
                    args.cnn_input_shape = x_train.shape[1:]
                    net = nnet.NeuralNetwork(params=args, fold_id=fold_name, paths=base_paths)
                    if args.fit_model:
                        # Build network architecture
                        net.define_cnn_static_timedistributed()
                        net.model_compile()
                        # Network training
                        net.model_fit(x_train, y_train, x_dev=x_dev, y_dev=y_dev)
                        if args.early_stopping:
                            validation_results['fold_' + str(test_fold)]['best_epoch'] = net._network.best_epoch + 1
                        logging.info("Fold training time (DAYS:HOURS:MIN:SEC):" + utils.GetTime(
                            datetime.datetime.now() - fold_start_time))
                        if not args.skip_predict:
                            if args.early_stopping:
                                net.load_model(
                                    model_path=os.path.join(base_paths['experiment_folder'], 'model_trained'),
                                    model_name='fold_' + str(test_fold) + '_best')
                            else:
                                net.load_model(
                                    model_path=os.path.join(base_paths['experiment_folder'], 'model_trained'),
                                    model_name='fold_' + str(test_fold))
                            predictions = net.model_predict(x_test)
                            err_rate = sed_eval_metric(predictions, y_test)
                            logging.info('Devel Error Rate: ' + "{0:.2f}".format(err_rate['er_overall_1sec']))
                            logging.info('Devel F1 Score: ' + "{0:.2f}".format(err_rate['f1_overall_1sec'] * 100))
                            validation_results['fold_' + str(test_fold)]['best_score'] = err_rate
                            print_scores(validation_results, path=base_paths['experiment_folder'])

                    elif args.load_model:
                        # Load trained model from disk and compute predictions
                        logging.info('Loading Trained Model from disk')
                        if args.early_stopping:
                            net.load_model(model_path=os.path.join(base_paths['root_path'], args.load_model),
                                           model_name='fold_' + str(test_fold) + '_best')
                        else:
                            net.load_model(model_path=os.path.join(base_paths['root_path'], args.load_model),
                                           model_name='fold_' + str(test_fold))
                        if not args.skip_predict:
                            # Generates output predictions for the input samples. If skip_predict = True, only model loading
                            net.name = 'fold_' + str(test_fold)
                            predictions = net.model_predict(x_test)
                            err_rate = sed_eval_metric(predictions, y_test)
                            logging.info('Devel Error Rate: ' + "{0:.2f}".format(err_rate['er_overall_1sec']))
                            logging.info('Devel F1 Score: ' + "{0:.2f}".format(err_rate['f1_overall_1sec'] * 100))
                            validation_results['fold_' + str(test_fold)]['best_score'] = err_rate
                            print_scores(validation_results, path=base_paths['experiment_folder'])
            # averaging partial scores
            average_fold_er_score = []
            average_fold_f1_score = []
            epochs_number = []
            for test_fold in fold_list:
                if validation_results['fold_' + str(test_fold)]:
                    average_fold_er_score.append(
                        validation_results['fold_' + str(test_fold)]['best_score']['er_overall_1sec'])
                    average_fold_f1_score.append(
                        validation_results['fold_' + str(test_fold)]['best_score']['f1_overall_1sec'])
                    epochs_number.append(int(validation_results['fold_' + str(test_fold)]['best_epoch']))
            best_epoch = validation_results['fold_' + str(np.argmax(average_fold_er_score) + 1)]['best_epoch']
            # best_epoch = np.max(epochs_number)
            mean_er_validation = np.mean(average_fold_er_score)
            mean_f1_validation = np.mean(average_fold_f1_score)
            logging.info('AVERAGE ER SCORE: {score} (%)'.format(score=mean_er_validation))
            logging.info('AVERAGE F1 SCORE: {score} (%)'.format(score=mean_f1_validation * 100))
            results_dict['valid_score'] = mean_er_validation
            results_dict['test_score'] = mean_f1_validation * 100
        ################################################################################################################
        elif args.evaluation_only:
            if not args.do_evaluation:
                # Only Averaged Final Scores
                overall_average = True
                if overall_average:
                    for test_fold in fold_list:
                        preds_file = os.path.join(base_paths['experiment_folder'], 'network_outputs',
                                                  'preds_fold_' + str(test_fold) + '.npy')
                        labels_file = os.path.join(os.path.join(temp_data_path, 'label_' + str(test_fold) + '_test.npy'))
                        if 'total_predictions' and 'total_labels' in locals():
                            total_predictions = np.concatenate((total_predictions, np.load(preds_file)))
                            fold_labels = dataset.chunk_data(np.load(labels_file), window_size=args.window_size,
                                                             overlap_size=0, flatten_inside_window=False)
                            total_labels = np.concatenate((total_labels, fold_labels))
                        else:
                            total_predictions = np.load(preds_file)
                            total_labels = dataset.chunk_data(np.load(labels_file), window_size=args.window_size,
                                                              overlap_size=0, flatten_inside_window=False)
                    final_results = sed_eval_metric(total_predictions, total_labels)
                    logging.info('AVERAGE ER SCORE: {score} (%)'.format(score=final_results['er_overall_1sec']))
                    logging.info('AVERAGE F1 SCORE: {score} (%)'.format(score=final_results['f1_overall_1sec'] * 100))
                    results_dict['valid_score'] = final_results['er_overall_1sec']
                    results_dict['test_score'] = final_results['f1_overall_1sec'] * 100
                else:
                    average_fold_er_score = []
                    average_fold_f1_score = []
                    for test_fold in fold_list:
                        logging.info(
                            '-----------------------------TEST FOLD {}------------------------------'.format(test_fold))
                        preds_file = np.load(os.path.join(base_paths['experiment_folder'], 'network_outputs',
                                                          'preds_fold_' + str(test_fold) + '.npy'))
                        labels_file = os.path.join(os.path.join(temp_data_path, 'label_' + str(test_fold) + '_test.npy'))
                        labels_file = dataset.chunk_data(np.load(labels_file), window_size=args.window_size, overlap_size=0,
                                                         flatten_inside_window=False)
                        err_rate = sed_eval_metric(preds_file, labels_file)
                        logging.info('Devel Error Rate: ' + "{0:.2f}".format(err_rate['er_overall_1sec']))
                        logging.info('Devel F1 Score: ' + "{0:.2f}".format(err_rate['f1_overall_1sec'] * 100))
                        validation_results['fold_' + str(test_fold)]['best_score'] = err_rate
                        average_fold_er_score.append(
                            validation_results['fold_' + str(test_fold)]['best_score']['er_overall_1sec'])
                        average_fold_f1_score.append(
                            validation_results['fold_' + str(test_fold)]['best_score']['f1_overall_1sec'])
                    mean_er_validation = np.mean(average_fold_er_score)
                    mean_f1_validation = np.mean(average_fold_f1_score)
                    logging.info('AVERAGE ER SCORE: {score} (%)'.format(score=mean_er_validation))
                    logging.info('AVERAGE F1 SCORE: {score} (%)'.format(score=mean_f1_validation * 100))
                    results_dict['valid_score'] = mean_er_validation
                    results_dict['test_score'] = mean_f1_validation * 100
        ################################################################################################################
        if args.do_evaluation:
            fold_start_time = datetime.datetime.now()
            logging.info("Evaluation timestamp: " + fold_start_time.strftime('%Y-%m-%d %H:%M:%S'))
            # del x_train, x_dev, x_test
            # del y_train, y_dev, y_test

            data_fold_list = ['train', 'test']

            for data_fold in data_fold_list:
                data_filename = os.path.join(temp_data_path, 'fold_1_' + data_fold + '.npy')
                label_filename = os.path.join(temp_data_path, 'label_1_' + data_fold + '.npy')
                if 'x_train' in locals():
                    x_train = np.concatenate((x_train, np.load(data_filename)))
                    y_train = np.concatenate((y_train, np.load(label_filename)))
                else:
                    x_train = np.load(data_filename)
                    y_train = np.load(label_filename)

            # TEST ###
            dataset.load_dataset_metadata(dataset_list=[args.evaluation_list])
            x_test, y_test = dataset.get_data_and_labels_dcase(args.evaluation_path,
                                                               features=args.input_type,
                                                               fold_name='testset')
            if args.standardize_data:
                if len(x_train.shape) == 2:
                    scaler_test = preproc.DataPreprocessing(data_dims_num=2)
                    train_dims = scaler_test.get_max_data_dimensions(x_train)
                    x_train = scaler_test.standardize_data(x_train)
                    x_test = scaler_test.standardize_data(x_test)
                else:
                    for channel in range(0, x_train.shape[2]):
                        scaler_test = preproc.DataPreprocessing(data_dims_num=2)
                        train_dims = scaler_test.get_max_data_dimensions(x_train[:, :, channel])
                        x_train[:, :, channel] = scaler_test.standardize_data(x_train[:, :, channel])
                        x_test[:, :, channel] = scaler_test.standardize_data(x_test[:, :, channel])
            if len(x_train.shape) == 2:
                x_train = np.expand_dims(
                    dataset.chunk_data(x_train, window_size=args.window_size, overlap_size=0, flatten_inside_window=False),
                    axis=4)
                x_test = np.expand_dims(
                    dataset.chunk_data(x_test, window_size=args.window_size, overlap_size=0, flatten_inside_window=False),
                    axis=4)
            else:
                for channel in range(0, x_train.shape[2]):
                    if 'x_train_chunked' in locals():
                        x_train_chunked = np.stack((x_train_chunked, dataset.chunk_data(x_train[:, :, channel],
                                                                                        window_size=args.window_size,
                                                                                        overlap_size=0,
                                                                                        flatten_inside_window=False)),
                                                   axis=3)
                    else:
                        x_train_chunked = dataset.chunk_data(x_train[:, :, channel], window_size=args.window_size,
                                                             overlap_size=0, flatten_inside_window=False)

                    if 'x_test_chunked' in locals():
                        x_test_chunked = np.stack((x_test_chunked, dataset.chunk_data(x_test[:, :, channel],
                                                                                      window_size=args.window_size,
                                                                                      overlap_size=0,
                                                                                      flatten_inside_window=False)),
                                                  axis=3)
                    else:
                        x_test_chunked = dataset.chunk_data(x_test[:, :, channel], window_size=args.window_size,
                                                            overlap_size=0, flatten_inside_window=False)

                x_train = x_train_chunked
                del x_train_chunked
                x_test = x_test_chunked
                del x_test_chunked

            args.n_classes = len(np.unique(np.argmax(y_train, 1)))
            y_train = dataset.chunk_data(y_train, window_size=args.window_size, overlap_size=0,
                                         flatten_inside_window=False)
            y_test = dataset.chunk_data(y_test, window_size=args.window_size, overlap_size=0,
                                        flatten_inside_window=False)

            f_in_s = 50
            evaluation_results = {'fold_eval': {'best_score': 0, 'best_epoch': 0}}
            err_rate = dict()
            ############################################################################################################
            # Initialize network object
            fold_name = 'eval'
            logging.info('Processing configuration: ' + fold_name)
            args.callbacks = ['TensorBoard', 'CSVLogger', 'LearningRateDecay']
            args.epochs = best_epoch
            # Build network architecture
            if not args.standard_CNN:
                net = CapsuleNeuralNetwork(params=args, fold_id=fold_name, paths=base_paths)
                model, eval_model = net.CapsNet(input_shape=x_train.shape[1:],
                                                n_class=args.n_classes)
                if args.fit_model_eval:
                    if args.weights is not None:  # init the model weights with provided one
                        logging.info('Loading Pre-Trained Model from disk')
                        model.load_weights(args.weights)
                    net.train(model=model, train_data=(x_train, y_train), list_of_callback=args.callbacks)
                    logging.info("Fold training time (DAYS:HOURS:MIN:SEC):" + utils.GetTime(
                        datetime.datetime.now() - fold_start_time))
                elif args.load_model_eval:
                    logging.info('Loading Pre-Trained Model from disk')
                    eval_model.load_weights(args.weights)
                if not args.skip_predict:
                    logging.info('Loading Trained Model with best epoch from disk')
                    save_dir = os.path.join(base_paths['experiment_folder'], 'fold_' + fold_name)
                    if not args.load_model_eval:
                        eval_model.load_weights(os.path.join(save_dir, 'trained_model.h5'))
                    predictions = net.test(model=eval_model, data=(x_test, y_test))
                    err_rate = sed_eval_metric(predictions, y_test)
                    logging.info('Test Error Rate: ' + "{0:.2f}".format(err_rate['er_overall_1sec']))
                    logging.info('Test F1 Score: ' + "{0:.2f}".format(err_rate['f1_overall_1sec'] * 100))
                    evaluation_results['fold_' + fold_name]['best_score'] = err_rate
                    print_scores(evaluation_results, path=base_paths['experiment_folder'], json_name='scores_eval.json')
            else:
                args.cnn_input_shape = x_train.shape[1:]
                net = nnet.NeuralNetwork(params=args, fold_id=fold_name, paths=base_paths)
                if args.fit_model_eval:
                    # Build network architecture
                    net.define_cnn_static_timedistributed()
                    net.model_compile()
                    # Network training
                    net.model_fit(x_train, y_train)
                    logging.info("Fold training time (DAYS:HOURS:MIN:SEC):" + utils.GetTime(
                        datetime.datetime.now() - fold_start_time))
                    if not args.skip_predict:
                        net.load_model(model_path=os.path.join(base_paths['experiment_folder'], 'model_trained'),
                                       model_name='fold_' + fold_name)
                        predictions = net.model_predict(x_test)
                        err_rate = sed_eval_metric(predictions, y_test)
                        logging.info('Test Error Rate: ' + "{0:.2f}".format(err_rate['er_overall_1sec']))
                        logging.info('Test F1 Score: ' + "{0:.2f}".format(err_rate['f1_overall_1sec'] * 100))
                        evaluation_results['fold_' + fold_name]['best_score'] = err_rate
                        print_scores(evaluation_results, path=base_paths['experiment_folder'],
                                     json_name='scores_eval.json')

                elif args.load_model_eval:
                    # Load trained model from disk and compute predictions
                    logging.info('Loading Trained Model from disk')
                    net.load_model(model_path=args.weights)
                    if not args.skip_predict:
                        # Generates output predictions for the input samples. If skip_predict = True, only model loading
                        net.name = 'fold_' + fold_name
                        predictions = net.model_predict(x_test)
                        err_rate = sed_eval_metric(predictions, y_test)
                        logging.info('Test Error Rate: ' + "{0:.2f}".format(err_rate['er_overall_1sec']))
                        logging.info('Test F1 Score: ' + "{0:.2f}".format(err_rate['f1_overall_1sec'] * 100))
                        evaluation_results['fold_' + fold_name]['best_score'] = err_rate
                        print_scores(evaluation_results, path=base_paths['experiment_folder'],
                                     json_name='scores_eval.json')

            results_dict_test = {'conf_id': base_paths['string_id'] + '_eval',
                                 'valid_score': err_rate['er_overall_1sec'],
                                 'test_score': err_rate['f1_overall_1sec'] * 100,
                                 'exp_time': utils.GetTime(datetime.datetime.now() - experiment_start_time)}
        ################################################################################################################
        # Finally
        experiment_time = utils.GetTime(datetime.datetime.now() - experiment_start_time)
        results_dict['exp_time'] = experiment_time
        score_report(base_paths['root_path'], results_dict)
        if args.do_evaluation:
            score_report(base_paths['root_path'], results_dict_test)
        logging.info("------------------------END EXPERIMENT-------------------------------")
        done = True
        logging.info('DONE')
        logging.info(
            "Experiment time (DAYS:HOURS:MIN:SEC):" + utils.GetTime(datetime.datetime.now() - experiment_start_time))
        utils.logcleaner()  # remove garbage character from log file

    except Exception:
        logging.exception("Unhandled exception:")

    finally:
        # Close Tensorflow session
        K.get_session().close()
        if not done:
            # if experiment is broken, trace on log file the index
            with open(os.path.join(base_paths['root_path'], 'Status_Processes_Report.txt'), 'a') as statusFile:
                statusFile.write('\nprocess_' + base_paths['string_id'] + ' ERROR\n')


if __name__ == '__main__':
    main()
