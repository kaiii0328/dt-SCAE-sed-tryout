#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:43:32 2017

@author: fabiov89
"""
import numpy as np

np.random.seed(888)  # for experiment repetibility: this goes here, before importing keras. It works?
from tensorflow.python.keras.models import Model, load_model, model_from_json
from tensorflow.python.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, BatchNormalization, AveragePooling2D, \
    concatenate, SimpleRNN, LSTM, GRU, Reshape, LeakyReLU, GaussianNoise, MaxPooling2D, Concatenate, TimeDistributed, \
    Permute, \
    Conv1D, Activation, MaxPooling1D, Lambda
from tensorflow.python.keras.optimizers import Adadelta, Adam, RMSprop, SGD
from tensorflow.python.keras.regularizers import *
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard, \
    LearningRateScheduler
import math
import json
import logging
from sklearn.utils import class_weight
from A3LAB_Framework.utility import utility as utils
import os
from evaluation.scores import compute_scores, sed_eval_metric, compute_auc, mono_sed_scores
import time
from .custom_loss_functions import *


class NeuralNetwork:
    def __init__(self, params=None, paths=[], fold_id=[], dataframe=None):
        """
        initialize model class
        """
        self.params = params
        self._network = None
        self.name = 'fold_' + str(fold_id)
        self.class_weight = []
        self.best_epoch = 0
        self.paths = paths
        self.batch_size = 0
        self.dataframe = dataframe

    def define_static_arch(self):
        """
        E' TEMPORANEA:QUESTA FUNZIONE VA ELIMINATA ALLA FINE
        QUESTa è usata solo per bypassare la creazione dinamica che vuole tutti i parametri!
        """
        logging.info('define TEST arch ')
        d = self.params.cnn_input_shape[0]
        h = self.params.cnn_input_shape[1]
        w = self.params.cnn_input_shape[2]
        logging.info("(" + str(d) + ", " + str(h) + ", " + str(w) + ")")

        input_data = Input(shape=self.params.cnn_input_shape)
        x = input_data
        x = Conv2D(filters=4,
                   kernel_size=[5, 5],
                   padding=self.params.border_mode,
                   activation='relu',
                   use_bias=self.params.bias)(x)

        x = MaxPooling2D(pool_size=(5, 1), strides=None, padding='same',
                         data_format='channels_first')(x)
        z = TimeDistributed(Flatten())(x)

        z = TimeDistributed(Dense(10,
                                  activation='relu',
                                  use_bias=self.params.bias))(z)
        predictions = Dense(1, activation='sigmoid')(z)

        self._network = Model(input_data, predictions)
        self._network.summary()

        return self._network

    def define_cnn_static(self):
        logging.info("define_CNN_classifier")
        input_data = Input(shape=self.params.cnn_input_shape)
        x = input_data
        for i in range(len(self.params.kernel_number)):
            x = Conv2D(filters=self.params.kernel_number[i],
                       kernel_size=self.params.kernel_shape[i],
                       kernel_initializer=self.params.cnn_weight_init,
                       activation=self.params.cnn_activation,
                       padding=self.params.border_mode,
                       strides=self.params.cnn_strides[i],
                       use_bias=self.params.bias)(x)
            if self.params.batch_norm:
                x = BatchNormalization(axis=-1)(x)
            if self.params.leaky_relu:
                x = LeakyReLU(alpha=self.params.leaky_relu_alpha)(x)
            if self.params.dropout_cnn:
                x = Dropout(self.params.drop_rate_cnn)(x)
            x = MaxPooling2D(pool_size=(tuple(self.params.pool_shapes[i])), strides=self.params.pool_strides[i],
                             padding=self.params.pooling_border_mode)(x)
        x = Flatten()(x)
        for i in range(len(self.params.dense_layer_shapes)):
            x = Dense(self.params.dense_layer_shapes[i],
                      kernel_initializer=self.params.dense_weight_init,
                      activation=self.params.dense_activation,
                      kernel_regularizer=eval(self.params.d_w_reg),
                      use_bias=self.params.bias)(x)
            if self.params.leaky_relu:
                x = LeakyReLU(alpha=self.params.leaky_relu_alpha)(x)
            if self.params.dropout_dense:
                x = Dropout(self.params.drop_rate_dense)(x)
            if self.params.batch_norm:
                x = BatchNormalization(axis=1)(x)

        x = Dense(self.params.n_classes,
                  activation=self.params.output_activation,
                  kernel_regularizer=eval(self.params.d_w_reg))(x)
        if self.params.dropout_dense:
            x = Dropout(self.params.drop_rate_dense)(x)

        predictions = x

        self._network = Model(input_data, predictions)
        self._network.summary()

        return self._network

    def define_cnn_static_timedistributed(self):
        logging.info("define_CNN_classifier")
        input_data = Input(shape=self.params.cnn_input_shape)
        x = input_data
        for i in range(len(self.params.kernel_number)):
            x = Conv2D(filters=self.params.kernel_number[i],
                       kernel_size=self.params.kernel_shape[i],
                       kernel_initializer=self.params.cnn_weight_init,
                       activation=self.params.cnn_activation,
                       padding=self.params.border_mode,
                       strides=self.params.cnn_strides[i],
                       use_bias=self.params.bias)(x)
            if self.params.batch_norm:
                x = BatchNormalization(axis=-1)(x)
            if self.params.leaky_relu:
                x = LeakyReLU(alpha=self.params.leaky_relu_alpha)(x)
            if self.params.dropout_cnn:
                x = Dropout(self.params.drop_rate_cnn)(x)
            x = MaxPooling2D(pool_size=(tuple(self.params.pool_shapes[i])), strides=self.params.pool_strides[i],
                             padding=self.params.pooling_border_mode)(x)
        x = TimeDistributed(Flatten())(x)
        for i in range(len(self.params.dense_layer_shapes)):
            x = TimeDistributed(Dense(self.params.dense_layer_shapes[i],
                                      kernel_initializer=self.params.dense_weight_init,
                                      activation=self.params.dense_activation,
                                      kernel_regularizer=eval(self.params.d_w_reg),
                                      use_bias=self.params.bias))(x)
            if self.params.leaky_relu:
                x = LeakyReLU(alpha=self.params.leaky_relu_alpha)(x)
            if self.params.dropout_dense:
                x = Dropout(self.params.drop_rate_dense)(x)
            if self.params.batch_norm:
                x = BatchNormalization(axis=1)(x)

        x = TimeDistributed(Dense(self.params.n_classes,
                                  activation=self.params.output_activation,
                                  kernel_regularizer=eval(self.params.d_w_reg)))(x)
        if self.params.dropout_dense:
            x = Dropout(self.params.drop_rate_dense)(x)

        predictions = x

        self._network = Model(input_data, predictions)
        self._network.summary()

        return self._network

    def model_compile(self, ):
        """
        compila il modello con i parametri passati: se non viene passato compila il modello istanziato dalla classe

        :param model:
        :param optimizer:
        :param learning_rate:
        :param loss:
        :return:
        """

        logging.info("Model_compile")

        # set optimizer
        if self.params.optimizer == "adadelta":
            opti = Adadelta(lr=self.params.learning_rate, rho=0.95, epsilon=1e-06)
        elif self.params.optimizer == "adam":
            opti = Adam(lr=self.params.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None,
                        decay=0.0)
        elif self.params.optimizer == "RMSprop":
            opti = RMSprop(decay=self.params.lr_decay)
        elif self.params.optimizer == "sgd":
            opti = SGD(lr=self.params.learning_rate, decay=self.params.lr_decay,
                       momentum=self.params.momentum,
                       nesterov=True)
        else:
            logging.info("Setting Default Optimizer: ADAM - lr 0.001")
            opti = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                        decay=0.0)
        # set loss
        if self.params.loss == "custom":
            if self.params.custom_loss == "contrastive_loss":
                loss = contrastive_loss
        else:
            loss = self.params.loss

        self._network.compile(optimizer=opti, loss=loss, metrics=self.params.metrics)
        return self._network

    def model_fit(self, x_train, y_train, x_dev=None, y_dev=None):
        '''

        :param x_train:
        :param y_train:
        :param x_dev:
        :param y_dev:
        :param list_of_callback: string callback name or fuction to use as callback. If a string is present,
         the callback is initialized inside this function with standar parametr (like destination path)
        :return:
        '''
        logging.info('Training model...')

        self.batch_size = self.params.batch_size

        # initalize folder with models and losses report files
        model_folder = os.path.join(self.paths['experiment_folder'], 'model_trained')
        utils.makedir(model_folder)
        train_loss_csv = os.path.join(model_folder, self.name + '.csv')
        unique_classes = self.params.n_classes

        if not self.params.balance_classes:
            self.class_weight = None
        else:
            c_weight = class_weight.compute_class_weight('balanced', classes=unique_classes,
                                                         y=np.argmax(y_train, axis=1))
            c_weight_dict = {cls: c_weight[cls] for cls in unique_classes}
            self.class_weight = c_weight_dict

        # keras callback definition

        callbacks = list()
        if 'TensorBoard' in self.params.callbacks:
            tensorboard = TensorBoard(
                log_dir=os.path.join(self.paths["experiment_folder"], "tensorboard_logs/{}".format(time.time())),
                histogram_freq=0,
                write_graph=True, write_images=True)
            callbacks.append(tensorboard)

        if 'CSVLogger' in self.params.callbacks:
            csv_logger = CSVLogger(train_loss_csv)
            callbacks.append(csv_logger)

        if 'CustomEarlyStopping' in self.params.callbacks:
            custom_early_stopping = CustomEarlyStopping(patience=self.params.patience,
                                                        path=os.path.join(model_folder, self.name + '_best.h5'),
                                                        batch_size=self.batch_size)
            callbacks.append(custom_early_stopping)

        if 'SEDEarlyStopping' in self.params.callbacks:
            custom_early_stopping = SEDEarlyStopping(patience=self.params.patience,
                                                     path=os.path.join(model_folder, self.name + '_best.h5'),
                                                     batch_size=self.batch_size,
                                                     resolution=self.params.resolution)
            callbacks.append(custom_early_stopping)

        if 'MonoSEDEarlyStopping' in self.params.callbacks:
            custom_early_stopping = MonoSEDEarlyStopping(patience=self.params.patience,
                                                     path=os.path.join(model_folder, self.name + '_best.h5'),
                                                     batch_size=self.batch_size,
                                                     dataframe=self.dataframe)
            callbacks.append(custom_early_stopping)

        if 'AUCEarlyStopping' in self.params.callbacks:
            custom_early_stopping = AUCEarlyStopping(patience=self.params.patience,
                                                     path=os.path.join(model_folder, self.name + '_best.h5'),
                                                     batch_size=self.batch_size)
            callbacks.append(custom_early_stopping)

        if 'LearningRateDecay' in self.params.callbacks:
            lr_decay = LearningRateScheduler(
                schedule=lambda epoch: self.params.learning_rate * (self.params.lr_decay ** epoch))
            callbacks.append(lr_decay)

        if 'ModelCheckPoint' in self.params.callbacks:
            checkpoint = ModelCheckpoint(model_folder + '/trained_model_best.h5', monitor='val_acc',
                                         save_best_only=True, save_weights_only=True, verbose=1)
            callbacks.append(checkpoint)

        if self.params.train_on_batch:
            # if I want to load data in memory batch-by-batch (for big models/dataset)

            if (x_dev is not None and y_dev is not None) or self.params.validation_split > 0.0:
                if self.params.validation_split > 0.0 and (x_dev is None and y_dev is None):
                    # compute validation set from validation split parameter
                    indices = range(0, len(x_train))
                    np.random.shuffle(indices)
                    marker = int(math.ceil(len(indices) * self.params.validation_split))
                    x_dev = x_train[indices[0:marker], :]
                    y_dev = y_train[indices[0:marker], :]
                    x_train_split = x_train[indices[marker:], :]
                    y_train_split = y_train[indices[marker:], :]

                    del x_train
                    del y_train

                    generator_valid = NeuralNetwork.generate_batch(x_dev, y_dev, self.batch_size)
                    generator_train = NeuralNetwork.generate_batch(x_train_split, y_train_split, self.batch_size)
                    train_steps = x_train_split.shape[0] / self.batch_size
                    validation_steps = x_dev.shape[0] / self.batch_size

                else:
                    # I pass validation set from external
                    generator_valid = NeuralNetwork.generate_batch(x_dev, y_dev, self.batch_size)
                    generator_train = NeuralNetwork.generate_batch(x_train, y_train, self.batch_size)
                    validation_steps = x_dev.shape[0] / self.batch_size

                self._network.fit_generator(generator_train,
                                            steps_per_epoch=train_steps,
                                            epochs=self.params.epochs,
                                            callbacks=callbacks,
                                            validation_data=generator_valid,
                                            validation_steps=validation_steps,
                                            initial_epoch=0,
                                            max_q_size=1,
                                            class_weight=self.class_weight,
                                            verbose=2)
            else:
                # model training ends when specified number of epochs are completed
                generator_train = NeuralNetwork.generate_batch(x_train, y_train, self.batch_size)
                train_steps = x_train.shape[0] / self.batch_size
                self._network.fit_generator(generator_train,
                                            steps_per_epoch=train_steps,
                                            epochs=self.params.epochs,
                                            callbacks=[csv_logger],
                                            initial_epoch=0,
                                            max_q_size=1,
                                            class_weight=self.class_weight,
                                            verbose=2)

        else:
            # Load all dataset in memory
            if (x_dev is not None and y_dev is not None) or self.params.validation_split > 0.0:

                if self.params.validation_split > 0.0 and (
                        x_dev is None and y_dev is None):  # todo rimuovere questa condizione perche keras da priprità  gia a x_dev e y_dev
                    self._network.fit(x_train, y_train,
                                      epochs=self.params.epochs,
                                      validation_split=self.params.validation_split,
                                      batch_size=self.batch_size,
                                      shuffle=self.params.shuffle,
                                      callbacks=callbacks,
                                      class_weight=self.class_weight,
                                      verbose=2)  # with a value != 1 ProbarLogging is not called
                else:
                    self._network.fit(x_train, y_train,
                                      epochs=self.params.epochs,
                                      validation_data=(x_dev, y_dev),
                                      batch_size=self.batch_size,
                                      shuffle=self.params.shuffle,
                                      callbacks=callbacks,
                                      class_weight=self.class_weight,
                                      verbose=2)  # with a value != 1 ProbarLogging is not called

            else:
                # model training ends when specified number of epochs are completed
                self._network.fit(x_train, y_train,
                                  epochs=self.params.epochs,
                                  batch_size=self.batch_size,
                                  shuffle=self.params.shuffle,
                                  callbacks=callbacks,
                                  class_weight=self.class_weight,
                                  verbose=2)

        if self.params.save_model:
            logging.info('saving model of last epoch ...')
            self.save_model(model_path=model_folder, model_name=self.name)
            logging.info('model saved')

        return self._network

    def load_model(self, model_path='.', model_name=None, method="model"):
        """
        load from disk the model
        :param model_path:
        :param model_name:
        :param method choice of [model, weights]. use model to load a previously saved model with model.save().
        use weights to load model from json and load weights after. Useful when custom loss( or custom layer) are specifeid in the network
        :return:
        """
        if model_name:
            full_model_path = os.path.join(model_path, model_name)
        else:
            full_model_path = model_path
        logging.info('Loading model from: ' + full_model_path)
        if method == "model":
            self._network = load_model(full_model_path + '.h5')
        elif method == "weights":
            # self._network = model_from_json(open(full_model_path + '.json').read())
            self._network.load_weights(full_model_path + '.h5')

        return self._network

    def save_model(self, model_path='.', model_name='my_model'):
        """
        Salva json, model e i weights  del modello che è istanziato attualmente nella classe

        :param model_path:
        :param model_name:
        :return:
        """
        model = self._network

        model.save(os.path.join(model_path, model_name + '.h5'))
        #model.save_weights(os.path.join(model_path, model_name + '_w.h5'))

        json_string = model.to_json()
        with open(os.path.join(model_path, model_name + '.json'), "w") as text_file:
            text_file.write(json.dumps(json_string, indent=4, sort_keys=True))

    def model_predict(self, x_test, labels=None, sequence_length=None, predict_classes=False,
                      save_preds=True):
        """
        Decode the input data

        :param x_test: The data to be decoded
        :param sequence_length: (int)
        :param model: model to use for decode the input data. If None a self model of class is used
        :param predict_classes: compute categorical outputs for classification problem (tipically with softmax last layer)
        :param save_preds:  (bool) save predictions on disk
        :return: the decoded data
        """
        logging.info("Computing network outputs")

        if self.batch_size == 0:
            # batch_size = NeuralNetwork.get_batch_size(x_test, self.params)
            batch_size = self.params.batch_size
            self.batch_size = batch_size

        outputs = self._network.predict(x_test, self.batch_size)

        if sequence_length is not None:
            outputs = outputs.reshape(-1, sequence_length, 1)

        if predict_classes:
            for i in range(0, outputs.shape[0]):
                outputs = np.argmax(outputs, axis=1)

        # Saving outputs on disk/DB (or HOW?)
        if save_preds:
            predictions_folder = os.path.join(self.paths['experiment_folder'],
                                              'network_outputs')  # network output folder
            utils.makedir(predictions_folder)
            np.save(os.path.join(predictions_folder, 'preds_' + self.name + '.npy'), outputs)
            if labels is not None:
                np.save(os.path.join(predictions_folder, 'true_labels_' + self.name + '.npy'), labels)

        return outputs

    @staticmethod
    def generate_batch(data, targets, batch_size):
        if len(data) != len(targets):
            raise Exception("Number of input samples and number of target samples must be the same!")
        while True:
            for i in range(0, len(data) / batch_size):
                # create batches of data
                x1 = data[i * batch_size:(i + 1) * batch_size]
                y1 = targets[i * batch_size:(i + 1) * batch_size]
                yield (x1, y1)

    @staticmethod
    def get_batch_size(data, params=None):
        # define mini batch size (based on training total data amount)
        if params.batch_size_fract:
            batch_size = int((data.shape[0]) * params.batch_size_fract)
        elif params.batch_size_effective:
            batch_size = params.batch_size_effective
        elif params.batch_size_sequences:
            batch_size = int(params.train_seq_len * params.batch_size_sequences)
        else:
            batch_size = 128
            logging.info("Batch size has been set by default =" + str(batch_size))

        logging.info("Training on " + str(int(data.shape[0])) + " samples")
        logging.info("Total training records " + str(params.num_train_records))
        logging.info("Batch size: " + str(batch_size) + " samples")

        return batch_size


class CustomEarlyStopping(Callback):
    def __init__(self, patience=0, path=None, batch_size=20):
        super(Callback, self).__init__()

        self.best = -np.Inf
        self.patience = patience
        self.wait = 0
        self.path = path
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):
        predictions = self.model.predict(self.validation_data[0], batch_size=self.batch_size)
        labels = self.validation_data[1]
        accuracy = compute_scores(labels, predictions)
        score = accuracy

        if score > self.best:
            self.best = score
            self.wait = 0
            self.model.best_epoch = epoch
            logging.info("\tNew best score {0:.2f}".format(score))
            if self.path is not None:
                self.model.save(self.path)
        else:
            if self.wait >= self.patience:
                self.model.stop_training = True
            self.wait += 1


class SEDEarlyStopping(Callback):
    def __init__(self, patience=0, path=None, batch_size=20, resolution=50):
        super(Callback, self).__init__()

        self.best = np.Inf
        self.patience = patience
        self.wait = 0
        self.path = path
        self.resolution = resolution
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):
        predictions = self.model.predict(self.validation_data[0], batch_size=self.batch_size)
        labels = self.validation_data[1]
        scores = sed_eval_metric(predictions, labels, frames_in_1_sec=self.resolution)
        score = scores['er_overall_1sec']

        if score < self.best:
            self.best = score
            self.wait = 0
            self.model.best_epoch = epoch
            logging.info("\tNew best score {0:.2f}".format(score))
            if self.path is not None:
                self.model.save(self.path)
        else:
            if self.wait >= self.patience:
                self.model.stop_training = True
            self.wait += 1


class AUCEarlyStopping(Callback):
    def __init__(self, patience=0, path=None, batch_size=20):
        super(Callback, self).__init__()

        self.best = -np.Inf
        self.patience = patience
        self.wait = 0
        self.path = path
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):
        predictions = self.model.predict(self.validation_data[0], batch_size=self.batch_size)
        labels = self.validation_data[1]
        # accuracy = compute_scores(labels, predictions)
        # score = accuracy
        results_dict = compute_auc(predictions[:,1], np.argmax(labels, axis=1))
        score = results_dict['roc']
        # logging.info("Accuracy (%) = " + "{0:.2f}".format(accuracy * 100))

        if score > self.best:
            self.best = score
            self.wait = 0
            self.model.best_epoch = epoch
            logging.info("\tNew best score {0:.2f}".format(score * 100))
            if self.path is not None:
                self.model.save_weights(self.path, overwrite=True)
        else:
            if self.wait >= self.patience:
                self.model.stop_training = True
            self.wait += 1


class MonoSEDEarlyStopping(Callback):
    def __init__(self, patience=0, path=None, batch_size=20, dataframe=None):
        super(Callback, self).__init__()

        self.best = np.Inf
        self.patience = patience
        self.wait = 0
        self.path = path
        self.dataframe = dataframe
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):
        predictions = self.model.predict(self.validation_data[0], batch_size=self.batch_size)
        filename = os.path.dirname(self.path)
        err_rate = mono_sed_scores(predictions, self.dataframe, paths=filename)
        score = err_rate['error_rate']

        if score < self.best:
            self.best = score
            self.wait = 0
            self.model.best_epoch = epoch
            logging.info("\tNew best score {0:.2f}".format(score))
            if self.path is not None:
                self.model.save(self.path)
        else:
            if self.wait >= self.patience:
                self.model.stop_training = True
            self.wait += 1
