"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
Based on the following code available on Github: `https://github.com/XifengGuo/CapsNet-Keras`
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`,
"""
import os
import numpy as np
from keras import layers, models, optimizers
from keras.regularizers import *
from keras.callbacks import Callback, CSVLogger, TensorBoard, LearningRateScheduler, ModelCheckpoint
from capsule.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
import logging
from evaluation.scores import compute_scores, sed_eval_metric, mono_sed_scores
from .loss_function import *

K.set_image_data_format('channels_last')


# from capsule.utils import combine_images, plot_log
# from PIL import Image


class CapsuleNeuralNetwork:
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
        self.save_preds = True
        self.dataframe = dataframe

    def CapsNet(self, input_shape, n_class):
        """
        A Capsule Network on MNIST.
        :param input_shape: data shape, 3d, [width, height, channels]
        :param n_class: number of classes
        :param params: args object
        :return: Two Keras Models, the first one used for training, and the second one for evaluation.
                `eval_model` can also be used for training.
        """
        x = layers.Input(shape=input_shape)
        conv1 = x

        # First Layers : N Conventional Conv2D layers
        for i in range(len(self.params.kernel_number)):
            conv1 = layers.Conv2D(filters=self.params.kernel_number[i],
                                  kernel_size=self.params.kernel_shape[i],
                                  strides=self.params.cnn_strides[i],
                                  padding=self.params.border_mode,
                                  activation=self.params.cnn_activation,
                                  kernel_regularizer=eval(self.params.cnn_w_reg),
                                  use_bias=self.params.bias)(conv1)

            if self.params.dropout_capsule:
                conv1 = layers.Dropout(self.params.drop_rate_capsule)(conv1)
            if self.params.batch_norm:
                conv1 = layers.BatchNormalization(axis=-1)(conv1)

            conv1 = layers.MaxPooling2D(pool_size=(tuple(self.params.pool_shapes[i])),
                                        strides=self.params.pool_strides[i],
                                        padding=self.params.pooling_border_mode)(conv1)

        # Number of time slices
        n_steps = int(x.shape[1])

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        primarycaps = PrimaryCap(conv1,
                                 dim_capsule=self.params.dim_primary_capsule,
                                 n_channels=self.params.primary_n_channels,
                                 kernel_size=self.params.primary_kernel_size,
                                 strides=self.params.primary_kernel_strides,
                                 cnn_k_regularizer=self.params.cnn_w_reg,
                                 dropout=self.params.dropout_capsule,
                                 drop_rate=self.params.drop_rate_capsule,
                                 padding=self.params.border_mode)

        primarycaps = layers.Reshape((n_steps, -1, self.params.dim_primary_capsule))(primarycaps)

        if self.params.batch_norm:
            primarycaps = layers.BatchNormalization(axis=-1)(primarycaps)

        # Layer 3: Capsule layer. Routing algorithm works here.
        digitcaps = layers.TimeDistributed(CapsuleLayer(num_capsule=n_class,
                                                        dim_capsule=self.params.dim_capsule,
                                                        routings=self.params.routings,
                                                        momentum=self.params.momentum,
                                                        name='digitcaps'))(primarycaps)

        # Layer 4: This is an auxiliary layer to replace each capsule with its length.
        # Just to match the true label's shape.
        # If using tensorflow, this will not be necessary. :)
        out_caps = Length(name='capsnet')(digitcaps)
        # Only onset output
        # ###############################################
        # out_caps = layers.Dense(2, activation='softmax',
        #                         kernel_regularizer=eval(self.params.cnn_w_reg))(out_caps)
        # if self.params.dropout_capsule:
        #     out_caps = layers.Dropout(self.params.drop_rate_capsule)(out_caps)
        # ###############################################
        # Models for training and evaluation (prediction)
        train_model = models.Model(x, out_caps)
        eval_model = train_model

        train_model.summary()

        return train_model, eval_model

    def train(self, model, train_data, validation_data=None, list_of_callback=list()):
        """
        Training a CapsuleNet
        :param model: the CapsuleNet model
        :param train_data: a tuple containing training data, like `(x_train, y_train)`
        :param validation_data: a tuple containing testing data, like `(x_test, y_test)`
        :param list_of_callback: list of callbacks to use during the training
        :return: The trained model
        """
        # callbacks
        callbacks = list()
        save_dir = os.path.join(self.paths['experiment_folder'], self.name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if 'TensorBoard' in list_of_callback:
            tb = TensorBoard(log_dir=save_dir + '/tensorboard-logs',
                             batch_size=self.params.batch_size,
                             histogram_freq=int(self.params.debug),
                             write_graph=True,
                             write_grads=True, write_images=True)
            callbacks.append(tb)

        if 'CSVLogger' in list_of_callback:
            csv_logger = CSVLogger(save_dir + '/log.csv')
            callbacks.append(csv_logger)

        if 'CustomEarlyStopping' in list_of_callback:
            custom_early_stopping = CustomEarlyStopping(patience=self.params.patience,
                                                        path=os.path.join(save_dir, 'trained_best_model.h5'))
            callbacks.append(custom_early_stopping)

        if 'SEDEarlyStopping' in self.params.callbacks:
            custom_early_stopping = SEDEarlyStopping(patience=self.params.patience,
                                                     path=os.path.join(save_dir, 'trained_best_model.h5'),
                                                     resolution=self.params.resolution)
            callbacks.append(custom_early_stopping)

        if 'MonoSEDEarlyStopping' in self.params.callbacks:
            custom_early_stopping = MonoSEDEarlyStopping(patience=self.params.patience,
                                                         path=os.path.join(save_dir, 'trained_best_model.h5'),
                                                         dataframe=self.dataframe)
            callbacks.append(custom_early_stopping)

        if 'LearningRateDecay' in list_of_callback:
            lr_decay = LearningRateScheduler(
                schedule=lambda epoch: self.params.learning_rate * (self.params.lr_decay ** epoch))
            callbacks.append(lr_decay)

        if 'ModelCheckPoint' in list_of_callback:
            checkpoint = ModelCheckpoint(save_dir + '/trained_model_best.h5', monitor='val_capsnet_acc',
                                         save_best_only=True, save_weights_only=True, verbose=1)
            callbacks.append(checkpoint)

        # set optimizer
        if self.params.optimizer == "adadelta":
            opti = optimizers.Adadelta(lr=self.params.learning_rate, rho=0.95, epsilon=1e-06)
        if self.params.optimizer == "adam":
            opti = optimizers.Adam(lr=self.params.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None,
                                   decay=0.0, amsgrad=False)
        if self.params.optimizer == "RMSprop":
            opti = optimizers.RMSprop(decay=self.params.learning_rate_decacy)

        # compile the model
        # unpacking the data
        (x_train, y_train) = train_data

        model.compile(optimizer=opti,
                      loss=[margin_loss],
                      metrics={'capsnet': 'accuracy'})

        if validation_data:
            (x_test, y_test) = validation_data
            model.fit(x_train, y_train, batch_size=self.params.batch_size,
                      epochs=self.params.epochs, validation_data=[x_test, y_test],
                      callbacks=callbacks, verbose=2)
        else:
            model.fit(x_train, y_train, batch_size=self.params.batch_size,
                      epochs=self.params.epochs, callbacks=callbacks, verbose=2)

        model.save_weights(os.path.join(save_dir, 'trained_model.h5'))
        logging.info('Trained model saved to \'%s/trained_model.h5\'' % save_dir)

        # plot_log(filepath=self.params.save_dir)

        return model

    def test(self, model, data):
        x_test, y_test = data

        logging.info("Computing network outputs")
        if self.batch_size == 0:
            # batch_size = NeuralNetwork.get_batch_size(x_test, self.params)
            batch_size = self.params.batch_size
            self.batch_size = batch_size

        y_pred = model.predict(x_test, batch_size=1)

        if self.save_preds:
            predictions_folder = os.path.join(self.paths['experiment_folder'],
                                              'network_outputs')  # network output folder
            os.makedirs(predictions_folder, exist_ok=True)
            np.save(os.path.join(predictions_folder, 'preds_' + self.name + '.npy'), y_pred)
        # img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
        # image = img * 255
        # Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
        # logging.info('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
        # plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
        # plt.show()
        return y_pred

    # def manipulate_latent(model, data, args):
    #     print('-'*30 + 'Begin: manipulate' + '-'*30)
    #     x_test, y_test = data
    #     index = np.argmax(y_test, 1) == args.digit
    #     number = np.random.randint(low=0, high=sum(index) - 1)
    #     x, y = x_test[index][number], y_test[index][number]
    #     x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    #     noise = np.zeros([1, 10, 16])
    #     x_recons = []
    #     for dim in range(16):
    #         for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
    #             tmp = np.copy(noise)
    #             tmp[:,:,dim] = r
    #             x_recon = model.predict([x, y, tmp])
    #             x_recons.append(x_recon)
    #
    #     x_recons = np.concatenate(x_recons)
    #
    #     img = combine_images(x_recons, height=16)
    #     image = img*255
    #     Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    #     print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    #     print('-' * 30 + 'End: manipulate' + '-' * 30)


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
        predictions = self.model.predict(self.validation_data[0], batch_size=1)
        labels = self.validation_data[1]
        scores = sed_eval_metric(predictions, labels, frames_in_1_sec=self.resolution)
        score = scores['er_overall_1sec']

        if score < self.best:
            self.best = score
            self.wait = 0
            self.model.best_epoch = epoch
            logging.info("\tNew best score {0:.2f}".format(score))
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
