# ======================================================================
#
# -*- coding: utf-8 -*-
#
# ======================================================================

import datetime
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from .MaskOptimizer import MaskOptimizer
from .MLP_l import MLP
from .FIR_net import FIR_Network

logs_base_dir = ".\logs"
os.makedirs(logs_base_dir, exist_ok=True)


def mean_squared_error(y_true, y_pred):
    return K.mean((y_true - y_pred) * (y_true - y_pred), axis=1)


def tf_mean_ax_0(losses):
    return tf.reduce_mean(losses, axis=0)


def progressbar(it, prefix="", size=60):
    count = len(it)

    def show(j):
        x = int(size * j / count)
        print("\r%s[%s%s] %i/%i" % (prefix, "#" * x, "." * (size - x), j, count), end=" ")

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print()


class FeatureImportanceRank():
    def __init__(self, data_shape, unmasked_data_size, data_batch_size, mask_batch_size, str_id="",
                 epoch_on_which_FIR_trained=8):
        self.data_shape = data_shape
        self.data_size = np.zeros(data_shape).size
        self.unmasked_data_size = unmasked_data_size
        self.logdir = os.path.join(logs_base_dir, datetime.datetime.now().strftime("%m%d-%H%M%S"))
        self.data_batch_size = data_batch_size
        self.mask_batch_size = mask_batch_size
        self.x_batch_size = mask_batch_size * data_batch_size
        self.str_id = str_id
        self.prev_mopt_condition = False
        self.epoch_on_which_FIR_trained = epoch_on_which_FIR_trained

    def create_dense_MLP(self, arch, activation, metrics=None, error_func=mean_squared_error, es_patience=800):
        self.MLP = MLP(self.data_batch_size, self.mask_batch_size,
                       self.logdir + "MLP" + self.str_id)
        self.MLP.create_dense_model(self.data_shape, arch, activation)
        self.MLP.compile_model(error_func, tf.reduce_mean, tf_mean_ax_0, metrics)

    def create_dense_FIR(self, arch):
        self.FIR = FIR_Network(self.mask_batch_size,
                               tensorboard_logs_dir=self.logdir + "FIR_" + self.str_id)
        self.FIR.create_dense_model(self.data_shape, arch)
        self.FIR.compile_model()

    def create_mask_optimizer(self, epoch_condition=5000, maximize_error=False, record_best_masks=False,
                              perturbation_size=2, use_new_optimization=False):
        self.mopt = MaskOptimizer(self.mask_batch_size, self.data_shape, self.unmasked_data_size,
                                  epoch_condition=epoch_condition, perturbation_size=perturbation_size)
        self.FIR.sample_weights = self.mopt.get_mask_weights(self.epoch_on_which_FIR_trained)

    def train_networks_on_data(self, x_tr, y_tr, number_of_batches):

        for i in progressbar(range(number_of_batches), "Training batch: ", 50):
            mopt_condition = self.mopt.check_condiditon()

            random_indices = np.random.randint(0, len(x_tr), self.data_batch_size)
            x = x_tr[random_indices, :]
            y = y_tr[random_indices]
            FIR_train_condition = ((self.MLP.epoch_counter % self.epoch_on_which_FIR_trained) == 0)
            m = self.mopt.get_new_mask_batch(self.FIR.model, self.FIR.best_performing_mask,
                                             gen_new_opt_mask=FIR_train_condition)

            self.MLP.train_one(x, m, y)
            losses = self.MLP.get_per_mask_loss()
            losses = losses.numpy()
            self.FIR.append_data(m, losses)
            if FIR_train_condition:
                self.FIR.train_one(self.MLP.epoch_counter, mopt_condition)

            self.prev_mopt_condition = mopt_condition
            if self.MLP.useEarlyStopping is True and self.MLP.ES_stop_training is True:
                print("Activate early stopping at training epoch/batch: " + str(self.MLP.epoch_counter))
                print("Loading weights from epoch: " + str(self.MLP.ES_best_epoch))
                self.MLP.model.set_weights(self.MLP.ES_best_weights)
                break

    def get_importances(self, return_chosen_features=True):
        features_opt_used = np.squeeze(
            np.argwhere(self.mopt.get_opt_mask(self.unmasked_data_size, self.FIR.model, 12) == 1))
        m_best_used_features = np.zeros((1, self.data_size))
        m_best_used_features[0, features_opt_used] = 1
        grad_used_opt = -MaskOptimizer.gradient(self.FIR.model, m_best_used_features)[0][0, :]
        importances = grad_used_opt
        if not return_chosen_features:
            return importances
        else:
            optimal_mask = m_best_used_features[0]
            return importances, optimal_mask
