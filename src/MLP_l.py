# ======================================================================
#
# -*- coding: utf-8 -*-
#
# ======================================================================


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.callbacks import TensorBoard


class MLP:
    def __init__(self, x_batch_size, mask_batch_size, tensorboard_logs_dir="", add_mopt_perf_metric=True,
                 use_early_stopping=True):
        self.batch_size = mask_batch_size * x_batch_size
        self.mask_batch_size = mask_batch_size
        self.x_batch_size = x_batch_size
        self.losses_per_sample = None
        self.tr_loss_history = []
        self.te_loss_history = []
        self.tf_logs = tensorboard_logs_dir
        self.epoch_counter = 0
        self.add_mopt_perf_metric = add_mopt_perf_metric
        self.useEarlyStopping = use_early_stopping

    def create_dense_model(self, input_shape, dense_arch, last_activation="linear"):
        self.x_shape = input_shape
        self.y_shape = dense_arch[-1]
        input_data_layer = Input(shape=input_shape)
        x = Flatten()(input_data_layer)
        input_mask_layer = Input(shape=input_shape)
        mask = Flatten()(input_mask_layer)
        x = tf.keras.layers.Concatenate(axis=1)([x, mask])
        for units in dense_arch[:-1]:
            x = Dense(units, activation="sigmoid")(x)
        x = Dense(dense_arch[-1], activation=last_activation)(x)
        self.model = Model(inputs=[input_data_layer, input_mask_layer], outputs=x)
        # self.model.summary()

    def create_batch(self, x, masks, y):
        """
        x =     [[1,2],[3,4]]       -> [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]]
        masks = [[0,0],[1,0],[1,1]] -> [[0,0],[1,0],[1,1],[0,0],[1,0],[1,1]]
        y =     [1,3]               -> [1    ,1    ,1    ,3    ,3    ,3    ]
        """
        x_prim = np.repeat(x, len(masks), axis=0)
        y_prim = np.repeat(y, len(masks), axis=0)
        masks_prim = np.tile(masks, (len(x), 1))
        x_prim *= masks_prim
        return x_prim, masks_prim, y_prim

    def named_logs(self, model, logs, mode="train"):
        result = {}
        try:
            iterator = iter(logs)
        except TypeError:
            logs = [logs]
        metricNames = (mode + "_" + i for i in model.metrics_names)
        for l in zip(metricNames, logs):
            result[l[0]] = l[1]
        return result

    def compile_model(self, loss_per_sample, combine_losses, combine_mask_losses, metrics=None):
        self.mask_loss_combine_function = combine_mask_losses
        if self.add_mopt_perf_metric is True:
            if metrics is None:
                metrics = [self.get_mopt_perf_metric()]
            else:
                metrics.append(self.get_mopt_perf_metric())

        def logging_loss_function(y_true, y_pred):
            losses = loss_per_sample(y_true, y_pred)[:, 0]
            # print(losses)
            self.losses_per_sample = losses
            return combine_losses(losses)

        self.model.compile(loss=logging_loss_function, optimizer='nadam', metrics=metrics, run_eagerly=True)
        if self.tf_logs != "":
            log_path = './logs'
            self.tb_clbk = TensorBoard(self.tf_logs)
            self.tb_clbk.set_model(self.model)

    def get_per_mask_loss(self, used_target_shape=None):
        if used_target_shape is None:
            used_target_shape = (self.x_batch_size, self.mask_batch_size)
        losses = tf.reshape(self.losses_per_sample, used_target_shape)
        losses = self.mask_loss_combine_function(losses)
        return losses

    def get_per_mask_loss_with_custom_batch(self, losses, new_x_batch_size, new_mask_batch_size):
        losses = np.reshape(losses, newshape=(new_x_batch_size, new_mask_batch_size))
        losses = np.apply_along_axis(self.mask_loss_combine_function, 0, losses)
        return losses

    def train_one(self, x, masks, y):
        x_prim, masks_prim, y_prim = self.create_batch(x, masks, y)
        curr_loss = self.model.train_on_batch(x=[x_prim, masks_prim], y=y_prim)
        self.tr_loss_history.append(curr_loss)
        self.epoch_counter += 1
        if self.tf_logs != "":
            self.tb_clbk.on_epoch_end(self.epoch_counter, self.named_logs(self.model, curr_loss))
        return x_prim, masks_prim, y_prim

    def test_one(self, x, masks, y):
        x_prim, masks_prim, y_prim = self.create_batch(x, masks, y)
        feature = self.model.predict(x=[x_prim, masks_prim])
        return feature

    def get_mopt_perf_metric(self):
        def m_opt_loss(y_pred, y_true):
            if self.losses_per_sample.shape[0] % self.mask_batch_size != 0:
                return 0.0
            else:
                losses = tf.reshape(self.losses_per_sample, (-1, self.mask_batch_size))
                self.last_m_opt_perf = np.mean(losses[:, int(0.5 * self.mask_batch_size)])
                return self.last_m_opt_perf

        return m_opt_loss

    def set_early_stopping_params(self, starting_epoch, patience_batches=800, minimize=True):
        self.ES_patience = patience_batches
        self.ES_minimize = minimize
        if minimize is True:
            self.ES_best_perf = 1000000.0
        else:
            self.ES_best_perf = -1000000.0
        self.ES_best_epoch = starting_epoch
        self.ES_stop_training = False
        self.ES_start_epoch = starting_epoch
        self.ES_best_weights = None
        return
