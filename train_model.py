import tensorflow as tf
from typing import NamedTuple
import tensorflow_datasets as tfds
from model import Segformer_B0
import cv2 as cv
import numpy as np
import os
import psutil
import keras
import gc
import cv2 as cv
import sys

import bdd100k_loader as bdd100k

'''
MODEL_PATH = ""
LOG_PATH = ""

BATCH_SIZE = 1
NUM_EPOCHS = 50
LEARNING_RATE = 0.00006
SAVE_FREQ = 1000
'''

class VisualizeModelPredictions(keras.callbacks.Callback):
    def __init__(self, log_path):
        (_, val_data) = bdd100k.load_data(1)

        self.val_data = tfds.as_numpy(val_data)
        self.log_path = log_path

    def on_epoch_begin(self, epoch, logs=None):
        results = []
        idx = 0
        for element in self.val_data:
            if idx > 25:
                break
            else:
                idx += 1

            res = self.model(element[0])
            mask = res.numpy()[0] * 255
            image = element[0][0]

            green = np.full_like(image,(0,255,0))

            img_green = cv.addWeighted(image, 0.5, green, 0.5, 0)

            result = np.where(mask == 255, img_green, image)

            results.append(mask)

        file_writer = tf.summary.create_file_writer(self.log_path + '/val')

        with file_writer.as_default():
            tf.summary.image("25 training data examples", results, max_outputs=25, step=epoch)


class ModelTrainer():
    def __init__(self, log_path, batch_size, gradient_accumulation_steps, num_epochs, learning_rate, save_freq, save_model_path, backup_path, backup_freq, load_model_path = ""):
        self.load_model_path = load_model_path
        self.save_model_path = save_model_path
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_path = log_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.save_freq = save_freq
        self.backup_path = backup_path
        self.backup_freq = backup_freq

    def main(self):
        keras.backend.clear_session()

        tf.keras.config.disable_traceback_filtering()

        print("Loading")

        model = Segformer_B0(input_shape = (None, bdd100k.IMAGE_H, bdd100k.IMAGE_W, 3), num_classes = bdd100k.NUM_CLASSES)

        print("Building")

        model.build((None, bdd100k.IMAGE_H, bdd100k.IMAGE_W, 3))

        print("Compiling")

        model.compile(
            optimizer = keras.optimizers.AdamW(
                learning_rate = self.learning_rate,
                gradient_accumulation_steps = self.gradient_accumulation_steps if self.gradient_accumulation_steps > 1 else None
            ),
            loss = keras.losses.Dice(axis=(2)),
            metrics = [
                keras.metrics.BinaryIoU(target_class_ids=[1], name="TrueIoU"),
                keras.metrics.BinaryIoU(target_class_ids=[0], name="FalseIoU"),
                keras.metrics.BinaryIoU(target_class_ids=[0, 1], name="MeanIoU"),
                keras.losses.Dice(axis=(0)),
                keras.losses.Dice(axis=(1)),
                keras.losses.Dice(axis=(2))
            ],
            run_eagerly = False,
            steps_per_execution = self.gradient_accumulation_steps,
            jit_compile = False
        )

        if self.load_model_path != "":
            model.load_weights(self.load_model_path)

        print("Loading Data")

        train, val = bdd100k.load_data(self.batch_size)

        print("Training")


        tensorboard_callback = keras.callbacks.TensorBoard(log_dir = self.log_path, update_freq = "batch")

        visualization_callback = VisualizeModelPredictions( self.log_path)

        backup = keras.callbacks.BackupAndRestore(backup_dir = self.backup_path, save_freq = self.backup_freq)
        hist = model.fit(
            x = train,
            epochs = self.num_epochs,
            validation_data = val,
            validation_freq = 1,
            validation_batch_size = 100,
            validation_steps = 1,
            verbose = 1,
            callbacks = [tensorboard_callback, visualization_callback, backup],
            steps_per_epoch = bdd100k.NUM_TRAINING // self.batch_size
        )
        model.save_weights(self.save_model_path)
