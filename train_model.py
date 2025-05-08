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


    def on_epoch_end(self, epoch, logs=None):
        results = []
        idx = 0
        for element in self.val_data:
            if idx > 25:
                break
            else:
                idx += 1

            res = self.model(element[0]).numpy()[0] * 255
            truth = element[1][0] * 255
            image = element[0][0]

            green = np.full_like(image,(0,255,0))
            blue = np.full_like(image, (255, 0, 0))

            img_green = cv.addWeighted(image, 0.5, green, 0.5, 0)
            img_blue = cv.addWeighted(image, 0.5, blue, 0.5, 0)

            result = np.where(res == 255, img_green, image)
            result = np.where(truth == 255, img_blue, result)

            results.append(result)

        file_writer = tf.summary.create_file_writer(log_path + '/train_data')

        with file_writer.as_default():
            tf.summary.image("25 training data examples", results, max_outputs=25, step=epoch)


class ModelTrainer():
    def __init__(self, log_path, batch_size, gradient_accumulation_steps, num_epochs, learning_rate, save_freq, save_model_path, load_model_path = ""):
        self.load_model_path = load_model_path
        self.save_model_path = save_model_path
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_path = log_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.save_freq = save_freq

    def main(self):
        keras.backend.clear_session()

        tf.keras.config.disable_traceback_filtering()

        print("Loading")

        model = Segformer_B0(input_shape = (None, 720, 1280, 3), num_classes = 1)

        print("Building")

        model.build((None, 720, 1280, 3))

        print("Compiling")

        model.compile(
            optimizer = keras.optimizers.AdamW(
                learning_rate = self.learning_rate,
                gradient_accumulation_steps = self.gradient_accumulation_steps
            ),
            loss = keras.losses.Dice(),
            metrics = [
                keras.metrics.BinaryIoU(target_class_ids=[1], name="TrueIoU"),
                keras.metrics.BinaryIoU(target_class_ids=[0], name="FalseIoU"),
                keras.metrics.BinaryIoU(target_class_ids=[0, 1], name="MeanIoU")
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

        save_callback = tf.keras.callbacks.ModelCheckpoint(
            self.save_model_path,
            save_freq=self.save_freq,
            initial_value_threshold=None,
            save_weights_only=True
        )

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir = self.log_path, update_freq = "batch")

        visualization_callback = VisualizeModelPredictions( self.log_path)

        hist = model.fit(
            x = train,
            epochs = self.num_epochs,
            validation_data = val,
            validation_freq = 1,
            validation_batch_size = 100,
            validation_steps = 1,
            verbose = 1,
            callbacks = [save_callback, tensorboard_callback, visualization_callback],
        )
        model.save_weights(self.save_model_path)
