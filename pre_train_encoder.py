import tensorflow as tf
from typing import NamedTuple
import tensorflow_datasets as tfds
from model import *
import cv2 as cv
import numpy as np
import os
import psutil
import keras
import gc
import cv2 as cv
import sys

import bdd100k_loader as bdd100k


class EncoderPreTrainer():
    def __init__(self, log_path, batch_size, gradient_accumulation_steps, num_epochs, learning_rate, save_freq, save_model_path, backup_path, backup_freq, false_pos_pen, false_neg_pen, load_model_path = ""):
        self.log_path = log_path
        self.load_model_path = load_model_path
        self.save_model_path = save_model_path
        self.gradient_accumulation_steps = gradient_accumulation_steps
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

        model = Segformer_B0_Encoder(input_shape = (None, bdd100k.IMAGE_H, bdd100k.IMAGE_W, 3), num_classes = bdd100k.NUM_CLASSES)

        print("Building")

        model.build((None, bdd100k.IMAGE_H, bdd100k.IMAGE_W, 3))

        print("Compiling")

        model.compile(
            optimizer = keras.optimizers.AdamW(
                learning_rate = self.learning_rate,
                gradient_accumulation_steps = self.gradient_accumulation_steps if self.gradient_accumulation_steps > 1 else None
            ),
            loss = "binary_crossentropy",
            # penalize false positives way more than false negatives
            metrics = ["accuracy"],
            run_eagerly = False,
            steps_per_execution = self.gradient_accumulation_steps,
        )

        if self.load_model_path != "":
            model.load_weights(self.load_model_path)

        print("Loading Data")

        train = bdd100k.load_pretrain_data(self.batch_size)

        print("Training")



        backup = keras.callbacks.BackupAndRestore(backup_dir = self.backup_path, save_freq = self.backup_freq)
        hist = model.fit(
            x = train,
            epochs = self.num_epochs,
            verbose = 1,
            callbacks = [backup]
        )
        model.save_weights(self.save_model_path)
