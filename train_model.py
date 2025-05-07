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

import bdd100k

MODEL_PATH = "model.weights.h5"
BATCH_SIZE = 1

def main():
    keras.backend.clear_session()

    tf.keras.config.disable_traceback_filtering()

    print("Loading")

    model = Segformer_B0(input_shape = (None, 720, 1280, 3), num_classes = 1)

    print("Building")

    model.build((None, 720, 1280, 3))

    print("Compiling")

    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = keras.losses.BinaryCrossentropy(label_smoothing=0.2),
        metrics = [keras.metrics.BinaryIoU(), keras.metrics.BinaryCrossentropy()],
        run_eagerly = False,
        steps_per_execution = 1,
        jit_compile = False
    )

    print("Loading Data")

    train, val = bdd100k.load_data(1)

    print("Training")

    save_callback = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        save_freq=50,
        initial_value_threshold=None
    )

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=LOG_PATH,update_freq = "batch")

    epoch_counter = 0

    hist = model.fit(
        x = train,
        epochs = max_epochs,
        validation_data = val,
        validation_freq = 1,
        validation_batch_size = 100,
        validation_steps = 1,
        verbose = 1,
        # steps_per_epoch = 10,
        callbacks = [save_callback, tensorboard_callback],
        # callbacks = [tensorboard_callback]
    )
    model.save_weights(MODEL_PATH)

if __name__ == '__main__':
    os.system("rm -rf /tmp/tfdbg2_logdir")
    main()