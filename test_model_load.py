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

import bdd100k_loader as bdd100k

MODEL_PATH = "model.weights.h5"
BATCH_SIZE = 1

def main():
    keras.backend.clear_session()

    tf.keras.config.disable_traceback_filtering()

    print("Loading Model")

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

    print("Loading weights")

    model.load_weights(MODEL_PATH)

    print("Loading Data")

    train, val = bdd100k.load_data(1)


    print("Running fit to test weights")

    model.fit(x = train, steps_per_epoch = 10)

if __name__ == '__main__':
    os.system("rm -rf /tmp/tfdbg2_logdir")
    main()