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

MODEL_PATH = "model.weights.h5"

BATCH_SIZE = 1
NUM_EPOCHS = 50

def main():
    keras.backend.clear_session()

    tf.keras.config.disable_traceback_filtering()

    print("Loading")

    model = Segformer_B0(input_shape = (None, 720, 1280, 3), num_classes = 1)

    print("Building")

    model.build((None, 720, 1280, 3))

    print("Compiling")

    model.compile(
        optimizer = keras.optimizers.AdamW(gradient_accumulation_steps=16),
        loss = keras.losses.BinaryCrossentropy(label_smoothing=0.2),
        metrics = [keras.metrics.BinaryIoU(), keras.metrics.BinaryCrossentropy()],
        run_eagerly = False,
        steps_per_execution = 1,
        jit_compile = False
    )

    # model.load_weights(MODEL_PATH)

    train, val = bdd100k.load_data(1)

    # test = tfds.as_numpy(val)

    results = self.model.predict(self.val_data, steps = 10)

    for element in results:
        print(element.shape)
        input()
        # res = model(element[0])

        '''
        while True:
            cv.imshow("window", element[0][0])
            cv.imshow("window2", element[1][0] * 255.0)

            cv.imshow("result", res.numpy()[0] * 255.0 )
            if cv.waitKey(1) == ord('q'):
                break
        '''



if __name__ == '__main__':
    main()