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

MODEL_PATH = "model.keras"
BATCH_SIZE = 1

def main():
    keras.backend.clear_session()

    tf.keras.config.disable_traceback_filtering()

    print("Loading Model")

    model = keras.saving.load_model(MODEL_PATH)

    print("Loading Data")

    train, val = bdd100k.load_data(1)

    print("Running fit to load weights")

    model.fit(x = train, steps_per_epoch = 10)

if __name__ == '__main__':
    os.system("rm -rf /tmp/tfdbg2_logdir")
    main()