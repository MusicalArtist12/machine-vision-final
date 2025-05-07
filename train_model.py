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

MODEL_PATH = sys.argv[2]
LOG_PATH = sys.argv[1]
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
        optimizer = keras.optimizers.AdamW(
            gradient_accumulation_steps=16,
            learning_rate=0.00006
        ),
        loss = keras.losses.BinaryCrossentropy(from_logits=True),
        metrics = [keras.metrics.BinaryIoU(target_class_ids=[1], name="TrueIoU"), keras.metrics.BinaryIoU(target_class_ids=[0], name="FalseIoU"), , keras.metrics.BinaryIoU(target_class_ids=[0, 1], name="MeanIoU")],
        run_eagerly = False,
        steps_per_execution = 1,
        jit_compile = False
    )

    if len(sys.argv) > 3 and sys.argv[3] == "preload":
        model.load_weights(MODEL_PATH)

    print("Loading Data")


    train, val = bdd100k.load_data(1)

    print("Training")

    save_callback = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        save_freq=50,
        initial_value_threshold=None,
        save_weights_only=True
    )

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=LOG_PATH,update_freq = "batch")


    epoch_counter = 0

    hist = model.fit(
        x = train,
        epochs = NUM_EPOCHS,
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
    os.system("rm -rf /content/logs")
    main()