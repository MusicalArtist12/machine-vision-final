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

LOG_PATH = "/tmp/tfdbg2_logdir"
MODEL_PATH = "model.keras"

class Dataset(NamedTuple):
    train: any
    val: any

def normalize_img(image, mask):
    """Normalizes images: `uint8` -> `float32`."""

    # image = [720, 1280, 3], each pixel being [R, G, B], R, G, B in {x in float32 | 0.0 <= x <= 1.0}
    # mask = [720, 1280, 1], each pixel being [P], P in {x in float32 | 0.0 <= x <= 1.0 }
    return tf.cast(image, tf.float32) / 255.0, tf.cast(mask, tf.float32) / 255.0

class TrainingDebugCallbacks(keras.callbacks.Callback):
    '''
    def on_train_end(self, logs = None):
        print(f"on_train_end({logs})")

    def on_train_begin(self, logs = None):
        print(f"on_train_begin({logs})")

    def on_train_batch_begin(self, step, logs = None):
        print(f"on_train_batch_begin({step}, {logs})")
    '''
    def __init__(self):
        self.counter = 0
        self.train_writer = tf.summary.create_file_writer(LOG_PATH)


    def on_train_batch_end(self, step, logs = None):

        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory()

        with self.train_writer.as_default(step=self.counter):
            tf.summary.scalar('cpu_usage', cpu_usage)
            tf.summary.scalar('memory_usage', memory_usage.percent)
            tf.summary.scalar('gpu_memory_peak', round(tf.config.experimental.get_memory_info('GPU:0')["peak"] / (1024 * 1024 * 1024), 2))

        self.counter = (self.counter + 1)

def load_data(batch_size) -> Dataset:
    (train, val) = tfds.load('bd100k', split = ['train', 'val'], as_supervised = True)

    train = train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    val = val.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

    train = train.batch(batch_size)
    val = val.batch(batch_size)

    train = train.prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)

    return Dataset(train, val)

def run_model(model, data, max_epochs):

    results: list[PlotResult] = []

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=LOG_PATH,update_freq = "batch")

    save_callback = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        save_freq=50,
        initial_value_threshold=None
    )

    epoch_counter = 0

    hist = model.fit(
        x = data.train,
        epochs = max_epochs,
        validation_data = data.val,
        validation_freq = 1,
        validation_batch_size = 100,
        validation_steps = 1,
        verbose = 1,
        # steps_per_epoch = 10,
        callbacks = [tensorboard_callback, TrainingDebugCallbacks(), save_callback],
        # callbacks = [tensorboard_callback]
    )
    keras.backend.clear_session()


    return (hist.history)

def output_summary(str):
    with open("Summary.txt", "a") as f:
        f.write(str)

def main():
    keras.backend.clear_session()
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


    # tf.debugging.experimental.enable_dump_debug_info(
    #     "/tmp/tfdbg2_logdir",
    #     tensor_debug_mode="FULL_HEALTH",
    #     circular_buffer_size=-1)

    tf.keras.config.disable_traceback_filtering()

    batch_size = 12

    data = load_data(batch_size = batch_size)

    try:
        model = keras.saving.load_model(MODEL_PATH, compile=True, safe_mode=True)
        print("Successfully loaded the previous model")
        raise Exception("not implemented")
    except Exception as e:
        print(e)
        print("Loading a new model")
        model = Segformer_B0(input_shape = (None, 720, 1280, 3), num_classes = 1)
        model.build((None, 720, 1280, 3))
        lr_schedule = keras.optimizers.schedules.CosineDecay(0.1, 1000)
        model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule),
            loss = keras.losses.BinaryCrossentropy(label_smoothing=0.2),
            metrics = [keras.metrics.BinaryIoU(), keras.metrics.BinaryCrossentropy()],
            run_eagerly = False,
            steps_per_execution = 1,
            jit_compile = False
        )

        with open("Summary.txt", "w") as f:
            f.write('')

        model.summary(
            expand_nested = True,
            show_trainable = True,
            line_length = 150,
            print_fn = output_summary
        )
        tf.keras.utils.plot_model(model, show_shapes = True, expand_nested = True, show_layer_names = True)

    print(data.val)

    run_model(model, data)

    cv.namedWindow("original")
    cv.namedWindow("mask")
    cv.namedWindow("true")
    for batch, true in data.val:
        output = model.call(batch)

        for val, msk, true in zip(batch, output, true):

            array0 = val.numpy()
            array1 = true.numpy()
            array2 = msk.numpy()

            cv.imshow("original", array0)
            cv.imshow("mask", array2)
            cv.imshow("true", array1)

            cv.waitKey(10000)

    # run_model(model, data)



if __name__ == '__main__':
    os.system("rm -rf /tmp/tfdbg2_logdir")
    main()