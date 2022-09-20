import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(5 * 5 * 20, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((5, 5, 20)))
    assert model.output_shape == (None, 5, 5, 20)  # Note: None is the batch size
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2DTranspose(10, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 5, 5, 10)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2DTranspose(1, (3, 3), strides=(2, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 10, 5, 1)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2DTranspose(1, (3, 3), strides=(2, 1), padding='same', use_bias=False, activation='sigmoid'))
    assert model.output_shape == (None, 20, 5, 1)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                                     input_shape=[20, 5, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (2, 2), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def make_estimator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                            input_shape=[20, 5, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (2, 2), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(25))
    model.add(layers.Dense(250))
    model.add(layers.Reshape((1, 25, 10)))
    assert model.output_shape == (None, 25, 1, 10)  # Note: None is the batch size

    model.add(layers.Conv1DTranspose(10, 2, strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, 25, 1, 10)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1DTranspose(1, 2, strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, 100, 1, 1)

    return model



