from model.bicgan import make_generator_model, make_discriminator_model, make_estimator_model
import numpy as np
import tensorflow as tf
import time
import os

matrix = np.load('sequence1.npy')
matrix_bulk = np.load('sequence_u.npy')

BUFFER_SIZE = 60000
BUFFER_SIZE_BULK = 500000
BATCH_SIZE = 256
BATCH_SIZE_BULK = 512

train_bulk = tf.data.Dataset.from_tensor_slices(matrix_bulk).shuffle(BUFFER_SIZE_BULK).batch(BATCH_SIZE_BULK)
train_dataset = tf.data.Dataset.from_tensor_slices(matrix).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


estimator = make_estimator_model()
prediction = estimator(train_dataset[0])
print('predicted: ', prediction)
