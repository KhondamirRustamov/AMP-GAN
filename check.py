import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from model import make_generator_model, make_discriminator_model
import time
import os

tf.config.run_functions_eagerly(True)

matrix = np.load('data/sequence_32.npy')
# matrix_bulk = np.load('sequence_u.npy')
print(matrix.shape)
# print(matrix_bulk.shape)

BUFFER_SIZE = 60000
BATCH_SIZE = 1024


# train_bulk = tf.data.Dataset.from_tensor_slices(matrix_bulk).shuffle(BUFFER_SIZE_BULK).batch(BATCH_SIZE_BULK)
train_dataset = tf.data.Dataset.from_tensor_slices(matrix).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator = make_generator_model()
discriminator = make_discriminator_model()
backup = make_discriminator_model()


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    w_fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    w_r_loss = tf.compat.v1.losses.compute_weighted_loss(real_output, tf.ones_like(real_output))
    w_f_loss = tf.compat.v1.losses.compute_weighted_loss(fake_output, tf.ones_like(fake_output))
    w_loss = w_f_loss - w_r_loss
    return total_loss


def generator_loss(fake_output):
    w_fake_loss = tf.compat.v1.losses.compute_weighted_loss(-fake_output, tf.ones_like(fake_output))
    fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return fake_loss


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 50
EPOCHS_BULK = 200
noise_dim = 100
num_examples_to_generate = 5000

checkpoint_dir = 'results'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


@tf.function
def d_loop(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=False)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


@tf.function
def unrolled_d_loop(images, noise):
    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=False)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


@tf.function
def g_loop(noise):
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=False)
        gen_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))


def train_unrolled(dataset, epochs, unrolled_steps, saved_file):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:

            d_loop(image_batch)
            noise = tf.random.normal([BATCH_SIZE, noise_dim])
            if unrolled_steps > 0:
                backup.set_weights(discriminator.get_weights())
                for i in range(unrolled_steps):
                    unrolled_d_loop(image_batch, noise)
                g_loop(noise)
                discriminator.set_weights(backup.get_weights())
            else:
                g_loop(noise)
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
    generate_and_save_images(generator,
                             seed,
                             saved_file)


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs, saved_file):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as you go

        # Save the model every 15 epochs
        #if (epoch + 1) % 15 == 0:
            #checkpoint.save(file_prefix=checkpoint_prefix)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    generate_and_save_images(generator,
                             seed,
                             saved_file)


def generate_and_save_images(model, test_input, saved_file):
    predictions = model(test_input, training=False)
    np.save(saved_file, predictions)


# train(train_bulk, EPOCHS_BULK, 'result/bulk_predictions_ugan.npy')
# train(train_dataset, EPOCHS, 'result/dcgan.npy')
# train_unrolled(train_bulk, EPOCHS_BULK, 5, 'u_gan/uniprot_predictions_ugan.npy')
train_unrolled(train_dataset, EPOCHS, 5, 'u_gan/u-gan50.npy')

