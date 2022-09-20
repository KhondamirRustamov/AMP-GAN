import numpy as np
import pandas as pd
import tensorflow as tf
from model import make_generator_model
import time
import os
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor

tf.config.run_functions_eagerly(True)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator = make_generator_model()


def generator_loss(fake_output):
    fake_output = tf.convert_to_tensor(fake_output)
    fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return fake_loss


generator_optimizer = tf.keras.optimizers.Adam(1e-4)

BATCH_SIZE = 1024
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 500
seed = tf.random.normal([num_examples_to_generate, noise_dim])


def translator(input):
    input1 = input.numpy()
    translate = []
    for z in input1:
        rounded = [[int(round(float(i), 0)) for i in x] for x in z]
        list_a = [''.join([str(int(x)) for x in i]) for i in rounded]
        a_l = [int(i, 2) for i in list_a]
        translate.append(a_l)
    a = ' ARNDCEQGHILKMFPSTWYV            '
    new_translation = [''.join([a[i] for i in x]) for x in translate]
    seq_list = []
    for i, c in zip(new_translation, range(len(new_translation))):
        z = []
        while i.startswith(' '):
            i = i[1:]
        for x in i:
            if x == 'B':
                z.append('N')
            elif x == 'Z':
                z.append('Q')
            elif x == 'X':
                z.append('A')
            elif x == 'J':
                z.append('L')
            elif x != ' ':
                z.append(x)
            else:
                continue
        seq_list.append(''.join(z))
    return seq_list


def mod_cheking(input):
    seq_list = translator(input)
    desc = GlobalDescriptor(seq_list)
    desc.calculate_charge(ph=7.4, amide=True)
    charge = desc.descriptor.reshape(-1).tolist()
    charge = [x/7 for x in charge]
    AMP = PeptideDescriptor(seq_list)
    AMP.calculate_global()  # H global hidrophobicity
    H = AMP.descriptor.reshape(-1).tolist()
    AMP.calculate_moment()  # uH moment hidrophobicity
    uH = AMP.descriptor.reshape(-1).tolist()
    result = np.array([charge, H, uH])
    return result


@tf.function
def g_loop():
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = mod_cheking(generated_images)
        gen_loss = generator_loss(fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))


def train(epochs):
    for epoch in range(epochs):
        start = time.time()

        g_loop()

        # Produce images for the GIF as you go

        # Save the model every 15 epochs
        print ('Time for epoch {} is {} sec;      generator loss:'.format(epoch + 1, time.time()-start))
    generate_and_save_images(generator,
                             seed)


def generate_and_save_images(model, test_input):
    predictions = model(test_input, training=False)
    translation = translator(predictions)
    predictions = mod_cheking(predictions)
    test = {'seq': translation,
            'C': predictions[0],
            'H': predictions[1],
            'uH': predictions[2]}
    test = pd.DataFrame(test)
    test.to_csv('generator.csv')


train(1)
