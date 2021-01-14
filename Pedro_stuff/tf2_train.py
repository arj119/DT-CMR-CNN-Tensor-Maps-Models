'''
Train U-Net to synthetise DTI maps from DWIs
'''

from tensorflow.python.keras import optimizers
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
import tensorflow as tf
from tensorflow.python.keras import backend as K
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, date
import functools
import sys
import glob
import os
import time
from IPython import get_ipython
from matplotlib import pyplot as plt
from IPython import display

# %%
# DTI parameter to train (DT)
dti_param = sys.argv[1]
dataset_code = sys.argv[2]
# dti_param = 'dt'
# dataset_code = '4'

if dti_param == 'dt':
    v_min = -1.0E-3
    v_max = 2.5E-3
    norm_factor = 500

# set up image size and training parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128

if (dataset_code == '3dir') or (dataset_code == 'no150'):
    INPUT_CHANNELS = 7
else:
    INPUT_CHANNELS = 13
OUTPUT_CHANNELS = 6
BUFFER_SIZE = 225
BATCH_SIZE = 8
EPOCHS = 4

# matplotlib defaults
font = {'family': 'Helvetica',
        'weight': 'bold',
        'size': 8}

plt.rc('font', **font)
print('====================================================')
print('DTI parameter: ' + dti_param)
print('Dataset code: ' + dataset_code)

# cnn filename with date dti parameter and dataset code
date_time_str = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
cnn_name = dti_param + '_' + dataset_code + \
    '_u_net_weights_' + date_time_str + '.hdf5'
loss_history_name = dti_param + '_' + dataset_code + \
    '_u_net_weights_' + date_time_str + '.pickle'
current_dir = sys.path[0]

# fix the random seed
np.random.seed(1)

input_dir = 'input_data_' + dataset_code
output_dir = 'output_data_' + dti_param

# list of arrays
input_list = glob.glob(os.path.join(input_dir, '*.npz'))
input_list.sort()

list_outputs = glob.glob(os.path.join(output_dir, '*.npz'))
list_outputs.sort()

# split list of inputs and outputs in train, validate and test
train_inputs, val_inputs, train_outputs, val_outputs = \
    train_test_split(input_list, list_outputs,
                     test_size=0.2, random_state=42)

val_inputs, test_inputs, val_outputs, test_outputs = \
    train_test_split(val_inputs, val_outputs,
                     test_size=0.5, random_state=42)

num_train_examples = len(train_inputs)
num_val_examples = len(val_inputs)
num_test_examples = len(test_inputs)

print("Number of training examples: {}".format(num_train_examples))
print("Number of validation examples: {}".format(num_val_examples))
print("Number of test examples: {}".format(num_test_examples))
print('====================================================')

# save the test subject list in a csv file
d = {'Input': test_inputs, 'Output': test_outputs}
df = pd.DataFrame(d)
df.to_csv('test_subject_list_' + dti_param + '.csv', index=False)


# %%
# Load numpy arrays


def normalise_matrices(array):
    '''normalise DWI images to the maximum value'''
    # ref_array = array[:, :, 0]
    # array = np.true_divide(array, ref_array.max())
    array = np.true_divide(array, np.amax(array))
    return array


def load_npz_files(input_list, output_list, dti_param):
    ''' load input and output arrays from lists of npz files '''
    # loop through the list and stack arrays
    for idx, val in enumerate(input_list):
        if idx == 0:
            input_arrays = np.load(val)
            input_arrays = input_arrays['arr_0']
            input_arrays = np.float32(input_arrays)
            input_arrays = normalise_matrices(input_arrays)
            input_arrays = np.expand_dims(input_arrays, axis=0)
        else:
            temp = np.load(val)
            temp = temp['arr_0']
            temp = np.float32(temp)
            temp = normalise_matrices(temp)
            temp = np.expand_dims(temp, axis=0)
            input_arrays = np.concatenate((input_arrays, temp), axis=0)

    for idx, val in enumerate(output_list):
        if idx == 0:
            output_arrays = np.load(val)
            output_arrays = output_arrays['arr_0']
            output_arrays = np.float32(output_arrays)
            output_arrays = norm_factor * output_arrays
            output_arrays = np.expand_dims(output_arrays, axis=0)
        else:
            temp = np.load(val)
            temp = temp['arr_0']
            temp = np.float32(temp)
            temp = norm_factor * temp
            temp = np.expand_dims(temp, axis=0)
            output_arrays = np.concatenate((output_arrays, temp), axis=0)

    if OUTPUT_CHANNELS == 1:
        output_arrays = np.expand_dims(output_arrays, axis=-1)

    # check there is no nan or inf
    input_arrays = np.nan_to_num(input_arrays)
    output_arrays = np.nan_to_num(output_arrays)
    assert np.isnan(np.sum(input_arrays)) == False, 'input_array has nans'
    assert np.isnan(np.sum(output_arrays)) == False, 'output_arrays has nans'
    assert np.isinf(np.sum(input_arrays)) == False, 'input_array has infs'
    assert np.isinf(np.sum(output_arrays)) == False, 'output_arrays has infs'

    assert input_arrays.shape[0] == output_arrays.shape[0]

    return input_arrays, output_arrays

# %%
# Data augmentation functions


def random_crop(img_in, img_out):
    #  stack images before rotation so that the rotation is the same
    stacked_image = tf.concat([img_in, img_out], axis=2)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS + OUTPUT_CHANNELS])

    r_img_in = cropped_image[:, :, 0:INPUT_CHANNELS]
    r_img_out = cropped_image[:, :, INPUT_CHANNELS:]
    # if OUTPUT_CHANNELS == 1:
    #     r_img_out = tf.expand_dims(r_img_out, axis=2)

    return r_img_in, r_img_out


def augment(img_in, img_out):
    # Add 20 pixels padding
    add_n = 20
    img_in = tf.image.resize_with_crop_or_pad(
        img_in, IMG_HEIGHT+add_n, IMG_WIDTH+add_n)
    img_out = tf.image.resize_with_crop_or_pad(
        img_out, IMG_HEIGHT+add_n, IMG_WIDTH+add_n)
    img_in, img_out = random_crop(img_in, img_out)

    return img_in, img_out

# %%
# Get tensorflow datasets (training and validation)


def get_baseline_dataset(input_list, output_list, dti_param):

    # read arrays
    input_arrays, output_arrays = load_npz_files(
        input_list, output_list, dti_param)

    # plt.hist(np.ndarray.flatten(
    #     output_arrays[output_arrays != 0]), bins=40)
    # plt.show()

    dataset = tf.data.Dataset.from_tensor_slices((input_arrays, output_arrays))

    return dataset


# train dataset
train_ds = get_baseline_dataset(train_inputs, train_outputs, dti_param)
train_ds = train_ds.map(augment,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)


# validation dataset
val_ds = get_baseline_dataset(val_inputs, val_outputs, dti_param)
val_ds = val_ds.repeat().batch(BATCH_SIZE)

# plot pixel values and maps to make sure everything is OK
train_in_arrays, train_out_arrays = load_npz_files(
    train_inputs, train_outputs, dti_param)
val_in_arrays, val_out_arrays = load_npz_files(
    val_inputs, val_outputs, dti_param)

f = plt.figure(figsize=(20, 8))
plt.subplot(1, 3, 1)
plt.hist(np.ndarray.flatten(
    train_out_arrays[train_out_arrays != 0]), bins=50, label='train')
plt.hist(np.ndarray.flatten(
    val_out_arrays[val_out_arrays != 0]), bins=50, label='val')
plt.legend()
plt.subplot(1, 3, 2)
im = plt.imshow(train_out_arrays[0, :, :, 0], vmin=-1, vmax=1)
plt.axis('off')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.subplot(1, 3, 3)
im = plt.imshow(val_out_arrays[0, :, :, 0], vmin=-1, vmax=1)
plt.axis('off')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.savefig(current_dir + '/pixel_values_histogram.png', dpi=f.dpi, bbox_inches='tight',
            pad_inches=0, transparent=False)
plt.close()


def conv_block(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    # encoder = layers.Activation('relu')(encoder)
    encoder = layers.LeakyReLU()(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    # encoder = layers.Activation('relu')(encoder)
    encoder = layers.LeakyReLU()(encoder)
    return encoder


def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(
        num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    # decoder = layers.Activation('relu')(decoder)
    decoder = layers.LeakyReLU()(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    # decoder = layers.Activation('relu')(decoder)
    decoder = layers.LeakyReLU()(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    # decoder = layers.Activation('relu')(decoder)
    decoder = layers.LeakyReLU()(decoder)
    return decoder


inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS))
# 256
encoder0_pool, encoder0 = encoder_block(inputs, 32)
# 128
encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
# 64
encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
# 32
encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
# 16
encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)
# 8
center = conv_block(encoder4_pool, 1024)
# center
decoder4 = decoder_block(center, encoder4, 512)
# 16
decoder3 = decoder_block(decoder4, encoder3, 256)
# 32
decoder2 = decoder_block(decoder3, encoder2, 128)
# 64
decoder1 = decoder_block(decoder2, encoder1, 64)
# 128
decoder0 = decoder_block(decoder1, encoder0, 32)
# 256

outputs = layers.Conv2D(OUTPUT_CHANNELS, (1, 1), activation='linear')(decoder0)

# Define your model
model = models.Model(inputs=[inputs], outputs=[outputs])

tf.keras.utils.plot_model(model, show_shapes=True, dpi=128)

print('Training model ...')

folder_name = dti_param + '_' + dataset_code + '_' + date_time_str
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
save_model_path = os.path.join(current_dir, folder_name, cnn_name)
save_loss_path = os.path.join(current_dir, folder_name, loss_history_name)

adam = tf.keras.optimizers.Adam(
    lr=1e-3, beta_1=0.9, beta_2=0.999, amsgrad=False)

# loss function
# mean square error


def custom_loss(y_true, y_pred):
    loss_mat = tf.abs(y_true - y_pred)
    loss_mat = loss_mat[y_true != 0]
    loss = tf.reduce_mean(loss_mat)
    return loss


model.compile(loss=custom_loss,
              optimizer=adam,
              metrics=None)

model.summary()


cp = tf.keras.callbacks.ModelCheckpoint(
    filepath=save_model_path, monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(train_ds,
                    steps_per_epoch=int(
                        np.ceil(num_train_examples / float(BATCH_SIZE))),
                    epochs=EPOCHS,
                    validation_data=val_ds,
                    validation_steps=int(
                        np.ceil(num_val_examples / float(BATCH_SIZE))),
                    callbacks=[cp])

pickle_out = open(save_loss_path, "wb")
pickle.dump(history.history, pickle_out)
pickle_out.close()
