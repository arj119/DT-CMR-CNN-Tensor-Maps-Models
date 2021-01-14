'''
Test U-Net to synthetise DTI maps from DWIs
'''
# %%
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
import tensorflow as tf
from tensorflow.python.keras import backend as K
from sklearn.model_selection import train_test_split

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
import matplotlib as mpl
from IPython import display
import pickle


from PyQt5.QtWidgets import (
    QFileDialog, QAbstractItemView, QListView, QTreeView, QApplication, QDialog)


class getExistingDirectories(QFileDialog):
    '''PyQt5 dialog to select multiple folders'''

    def __init__(self, *args):
        super(getExistingDirectories, self).__init__(*args)
        self.setOption(self.DontUseNativeDialog, True)
        self.setFileMode(self.Directory)
        self.setOption(self.ShowDirsOnly, True)
        self.findChildren(QListView)[0].setSelectionMode(
            QAbstractItemView.ExtendedSelection)
        self.findChildren(QTreeView)[0].setSelectionMode(
            QAbstractItemView.ExtendedSelection)


# %%
# DTI parameter to train (DT)
dti_param = 'dt'
dataset_code = '4'

if dti_param == 'dt':
    v_min = -1.0E-3
    v_max = 2.5E-3
    norm_factor = 500


# set up image size and training parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
if (dataset_code == '3dir'):
    INPUT_CHANNELS = 7
else:
    INPUT_CHANNELS = 13
if dti_param == 'dt':
    OUTPUT_CHANNELS = 6

BUFFER_SIZE = 225
BATCH_SIZE = 8


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
cnn_name_new = dti_param + '_' + dataset_code + \
    '_u_net_weights_' + date_time_str + '.hdf5'
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
def get_baseline_dataset(input_list, output_list, dti_param):

    # read arrays
    input_arrays, output_arrays = load_npz_files(
        input_list, output_list, dti_param)

    dataset = tf.data.Dataset.from_tensor_slices((input_arrays, output_arrays))

    return dataset


# test dataset
test_ds = get_baseline_dataset(test_inputs, test_outputs, dti_param)
test_ds = test_ds.batch(BATCH_SIZE)


# dialog to select folders
qapp = QApplication(sys.argv)
dlg = getExistingDirectories()

if dlg.exec_() == QDialog.Accepted:
    trained_path = dlg.selectedFiles()
trained_path = str(trained_path).strip('[]')
trained_path = trained_path.replace("'", "")


cnn_file = glob.glob(os.path.join(trained_path, '*.hdf5'))
if len(cnn_file) == 0:
    cnn_file = glob.glob(os.path.join(trained_path, '*.h5'))

plot_file = glob.glob(os.path.join(trained_path, '*.pickle'))

assert len(cnn_file) == 1, 'Folder needs to have 1 h5 file'
assert len(plot_file) == 1, 'Folder needs to have 1 pickle file'

print('Model found: loading model ...')

# loss function


def custom_loss(y_true, y_pred):
    loss_mat = tf.abs(y_true - y_pred)
    loss_mat = loss_mat[y_true != 0]
    loss = tf.reduce_mean(loss_mat)
    return loss


model = models.load_model(cnn_file[0], custom_objects={
                          'custom_loss': custom_loss})

pickle_in = open(plot_file[0], "rb")
train_history = pickle.load(pickle_in)

train_loss = train_history['loss']
val_loss = train_history['val_loss']

epochs_range = range(len(val_loss))

m = min(val_loss)
pos = [i for i, j in enumerate(val_loss) if j == m]

f = plt.figure(figsize=(8, 8))
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss')
axes = plt.gca()
# axes.set_ylim([0, 0.1])
axes.set_yscale('log')
plt.title('Final val loss: ' + str('%.4e' % m) +
          ' at epoch: ' + str('%.0i' % pos[-1]))
plt.savefig(trained_path + '/train_and_validation_loss.png', dpi=f.dpi, bbox_inches='tight',
            pad_inches=0, transparent=False)
plt.close()


# run the model predictions for all test images
test_input_arrays, test_output_arrays = load_npz_files(
    test_inputs, test_outputs, dti_param)

predicted_maps = model.predict(test_input_arrays)

# zero the background pixels
ref = test_input_arrays[:, :, :, 0]
predicted_maps[ref == 0] = 0

# plt.figure(1)
# plt.subplot(2, 2, 1)
# ax1 = plt.hist(np.ndarray.flatten(
#     test_output_arrays[test_output_arrays != 0]), bins=40)
# plt.title("Ground truth")
# plt.subplot(2, 2, 2)
# im = plt.imshow(test_output_arrays[0, :, :, 0])
# plt.axis('off')
# plt.colorbar(im, fraction=0.046, pad=0.04)
# plt.subplot(2, 2, 3)
# ax2 = plt.hist(np.ndarray.flatten(
#     predicted_maps[predicted_maps != 0]), bins=40)
# plt.title("Predicted")
# plt.subplot(2, 2, 4)
# im = plt.imshow(predicted_maps[0, :, :, 0])
# plt.axis('off')
# plt.colorbar(im, fraction=0.046, pad=0.04)
# plt.show()

# tweak the predicted values from the maps
if dti_param == 'dt':
    predicted_maps = (1/norm_factor) * predicted_maps
    test_output_arrays = (1/norm_factor) * test_output_arrays

f = plt.figure(figsize=(20, 8))
plt.subplot(1, 3, 1)
plt.hist(np.ndarray.flatten(
    test_output_arrays[test_output_arrays != 0]), bins=50, label='GT')
plt.hist(np.ndarray.flatten(
    predicted_maps[predicted_maps != 0]), bins=50, label='Pred')
plt.legend()
plt.subplot(1, 3, 2)
im = plt.imshow(test_output_arrays[0, :, :, 0], vmin=v_min, vmax=v_max)
plt.axis('off')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.subplot(1, 3, 3)
im = plt.imshow(predicted_maps[0, :, :, 0], vmin=v_min, vmax=v_max)
plt.axis('off')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.savefig(trained_path + '/pixel_values_histogram.png', dpi=f.dpi, bbox_inches='tight',
            pad_inches=0, transparent=False)
plt.close()

# # load the MATLAB colourmaps
# fa_cmap = mpl.colors.ListedColormap(
#     np.genfromtxt('fa_cmap.csv', delimiter=','))
# md_cmap = mpl.colors.ListedColormap(
#     np.genfromtxt('md_cmap.csv', delimiter=','))
# ha_cmap = mpl.colors.ListedColormap(
#     np.genfromtxt('ha_cmap.csv', delimiter=','))
# e2a_cmap = mpl.colors.ListedColormap(
#     np.genfromtxt('e2a_cmap.csv', delimiter=','))


# save for each validation image the magnitude image, the human mask, and the predicted mask
if not os.path.exists(os.path.join(trained_path, 'prediction_maps')):
    os.makedirs(os.path.join(trained_path, 'prediction_maps'))


def crop_images(ref, img_a, img_b):
    sum_x = np.sum(ref, axis=0)
    sum_x = np.nonzero(sum_x)[0]
    x_start = sum_x[0]
    x_end = sum_x[-1]

    sum_y = np.sum(ref, axis=1)
    sum_y = np.nonzero(sum_y)[0]
    y_start = sum_y[0]
    y_end = sum_y[-1]

    ref = np.delete(ref, np.s_[y_end+1:ref.shape[0]], axis=0)
    ref = np.delete(ref, np.s_[0:y_start], axis=0)
    ref = np.delete(ref, np.s_[x_end+1:ref.shape[1]], axis=1)
    ref = np.delete(ref, np.s_[0:x_start], axis=1)

    img_a = np.delete(img_a, np.s_[y_end+1:img_a.shape[0]], axis=0)
    img_a = np.delete(img_a, np.s_[0:y_start], axis=0)
    img_a = np.delete(img_a, np.s_[x_end+1:img_a.shape[1]], axis=1)
    img_a = np.delete(img_a, np.s_[0:x_start], axis=1)

    img_b = np.delete(img_b, np.s_[y_end+1:img_b.shape[0]], axis=0)
    img_b = np.delete(img_b, np.s_[0:y_start], axis=0)
    img_b = np.delete(img_b, np.s_[x_end+1:img_b.shape[1]], axis=1)
    img_b = np.delete(img_b, np.s_[0:x_start], axis=1)

    # pad image
    ref = np.pad(ref, ((3, 3)))
    if img_a.ndim == 3:
        img_a = np.pad(img_a, ((3, 3), (3, 3), (0, 0)))
        img_b = np.pad(img_b, ((3, 3), (3, 3), (0, 0)))
    else:
        img_a = np.pad(img_a, ((3, 3)))
        img_b = np.pad(img_b, ((3, 3)))

    # plt.figure(1)
    # plt.subplot(1,2,1)
    # im = plt.imshow(ref)
    # plt.subplot(1,2,2)
    # im = plt.imshow(t)
    # plt.show()

    return ref, img_a, img_b


loss = np.empty(num_test_examples)
for i in range(num_test_examples):

    a = test_input_arrays[i, :, :, 0]

    if dti_param == 'dt':
        d = test_output_arrays[i, :, :, :]
        e = predicted_maps[i, :, :, :]

    a, d, e = crop_images(a, d, e)

    # create a background mask to match MATLAB
    mask = np.ones(a.shape)
    mask[a != 0] = 0
    mask = np.ma.masked_where(mask < 1E-3, mask)

    # remove background bases on output arrays
    f = np.absolute(np.subtract(d, e))

    # find all non-background pixels
    temp = custom_loss(d, e)
    loss[i] = temp.numpy()

    plt.figure(figsize=(20, 5))
    plt.subplot(2, 6, 1)
    im = plt.imshow(d[:, :, 0], cmap='viridis', vmin=v_min, vmax=v_max)
    plt.title("xx")
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    im = plt.imshow(mask, cmap='gray', alpha=1, vmin=1, vmax=2)

    plt.subplot(2, 6, 2)
    im = plt.imshow(d[:, :, 1], cmap='viridis', vmin=v_min, vmax=v_max)
    plt.title("xy")
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    im = plt.imshow(mask, cmap='gray', alpha=1, vmin=1, vmax=2)

    plt.subplot(2, 6, 3)
    im = plt.imshow(d[:, :, 2], cmap='viridis', vmin=v_min, vmax=v_max)
    plt.title("xz")
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    im = plt.imshow(mask, cmap='gray', alpha=1, vmin=1, vmax=2)

    plt.subplot(2, 6, 4)
    im = plt.imshow(d[:, :, 3], cmap='viridis', vmin=v_min, vmax=v_max)
    plt.title("yy")
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    im = plt.imshow(mask, cmap='gray', alpha=1, vmin=1, vmax=2)

    plt.subplot(2, 6, 5)
    im = plt.imshow(d[:, :, 4], cmap='viridis', vmin=v_min, vmax=v_max)
    plt.title("yz")
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    im = plt.imshow(mask, cmap='gray', alpha=1, vmin=1, vmax=2)

    plt.subplot(2, 6, 6)
    im = plt.imshow(d[:, :, 5], cmap='viridis', vmin=v_min, vmax=v_max)
    plt.title("zz")
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    im = plt.imshow(mask, cmap='gray', alpha=1, vmin=1, vmax=2)

    plt.subplot(2, 6, 7)
    im = plt.imshow(e[:, :, 0], cmap='viridis', vmin=v_min, vmax=v_max)
    plt.title("xx")
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    im = plt.imshow(mask, cmap='gray', alpha=1, vmin=1, vmax=2)

    plt.subplot(2, 6, 8)
    im = plt.imshow(e[:, :, 1], cmap='viridis', vmin=v_min, vmax=v_max)
    plt.title("xy")
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    im = plt.imshow(mask, cmap='gray', alpha=1, vmin=1, vmax=2)

    plt.subplot(2, 6, 9)
    im = plt.imshow(e[:, :, 2], cmap='viridis', vmin=v_min, vmax=v_max)
    plt.title("xz")
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    im = plt.imshow(mask, cmap='gray', alpha=1, vmin=1, vmax=2)

    plt.subplot(2, 6, 10)
    im = plt.imshow(e[:, :, 3], cmap='viridis', vmin=v_min, vmax=v_max)
    plt.title("yy")
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    im = plt.imshow(mask, cmap='gray', alpha=1)

    plt.subplot(2, 6, 11)
    im = plt.imshow(e[:, :, 4], cmap='viridis', vmin=v_min, vmax=v_max)
    plt.title("yz")
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    im = plt.imshow(mask, cmap='gray', alpha=1, vmin=1, vmax=2)

    plt.subplot(2, 6, 12)
    im = plt.imshow(e[:, :, 5], cmap='viridis', vmin=v_min, vmax=v_max)
    plt.title("zz")
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    im = plt.imshow(mask, cmap='gray', alpha=1, vmin=1, vmax=2)

    path, file = os.path.split(test_inputs[i])
    plt.savefig(os.path.join(trained_path, 'prediction_maps') + '/' + file[0:-4] + '_' + str(i), bbox_inches='tight',
                pad_inches=0, transparent=False)
    plt.close()


# histogram with median error
plt.figure(figsize=(10, 10))
plt.hist(loss, bins=20)
plt.title('Median MAE = {:.2}'.format(np.median(loss)))
plt.savefig(trained_path + '/' + 'MSE', bbox_inches='tight',
            pad_inches=0, transparent=False)
plt.close()
