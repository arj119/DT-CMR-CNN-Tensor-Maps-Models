import numpy as np
import tensorflow as tf


# Normalise array to range [-1,1]
def scale_output(array, scale_factor):
    array = array * scale_factor
    return array


def normalise_matrices(array):
    array = np.true_divide(array, 127.5) - 1
    return array


def load_data_into_array(name, scale_factor, is_input=True, normalise_input=True):
    array = np.load(name)
    array = array['arr_0']
    array = np.float32(array)
    if is_input:
        if normalise_input:
            array = normalise_matrices(array)
    else:
        array = scale_output(array, scale_factor)
    array = np.expand_dims(array, axis=0)
    return array


def data_info(data_arr, dti_param, is_output=False):
    data = data_arr.flatten()
    out = ""
    if is_output:
        print("Output data from", dti_param)
    else:
        print("Input data from", dti_param)
    print('min', np.amin(data))
    print('max', np.amax(data))
    print('mean', np.mean(data))
    print()


# load numpy arrays
def load_npz_files(input_list, output_list, dti_param, output_channels):
    ''' load input and output arrays from lists of npz files '''
    # loop through the list and stack arrays
    global input_arrays, output_arrays
    for idx, val in enumerate(input_list):
        if idx == 0:
            input_arrays = load_data_into_array(val)
        else:
            temp = load_data_into_array(val)
            input_arrays = np.concatenate((input_arrays, temp), axis=0)

    for idx, val in enumerate(output_list):
        if idx == 0:
            output_arrays = load_data_into_array(val, is_input=False)
        else:
            temp = load_data_into_array(val, is_input=False)
            output_arrays = np.concatenate((output_arrays, temp), axis=0)

    if output_channels == 1:
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


#  Data augmentation functions
def random_crop(img_in, img_out, height, width, input_channels, output_channels):
    #  stack images before rotation so that the rotation is the same
    stacked_image = tf.concat([img_in, img_out], axis=2)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[height, width, input_channels + output_channels])

    r_img_in = cropped_image[:, :, 0:input_channels]
    r_img_out = cropped_image[:, :, input_channels:]
    # if OUTPUT_CHANNELS == 1:
    #     r_img_out = tf.expand_dims(r_img_out, axis=2)

    return r_img_in, r_img_out


def augment(img_in, img_out, height, width):
    # Add 20 pixels padding
    add_n = 20
    img_in = tf.image.resize_with_crop_or_pad(
        img_in, height + add_n, width + add_n)
    img_out = tf.image.resize_with_crop_or_pad(
        img_out, height + add_n, width + add_n)
    img_in, img_out = random_crop(img_in, img_out)

    return img_in, img_out


# get the tensorflow train and validate datasets to tensorflow
def get_baseline_dataset(input_list, output_list, dti_param):
    # read arrays
    input_arrays, output_arrays = load_npz_files(
        input_list, output_list, dti_param)

    dataset = tf.data.Dataset.from_tensor_slices((input_arrays, output_arrays))
    data_info(input_arrays, False)
    data_info(output_arrays, True)
    return dataset
