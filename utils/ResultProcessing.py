import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

"""## Generate Images

Write a function to plot some images during training.

* We pass images from the test dataset to the generator.
* The generator will then translate the input image into the output.
* Last step is to plot the predictions and **voila!**

Note: The `training=True` is intentional here since
we want the batch statistics while running the model
on the test dataset. If we use training=False, we will get
the accumulated statistics learned from the training dataset (which we don't want)
"""


# Returns mean absolute error or l1 loss of predicted vs target image
def mae(predicted, target):
    return tf.reduce_mean(tf.abs(target - predicted))


def denormalise_output(arr, scale_factor):
    return arr / scale_factor


def generate_images(model, test_input, tar, scale_factor):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    target = denormalise_output(tar[0, :, :, 5], scale_factor)
    prediction = denormalise_output(prediction[0, :, :, 5], scale_factor)

    abs_diff = abs(target - prediction)
    mean_abs_diff = np.mean(abs_diff)
    median_abs_diff = np.median(abs_diff)
    mean_relative_difference = abs(np.mean(target) - np.mean(prediction)) / np.mean(target) * 100

    display_list = [target, prediction, abs_diff]
    title = ['Ground Truth', 'Predicted Image', 'Absolute Difference']

    fig = plt.figure(frameon=False)
    fig.set_size_inches(20, 20)

    for i in range(3):
        plt.subplot(1, 4, i + 1)
        # Get rid of white colour if not in dark mode
        plt.title(title[i], {'color': 'white'})
        im = plt.imshow(display_list[i])
        if title[i] != 'Input':
            plt.clim(0, 0.003)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.axis('off')
    plt.show()
    print("Mean absolute error: " + str(mean_abs_diff))
    print("Median absolute error: " + str(median_abs_diff))
    print("Mean difference percentage error: " + str(mean_relative_difference))
