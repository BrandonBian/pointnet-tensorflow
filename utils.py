import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
import time
import seaborn as sn
import shutil

from tqdm import tqdm
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def parse_dataset(DATA_DIR, num_points=2048):
    """
    To generate a `tf.data.Dataset()` we need to first parse through the ModelNet data
    folders. Each mesh is loaded and sampled into a point cloud before being added to a
    standard python list and converted to a `numpy` array. We also store the current
    enumerate index value as the object label and use a dictionary to recall this later.
    """

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}  # <key, val> = <folder name, material category id>
    folders = glob.glob(os.path.join(DATA_DIR, "*"))
    final_folders = []

    cnt = 0
    for folder in folders:
        class_id = os.path.basename(folder)
        if class_id == "README.txt":
            continue
        cnt += 1
        print(f"Detected class {cnt}: ", class_id)
        final_folders.append(folder)

    time.sleep(3)

    for i, folder in enumerate(tqdm(final_folders, desc="Processing Class Folders")):
        # print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_id = os.path.basename(folder)
        if class_id == "README.txt":
            continue
        class_map[i] = class_id
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )


def augment(points, label):
    """
    Our data can now be read into a `tf.data.Dataset()` object. We set the shuffle buffer
    size to the entire size of the dataset as prior to this the data is ordered by class.
    Data augmentation is important when working with point cloud data. We create a
    augmentation function to jitter and shuffle the train dataset.
    """
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


"""
### Build a model

Each convolution and fully-connected layer (with exception for end layers) consits of
Convolution / Dense -> Batch Normalization -> ReLU Activation.
"""


def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


"""
PointNet consists of two core components. The primary MLP network, and the transformer
net (T-net). The T-net aims to learn an affine transformation matrix by its own mini
network. The T-net is used twice. The first time to transform the input features (n, 3)
into a canonical representation. The second is an affine transformation for alignment in
feature space (n, 3). As per the original paper we constrain the transformation to be
close to an orthogonal matrix (i.e. ||X*X^T - I|| = 0).
"""


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))


"""
 We can then define a general function to build T-net layers.
"""


def tnet(inputs, num_features):
    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])
