from utils import *

tf.random.set_seed(1234)

"""
# Ref: https://keras.io/examples/vision/pointnet/
"""

DATA_DIR = "data/FusionOFF"  # directory of dataset
NUM_CLASSES = 2  # number of classes to predict
NUM_EPOCHS = 3  # number of epochs

NUM_POINTS = 2048  # number of points to sample for each object
BATCH_SIZE = 64  # batch size


def create_model():
    """
    The main network can be then implemented in the same manner where the t-net mini models
    can be dropped in a layers in the graph. Here we replicate the network architecture
    published in the original paper but with half the number of weights at each layer as we
    are using the smaller 10 class ModelNet dataset.
    """

    inputs = keras.Input(shape=(NUM_POINTS, 3))

    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 128)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    return model


def train(save_path):
    """
    Set the number of points to sample and batch size and parse the dataset. This can take
    ~5minutes to complete.
    """

    print("Running Training")

    train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
        DATA_DIR, NUM_POINTS
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

    train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
    test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

    """
    ### Train model
    
    Once the model is defined it can be trained like any other standard classification model
    using `.compile()` and `.fit()`.
    """

    model = create_model()

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["sparse_categorical_accuracy"],
    )

    model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)

    model.save_weights(save_path)


def inference(load_path):
    """
    ## Visualize predictions

    We can use matplotlib to visualize our trained model performance.
    """
    print("Running Inference")

    train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
        DATA_DIR, NUM_POINTS
    )

    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
    test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

    model = create_model()
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["sparse_categorical_accuracy"],
    )
    model.load_weights(load_path)

    data = test_dataset.take(1)

    points, labels = list(data)[0]
    points = points[:8, ...]
    labels = labels[:8, ...]

    # run test data through model
    preds = model.predict(points)
    preds = tf.math.argmax(preds, -1)

    points = points.numpy()

    # plot points with predicted class and label
    fig = plt.figure(figsize=(20, 30))
    for i in range(8):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
        ax.set_title(
            "pred: {:}, label: {:}".format(
                CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
            )
        )
        ax.set_axis_off()

    plt.show()


if __name__ == "__main__":
    train("checkpoints/FusionOFF/model_weights")
    inference("checkpoints/FusionOFF/model_weights")
