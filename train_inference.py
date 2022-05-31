from utils import *

tf.random.set_seed(1234)

"""
# Ref: https://keras.io/examples/vision/pointnet/
"""

DATA_DIR = "data/FusionOFF"  # directory of dataset
NUM_CLASSES = 8  # number of classes to predict
NUM_EPOCHS = 5  # number of epochs

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


def train(save_path, load_path=None):
    """
    Set the number of points to sample and batch size and parse the dataset. This can take
    ~5minutes to complete.
    """

    print(f"[Running Training: {NUM_EPOCHS} Epochs]")

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

    if load_path:
        print("Loading weights from:", load_path)
        model.load_weights(load_path)

    model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)

    model.save_weights(save_path)


def draw_confusion(predicted, truths, labels):
    confusion = confusion_matrix(y_true=truths, y_pred=predicted)

    print(confusion)

    plt.figure(figsize=(24, 18))
    label = labels
    sn.heatmap(confusion, annot=False, fmt='.2f', cmap='Blues', xticklabels=label, yticklabels=label)
    plt.xticks(size='xx-large', rotation=45)
    plt.yticks(size='xx-large', rotation=45)
    plt.tight_layout()

    plt.show()
    plt.savefig("inference_results\\confusion_matrix.pdf", format='pdf')

    print(classification_report(truths, predicted))


def inference(load_path):
    """
    ## Visualize predictions

    We can use matplotlib to visualize our trained model performance.
    """
    print("[Running Inference]")
    if os.path.exists("inference_results"):
        shutil.rmtree("inference_results")
    os.mkdir("inference_results")

    _, test_points, _, test_labels, CLASS_MAP = parse_dataset(
        DATA_DIR, NUM_POINTS
    )

    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
    test_dataset = test_dataset.shuffle(len(test_points)).batch(len(test_points))

    model = create_model()
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["sparse_categorical_accuracy"],
    )
    model.load_weights(load_path)

    # data = test_dataset.take(1)
    data = test_dataset
    points, labels = list(data)[0]  # choose the first batch

    print(f"[Inference on: {len(points)} objects]")

    points = points[:, ...]
    labels = labels[:, ...]

    # run test data through model
    preds = model.predict(points)
    preds = tf.math.argmax(preds, -1)

    points = points.numpy()

    total = len(points)
    correct = 0

    correct_cnt = 0
    faulty_cnt = 0

    # plot points with predicted class and label

    for i in tqdm(range(len(points)), desc="Visualizing Inference Results"):

        # create plots
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
        ax.set_title(
            "[GROUND TRUTH]: {:} | [PRED]: {:}".format(
                CLASS_MAP[labels.numpy()[i]], CLASS_MAP[preds[i].numpy()]
            )
        )
        ax.set_axis_off()

        if CLASS_MAP[labels.numpy()[i]] == CLASS_MAP[preds[i].numpy()]:
            correct += 1
            correct_cnt += 1
            save_name = f"correct_{correct_cnt}.png"
        else:
            faulty_cnt += 1
            save_name = f"faulty_{faulty_cnt}.png"

        plt.savefig(f"inference_results\\{save_name}")
        plt.close(fig)

    print(f"[Number of correct predictions]: {correct} / {total}; (Acc: {correct * 100 / total}%)")

    # Plot confusion matrix

    truths, predicts = [], []
    for i in range(len(points)):
        truths.append(CLASS_MAP[labels.numpy()[i]])
        predicts.append(CLASS_MAP[preds[i].numpy()])

    draw_confusion(truths, predicts, CLASS_MAP.values())


if __name__ == "__main__":
    # train(save_path="checkpoints/FusionOFF_20/model_weights",
    #       load_path="checkpoints/FusionOFF_15/model_weights")

    inference("checkpoints/FusionOFF_15/model_weights")
