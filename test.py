import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold


from sklearn.model_selection import KFold

import numpy as np


layers = tf.keras.layers

print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist


def colorNormalization(images):
    """Normalize colors in the 255 range to the 0-1 range"""
    return images / 255.0


def getDatasets():
    """Get normalized Datasets to work"""
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize Colors to 0..1
    x_train = colorNormalization(x_train)
    x_test = colorNormalization(x_test)

    # Fix missing dimension
    x_train = np.array(tf.expand_dims(x_train, axis=-1))
    x_test = np.array(tf.expand_dims(x_test, axis=-1))

    return (x_train, y_train), (x_test, y_test)


def getModelsGenerators():
    input_shape = (28, 28, 1)
    no_classes = 10

    models = []

    # First Model
    model1_name = "First_Model"

    def generateModel1():
        model = Sequential(name=model1_name)
        model.add(
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape)
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(no_classes, activation="softmax"))

        # Compile for training
        model.compile(
            loss=sparse_categorical_crossentropy,
            optimizer=Adam(learning_rate=0.01),
            metrics=["accuracy"],
        )
        return model

    models.append((model1_name, generateModel1))

    # TODO: Other models follow a similar structure to define them
    return models


def plotTrainingHistory(title, filename, history):
    plt.figure()
    plt.axes()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    plt.ylim(0.0, 1.0)

    # Calculate epoch info
    epoch_number = len(next(iter(history.values())))
    plt.xticks(range(1, epoch_number + 1))

    # Plot everything
    for (name, data) in history.items():
        plt.plot(range(1, epoch_number + 1), data, label=name)

    # Draw legend
    plt.legend(loc="upper left")

    plt.show()


def main():
    (x_train, y_train), (x_test, y_test) = getDatasets()

    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    """ plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[y_train[i]])
    plt.show() """

    # Models to run
    models = getModelsGenerators()

    # Models' scores
    models_scores = [[] for _ in range(len(models))]

    # K fold validation
    num_folds = 10
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # Training parameters
    no_epochs = 15
    batch_size = 64
    verbosity = 1

    # Misc variables for loop
    fold_number = 1
    for train_index, test_index in kfold.split(x_train, y_train):
        x_fold_train = x_train[train_index]
        y_fold_train = y_train[train_index]

        x_fold_test = x_train[test_index]
        y_fold_test = y_train[test_index]

        for index, (name, generator) in enumerate(models):
            # Copy model to train copy
            model = generator()

            # Train and Test
            print(f'Training and Testing Model "{name}" for fold {fold_number}/{num_folds}')
            history = model.fit(
                x_fold_train,
                y_fold_train,
                batch_size=batch_size,
                epochs=no_epochs,
                verbose=verbosity,
                validation_data=(x_fold_test, y_fold_test),
            )

            # Plot
            plotTrainingHistory(
                f'Model "{name}" Fold-{fold_number}/{num_folds} Training', "", history.history
            )

            # Save Test
            models_scores[index].append(
                (history.history["val_loss"], history.history["val_accuracy"])
            )
            print("")

        fold_number += 1


if __name__ == "__main__":
    main()
