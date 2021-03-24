import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.platform import tf_logging as logging

from keras.callbacks import EarlyStopping, Callback

from sklearn.model_selection import KFold
from sklearn.model_selection import KFold

import numpy as np

from pathlib import Path

import math

RUNNING_IN_COLAB = False

try:
    import google.colab  # pylint: disable=import-error

    RUNNING_IN_COLAB = True
except:
    RUNNING_IN_COLAB = False

print(f"Running in Collab: {RUNNING_IN_COLAB}")
print(f"Tensor Flow Version: {tf.__version__}\n")

MNIST = tf.keras.datasets.fashion_mnist


def getDatasets():
    """Load and Augment Training Data. Load Testing Data"""
    # Load data
    (x_train, y_train), (x_test, y_test) = MNIST.load_data()

    # Fix missing dimension and normalize
    x_train = np.array(tf.expand_dims(x_train / 255.0, axis=-1))
    x_test = np.array(tf.expand_dims(x_test / 255.0, axis=-1))

    # Data Augmentation
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip(
                "horizontal_and_vertical"
            ),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
            tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),
        ]
    )

    x_train = np.concatenate((x_train, data_augmentation(x_train, training=True)), axis=0)
    y_train = np.concatenate((y_train, y_train), axis=0)

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

    # Second Model
    model2_name = "Second_Model"

    def generateModel2():
        model = Sequential(name=model2_name)
        model.add(
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape)
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
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

    models.append((model2_name, generateModel2))

    # TODO: Other models follow a similar structure to define them
    return models


# Path info
ROOT = Path(".") / "results"
if not RUNNING_IN_COLAB and not ROOT.exists():
    ROOT.mkdir()

TRAINING_FOLDER = ROOT / "training"
if not RUNNING_IN_COLAB and not TRAINING_FOLDER.exists():
    TRAINING_FOLDER.mkdir()

HISTORY_FOLDER = TRAINING_FOLDER / "history"
if not RUNNING_IN_COLAB and not HISTORY_FOLDER.exists():
    HISTORY_FOLDER.mkdir()

COMPARISON_FOLDER = TRAINING_FOLDER / "comparison"
if not RUNNING_IN_COLAB and not COMPARISON_FOLDER.exists():
    COMPARISON_FOLDER.mkdir()

TESTING_FOLDER = ROOT / "testing"
if not TESTING_FOLDER.exists():
    TESTING_FOLDER.mkdir()

MODELS_FOLDER = ROOT / "models"
if not MODELS_FOLDER.exists():
    MODELS_FOLDER.mkdir()


def plotTrainingHistory(folder, title, filename, history, bestEpoch):
    """Plot training history"""
    fig = plt.figure()
    plt.axes()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    plt.ylim(0.0, 1.0)

    # Calculate epoch info
    epoch_number = len(next(iter(history.values())))
    plt.xticks(range(1, epoch_number + 1))

    # Plot everything
    plt.plot(range(1, epoch_number + 1), history["loss"], label="Loss", color="r")
    plt.plot(
        range(1, epoch_number + 1), history["accuracy"], label="Accuracy", color="gold"
    )
    plt.plot(
        range(1, epoch_number + 1),
        history["val_loss"],
        label="Validation Loss",
        color="b",
    )
    plt.plot(
        range(1, epoch_number + 1),
        history["val_accuracy"],
        label="Validation  Accuracy",
        color="magenta",
    )
    plt.axvline(x=bestEpoch + 1, label="Best Epoch", color="lime")

    # Draw legend
    plt.legend(loc="lower left")

    if RUNNING_IN_COLAB:
        # On Google Colab is better to show the image
        plt.show()
    else:
        fig.savefig(folder / f"{filename}.png", dpi=fig.dpi)


def processKFoldScores(folder, title, filename, scores):
    """Process and plot k-fold training"""
    loss_scores = [0 for _ in range(len(scores))]
    accuracy_scores = [0 for _ in range(len(scores))]

    # Separate values
    for index, (loss, acc) in enumerate(scores):
        loss_scores[index] = loss
        accuracy_scores[index] = acc

    loss_scores = np.array(loss_scores)
    accuracy_scores = np.array(accuracy_scores)

    # Box Plot values
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel("Validation")
    plt.ylabel("Value")

    plt.ylim(0.0, 1.0)

    bp = ax.boxplot(
        [loss_scores, accuracy_scores], labels=["Loss", "Accuracy"], showmeans=True
    )

    means = [np.mean(loss_scores), np.mean(accuracy_scores)]
    stds = [np.std(loss_scores), np.std(accuracy_scores)]

    # From: https://stackoverflow.com/a/58068611
    for i, line in enumerate(bp["medians"]):
        x, y = line.get_xydata()[1]
        text = " μ={:.2f}\n σ={:.2f}".format(means[i], stds[i])
        ax.annotate(text, xy=(x, y))

    if RUNNING_IN_COLAB:
        # On Google Colab is better to show the image
        plt.show()
    else:
        fig.savefig(folder / f"{filename}.png", dpi=fig.dpi)

    # Return Info
    return (means, stds)


class BestEpochCallback(Callback):
    def __init__(self, *args, **kwargs):
        super(BestEpochCallback, self).__init__(*args, **kwargs)
        self.monitor = "val_loss"
        self.reset()

    def reset(self):
        self.bestEpoch = -1
        self.bestValue = math.inf
        self.bestWeights = None

    def get_monitor_value(self, logs):
        # From https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/callbacks.py#L1660-L1791
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                "Early stopping conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        return monitor_value

    def on_epoch_end(self, epoch, logs=None):
        value = self.get_monitor_value(logs)
        if value < self.bestValue:
            self.bestEpoch = epoch
            self.bestValue = value
            self.bestWeights = self.model.get_weights()


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
    """ 
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[y_train[i]])
    plt.show() 
    """

    # Models to run
    models = getModelsGenerators()

    # Save models info
    for (name, generator) in models:
        model = generator()

        filename = None
        if RUNNING_IN_COLAB:
            # On Google Colab is better to show the image
            filename = f"{name.lower()}"
            model.summary()
        else:
            filename = MODELS_FOLDER / f"{name.lower()}"
            with open(f"{filename}.txt", "w") as f:
                model.summary(print_fn=lambda x: f.write(x + "\n"))
                f.close()

        # Model Summary Image
        tf.keras.utils.plot_model(
            model,
            to_file=f"{filename}.png",
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )

        # Save Model Architecture
        model.save(filename)

    # Models for Final training
    num_models_final = 2

    # Models' scores
    models_scores = [[] for _ in range(len(models))]

    # K fold validation
    num_folds = 10
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # Training parameters
    no_epochs = 15
    batch_size = 64
    verbosity = 1
    final_validation_split = 0.1

    # Early stopping to avoid overfitting
    patience = int(math.ceil(0.3 * no_epochs))
    monitor = "val_loss"
    mode = "min"

    # Misc variables for loop
    fold_number = 1
    for train_index, test_index in kfold.split(x_train, y_train):
        x_fold_train = x_train[train_index]
        y_fold_train = y_train[train_index]

        x_fold_test = x_train[test_index]
        y_fold_test = y_train[test_index]

        for index, (name, generator) in enumerate(models):
            # Callbacks for training
            callbacks = []
            early_stopping = EarlyStopping(
                monitor=monitor, mode=mode, verbose=verbosity, patience=patience
            )
            callbacks.append(early_stopping)

            # Saving the best model
            best_epoch = BestEpochCallback()
            callbacks.append(best_epoch)

            # Copy model to train copy
            model = generator()

            # Train and Test
            print(
                f'Training and Testing Model "{name}" for fold {fold_number}/{num_folds}'
            )
            history = model.fit(
                x_fold_train,
                y_fold_train,
                batch_size=batch_size,
                epochs=no_epochs,
                verbose=verbosity,
                validation_data=(x_fold_test, y_fold_test),
                callbacks=callbacks,
            )

            # Plot
            plotTrainingHistory(
                HISTORY_FOLDER,
                f'Model "{name}" Fold-{fold_number}/{num_folds} Training. Best Epoch {best_epoch.bestEpoch+1}.',
                f"{name.lower()}-fold-{fold_number}-{num_folds}",
                history.history,
                best_epoch.bestEpoch,
            )

            # Save Test
            models_scores[index].append(
                (
                    history.history["val_loss"][best_epoch.bestEpoch],
                    history.history["val_accuracy"][best_epoch.bestEpoch],
                )
            )
            print("")

        fold_number += 1

    # Models' processed scores
    models_processed_scores = [[] for _ in range(len(models))]
    models_processed_csv_scores = [{} for _ in range(len(models))]

    for index, (name, _) in enumerate(models):
        means, stds = processKFoldScores(
            COMPARISON_FOLDER,
            f'Model "{name}" {num_folds}-Fold Training',
            f"{name.lower()}-folds-{num_folds}",
            models_scores[index],
        )
        models_processed_scores[index] = means
        models_processed_csv_scores[index] = {
            "model_name": name,
            "val_loss_mean": means[0],
            "val_loss_std": means[1],
            "val_accuracy_mean": stds[0],
            "val_accuracy_std": stds[1],
        }

    if not RUNNING_IN_COLAB:
        import csv

        with open(COMPARISON_FOLDER / "models_values.csv", "w") as f:
            fieldnames = [
                "model_name",
                "val_loss_mean",
                "val_loss_std",
                "val_accuracy_mean",
                "val_accuracy_std",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for score in models_processed_csv_scores:
                writer.writerow(score)

    # Convert to numpy array
    models_processed_scores = np.array(models_processed_scores)
    comparison_axis = 0
    models_idx = np.argpartition(
        models_processed_scores, num_models_final - 1, axis=comparison_axis
    )

    # Get Selected models for final training
    print("Selected Models")
    final_models = [() for _ in range(num_models_final)]
    for index in range(num_models_final):
        og_index = models_idx[index][comparison_axis]

        final_models[index] = models[og_index]
        name = final_models[index][0]

        print(
            f"Model {index+1}: {name} with mean Loss of {models_processed_scores[og_index][0]:.4f} and mean Accuracy of {models_processed_scores[og_index][1]:.4f}"
        )
    print("")

    # Shuffle training data for final training
    p = np.random.permutation(len(x_train))
    x_train = x_train[p]
    y_train = y_train[p]

    # Scores for CSV
    final_scores = [{} for _ in range(len(final_models))]

    for index, (name, generator) in enumerate(final_models):
        # Callbacks for training
        callbacks = []
        early_stopping = EarlyStopping(
            monitor=monitor, mode=mode, verbose=verbosity, patience=patience
        )
        callbacks.append(early_stopping)

        # Saving the best model
        best_epoch = BestEpochCallback()
        callbacks.append(best_epoch)

        # Copy model to train copy
        model = generator()

        # Train and Test
        print(f'Training Final Model {index+1}: "{name}"')
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=final_validation_split,
            callbacks=callbacks,
        )

        filename = f"final-model-{index+1}-{name.lower()}"
        # Plot
        plotTrainingHistory(
            TESTING_FOLDER,
            f'Final Model {index+1} "{name}" Training. Best Epoch {best_epoch.bestEpoch+1}.',
            filename,
            history.history,
            best_epoch.bestEpoch,
        )

        # Set Best Weights
        model.set_weights(best_epoch.bestWeights)

        # Save Model
        model.save(TESTING_FOLDER / filename)

        # Get and Print Results
        val_loss = history.history["val_loss"][best_epoch.bestEpoch]
        val_acc = history.history["val_accuracy"][best_epoch.bestEpoch]

        print(f'Testing Final Model {index+1}: "{name}"')
        test_loss, test_acc = model.evaluate(
            x_test, y_test, batch_size=batch_size, verbose=verbosity
        )

        print(
            f'Final Model {index+1} "{name}" with Testing Loss {test_loss:.4f}, Testing Accuracy {test_acc:.4f}, Validation Loss {val_loss:.4f} and Validation Accuracy {val_acc:.4f}'
        )

        # Save for CSV
        final_scores[index] = {
            "model_number": index + 1,
            "model_name": name,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

        print("")

    if not RUNNING_IN_COLAB:
        import csv

        with open(TESTING_FOLDER / "models_values.csv", "w") as f:
            fieldnames = [
                "model_number",
                "model_name",
                "test_loss",
                "test_acc",
                "val_loss",
                "val_acc",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for score in final_scores:
                writer.writerow(score)


if __name__ == "__main__":
    main()
