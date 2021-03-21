import matplotlib.pyplot as plt
import tensorflow as tf

layers = tf.keras.layers
import numpy as np

print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist


def colorNormalization(images):
    """Normalize colors in the 255 range to the 0-1 range"""
    return images / 255.0


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = colorNormalization(x_train)
    x_test = colorNormalization(x_test)

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
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[y_train[i]])
    plt.show()


if __name__ == "__main__":
    main()
