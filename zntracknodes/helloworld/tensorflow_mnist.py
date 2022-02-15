import tensorflow as tf
from zntrack import Node, zn


class PrepareMnist(Node):
    """Gather the MNIST dataset

    References
    ----------
    https://www.tensorflow.org/tutorials/quickstart/beginner
    """

    x_train = zn.outs()
    y_train = zn.outs()
    x_test = zn.outs()
    y_test = zn.outs()

    def run(self):
        """Download the dataset and save the normalized values"""
        mnist = tf.keras.datasets.mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0


class TrainMnistModel(Node):
    """Train a TensorFlow Model on the MNIST Dataset

    References
    ----------
    https://www.tensorflow.org/tutorials/quickstart/beginner
    """

    mnist_data: PrepareMnist = zn.deps(PrepareMnist.load())

    epochs = zn.params(5)

    model_metrics = zn.metrics()

    def run(self):
        """Train and evaluate the TensorFlow model on MNIST"""
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10),
            ]
        )

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

        model.fit(self.mnist_data.x_train, self.mnist_data.y_train, epochs=self.epochs)

        loss, accuracy = model.evaluate(self.mnist_data.x_test, self.mnist_data.y_test)
        self.model_metrics = {"loss": loss, "accuracy": accuracy}
