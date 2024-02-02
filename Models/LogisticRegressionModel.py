from tensorflow.keras.models import Model
from tensorflow import GradientTape
import tensorflow as tf
from abc import ABC
from tensorflow.keras.layers import Dense, Flatten


class LogisticRegressionModel(Model, ABC, object):

    def __init__(self, ds, unique_labels=4):
        super(LogisticRegressionModel, self).__init__()
        self.ds = []
        self.i = Flatten()
        for d in ds:
            self.ds.append(Dense(units=d, activation='relu'))
        self.o = Dense(units=unique_labels, activation='softmax')

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.gpu = tf.device("/GPU:0")

    def call(self, inputs, training=None, mask=None):
        x = self.i(inputs)
        for d in self.ds:
            x = d(x)
        x = self.o(x)
        return x

    @tf.function
    def LR_model_train_step(self, x_batch, y_batch, optim):
        with GradientTape() as t:
            logits = self(x_batch)
            loss_value = self.loss(y_batch, logits)

        gradients = t.gradient(loss_value, self.trainable_weights)
        optim.apply_gradients(zip(gradients, self.trainable_weights))

        self.train_loss(loss_value)
        self.train_accuracy(y_batch, logits)

    @tf.function
    def LR_test_step(self, images, labels):
        predictions = self(images, training=False)
        t_loss = self.loss(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def train_model(self, optim, epochs, train_dataset, validation_dataset,
                    dynamic_saver=None):
        with self.gpu:

            for epoch in range(epochs):
                print(f"Epoch {epoch+1} of training is now starting.")
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()
                self.test_loss.reset_states()
                self.test_accuracy.reset_states()

                for step, (images, labels) in enumerate(train_dataset):
                    print(f"Starting train step {step} of epoch {epoch+1}")
                    self.LR_model_train_step(x_batch=images, y_batch=labels,
                                             optim=optim)

                for step, (images, labels) in enumerate(validation_dataset):
                    print(f"Starting test step {step} of epoch {epoch+1}")
                    self.LR_test_step(images, labels)

                if dynamic_saver is not None:
                    if epoch % dynamic_saver.iota == 0:
                        dynamic_saver.save(model=self)
                print(
                    f'Epoch {epoch + 1}, '
                    f'Loss: {self.train_loss.result()}, '
                    f'Accuracy: {self.train_accuracy.result() * 100}, '
                    f'Test Loss: {self.test_loss.result()}, '
                    f'Test Accuracy: {self.test_accuracy.result() * 100}'
                )

    def load_mask_detector_weights(self, path):
        self.load_weights(filepath=path)

    def save_mask_detector_weights(self, path, overwrite=True):
        self.save_weights(filepath=path, overwrite=overwrite)

    def save_mask_detector_model(self, path, overwrite):
        self.save(filepath=path, overwrite=overwrite)
