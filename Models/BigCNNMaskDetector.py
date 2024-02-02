from tensorflow.keras.models import Model
from tensorflow import GradientTape
import tensorflow as tf
from abc import ABC
from tensorflow.keras.layers import Dense, Flatten, Dropout
from Layers import ConvBlock


class FaceMaskDetector(Model, ABC, object):

    def __init__(self, mdix, mdiy, mdc):
        super(FaceMaskDetector, self).__init__()
        self.mdix = mdix
        self.mdiy = mdiy
        self.mdc = mdc

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.cb1 = ConvBlock(convs=[[128, (3, 3), None], [128, (3, 3), None], [128, (3, 3), None]],
                             dropout=[True, 0.3], pool=[(2, 2), None], first=True, input_shape=(mdix, mdiy, mdc))
        self.cb2 = ConvBlock(convs=[[128, (3, 3), None], [128, (3, 3), None], [128, (3, 3), None]],
                             dropout=[True, 0.3], pool=[(2, 2), None])
        self.cb3 = ConvBlock(convs=[[128, (3, 3), None], [128, (3, 3), None], [128, (3, 3), None]],
                             dropout=[False, 0.3], pool=[(2, 2), None])
        self.f = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.dr1 = Dropout(0.2)
        self.d2 = Dense(128, activation='relu')
        self.dr2 = Dropout(0.2)
        self.d3 = Dense(128, activation='relu')
        self.e = Dense(4, activation='softmax')

        self.gpu = tf.device("/GPU:0")

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def call(self, inputs, training=False, mask=None):
        x = self.cb1(inputs)
        x = self.cb2(x)
        x = self.cb3(x)
        x = self.f(x)
        x = self.d1(x)
        x = self.dr1(x)
        x = self.d2(x)
        x = self.dr2(x)
        x = self.d3(x)

        return self.e(x)

    @tf.function
    def model_train_step(self, x_batch, y_batch, optim):
        with GradientTape() as t:
            logits = self(x_batch)
            loss_value = self.loss(y_batch, logits)

        gradients = t.gradient(loss_value, self.trainable_weights)
        optim.apply_gradients(zip(gradients, self.trainable_weights))

        self.train_loss(loss_value)
        self.train_accuracy(y_batch, logits)

    @tf.function
    def test_step(self, images, labels):
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
                    self.model_train_step(x_batch=images, y_batch=labels,
                                          optim=optim)

                for step, (images, labels) in enumerate(validation_dataset):
                    print(f"Starting test step {step} of epoch {epoch+1}")
                    self.test_step(images, labels)

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