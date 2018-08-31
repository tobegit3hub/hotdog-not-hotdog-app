#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import keras
import tensorflow as tf
from keras import backend as K
from keras.layers.core import Activation, Dense, Dropout
from keras.optimizers import SGD
from tensorflow.python.util import compat


def build_dnn_model():
  input_size = 784
  output_size = 10
  model = keras.models.Sequential()
  model.add(Dense(128, input_shape=(input_size, )))
  model.add(Activation("relu"))
  model.add(Dense(64))
  model.add(Activation("relu"))
  model.add(Dense(32))
  model.add(Activation("relu"))
  model.add(Dense(output_size))
  model.add(Activation("softmax"))
  return model


def export_savedmodel(model):
  print("input: {}, output: {}".format(model.input, model.output))
  model_signature = tf.saved_model.signature_def_utils.predict_signature_def(
      inputs={'input': model.input}, outputs={'output': model.output})

  model_path = "model"
  model_version = 1
  export_path = os.path.join(
      compat.as_bytes(model_path), compat.as_bytes(str(model_version)))
  logging.info("Export the model to {}".format(export_path))

  builder = tf.saved_model.builder.SavedModelBuilder(export_path)
  builder.add_meta_graph_and_variables(
      sess=K.get_session(),
      tags=[tf.saved_model.tag_constants.SERVING],
      clear_devices=True,
      signature_def_map={
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          model_signature
      })
  builder.save()


def main():
  # Define hyper-parameters
  OUTPUT_CLASS = 10
  EPOCH_NUMBER = 3
  BATCH_SIZE = 128
  VERBOSE = 1
  OPTIMIZER = SGD()
  VALIDATION_SPLIT = 0.2
  LOSS_TYPE = "categorical_crossentropy"

  # Get training dataset
  (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
  X_train = X_train.reshape(60000, 784)
  X_test = X_test.reshape(10000, 784)
  X_train = X_train.astype("float32")
  X_test = X_test.astype("float32")
  X_train = X_train / 255
  X_test = X_test / 255
  Y_train = keras.utils.np_utils.to_categorical(Y_train, OUTPUT_CLASS)
  Y_test = keras.utils.np_utils.to_categorical(Y_test, OUTPUT_CLASS)

  # Build or load model
  model = build_dnn_model()
  model.summary()
  model.compile(loss=LOSS_TYPE, optimizer=OPTIMIZER, metrics=["accuracy"])

  # Train or load variables
  model.fit(
      X_train,
      Y_train,
      batch_size=BATCH_SIZE,
      epochs=EPOCH_NUMBER,
      validation_split=VALIDATION_SPLIT,
      verbose=VERBOSE)

  # Example saved model
  export_savedmodel(model)

  # Make prediction
  metrics = model.evaluate(X_test, Y_test, verbose=VERBOSE)
  test_loss = metrics[0]
  test_accuracy = metrics[1]
  print("Loss: {}, accuracy: {}".format(test_loss, test_accuracy))


if __name__ == "__main__":
  main()