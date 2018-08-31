#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import pandas as pd
import time
import json
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (
    signature_constants, signature_def_utils, tag_constants, utils)


from nets import nets_factory

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def define_flags():
  """
  Define all the command-line parameters.

  Return:
    The FLAGS object.
  """

  flags = tf.app.flags
  flags.DEFINE_string("mode", "train", "Train or export model")
  flags.DEFINE_integer("image_width", 299, "Width of the image")
  flags.DEFINE_integer("image_height", 299, "Height of the image")
  flags.DEFINE_integer("channels", 3, "Channel of the image")
  flags.DEFINE_integer("label_size", 2, "Number of label size")
  flags.DEFINE_string("model", "dnn", "The model")
  flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
  flags.DEFINE_integer("epoch_number", 1, "Number of epoches")
  flags.DEFINE_integer("train_batch_size", 16, "Batch size for training")
  flags.DEFINE_integer("validation_batch_size", 16, "Batch size for validation")
  flags.DEFINE_float("validation_dataset_ratio", 0.1,
                     "The ratio of validation dataset to validate")
  flags.DEFINE_integer("train_shuffle_buffer_size", 1000,
                       "Train shuffle buffer size")
  flags.DEFINE_integer("validation_shuffle_buffer_size", 1000,
                       "Validation shuffle buffer size")
  flags.DEFINE_integer("steps_to_validate", 1, "Steps to validate")
  flags.DEFINE_boolean("load_checkpoint", False, "Load checkpoint or not")
  flags.DEFINE_string("checkpoint_file_path", "inception_v3.ckpt", "Checkpoint file path")
  flags.DEFINE_integer("steps_to_save_checkpoint", -1,
                       "Steps to save checkpoint")
  flags.DEFINE_string("save_checkpoint_path", "./checkpoint/",
                      "Save checkpoint path")
  flags.DEFINE_string("restore_checkpoint_include_scopes", "InceptionV3",
                      "Include variable scopes to restore")
  flags.DEFINE_string("restore_checkpoint_exclude_scopes", "InceptionV3/Logits,InceptionV3/AuxLogits",
                      "Exclude variable scopes to restore")
  flags.DEFINE_string("model_trainable_scopes", "",
                      "Use the variable scopes to train")
  flags.DEFINE_string("output_path", "./tensorboard/", "Path for tensorboard")
  flags.DEFINE_string("model_path", "./model/", "Path of the model")
  flags.DEFINE_integer("model_version", 1, "Version of the model")
  flags.DEFINE_string("train_csv_file", "./generated_hotdog_train.csv", "Train csv file")
  flags.DEFINE_string("validation_csv_file", "./generated_hotdog_test.csv",
                      "Validation csv file")
  FLAGS = flags.FLAGS

  # Print parameters
  FLAGS.model
  parameter_value_map = {}
  for key in FLAGS.__flags.keys():
    parameter_value_map[key] = FLAGS.__flags[key].value
  logging.info("Parameters: {}".format(parameter_value_map))

  return FLAGS


FLAGS = define_flags()

def _decode_image_file(filename, label):
  image_string = tf.read_file(filename)
  features = tf.image.decode_jpeg(image_string, channels=FLAGS.channels)
  features = tf.image.resize_images(features,
                                    [FLAGS.image_height, FLAGS.image_width])

  return features, label


def dnn_model(input, feature_size, label_size, is_training=True):
  """
  Implement the basic DNN model.
  """

  input = tf.reshape(input, (-1, feature_size))

  weights = tf.get_variable(
          "weight", [feature_size, label_size],
          dtype=tf.float32,
          initializer=tf.zeros_initializer)
  bias = tf.get_variable(
          "bias", [label_size], dtype=tf.float32, initializer=tf.zeros_initializer)

  logits = tf.matmul(input, weights) + bias

  return logits


def main():

  # Get the train images and labels
  train_dataframe = pd.read_csv(
      FLAGS.train_csv_file,
      delimiter=",",
      header=None,
      keep_default_na=False,
      na_values=[""],
      dtype={"label": "int64"}).sample(frac=1.0)
  train_image_list = train_dataframe[0].tolist()
  train_label_list = train_dataframe[1].tolist()
  train_instance_number = len(train_label_list)

  validation_dataframe = pd.read_csv(
      FLAGS.validation_csv_file,
      delimiter=",",
      header=None,
      dtype={"label": "int64"})
  validation_image_list = validation_dataframe[0].tolist()
  validation_label_list = validation_dataframe[1].tolist()
  validation_instance_number = len(validation_label_list)

  # Construct the dataset op
  train_image_list_placeholder = tf.placeholder(tf.string, [None])
  train_label_list_placeholder = tf.placeholder(tf.int64, [None])
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (train_image_list_placeholder, train_label_list_placeholder))
  if FLAGS.train_shuffle_buffer_size > 0:
    train_dataset = train_dataset.repeat(FLAGS.epoch_number).shuffle(
        FLAGS.train_shuffle_buffer_size).map(_decode_image_file).batch(
            FLAGS.train_batch_size)
  else:
    train_dataset = train_dataset.repeat(FLAGS.epoch_number).map(
        _decode_image_file).batch(FLAGS.train_batch_size)
  train_iterator = train_dataset.make_initializable_iterator()
  train_features_op, train_label_op = train_iterator.get_next()
  train_features_op = tf.cast(train_features_op, tf.float32)

  validation_image_list_placeholder = tf.placeholder(tf.string, [None])
  validation_label_list_placeholder = tf.placeholder(tf.int64, [None])
  validation_dataset = tf.data.Dataset.from_tensor_slices(
      (validation_image_list_placeholder, validation_label_list_placeholder))
  # Notice that epoch is not limited for validation dataset
  validation_dataset = validation_dataset.repeat().shuffle(
      FLAGS.validation_shuffle_buffer_size).map(_decode_image_file).batch(
          FLAGS.validation_batch_size)
  validation_iterator = validation_dataset.make_initializable_iterator()
  validation_features_op, validation_label_op = validation_iterator.get_next()
  validation_features_op = tf.cast(validation_features_op, tf.float32)

  # Define the model
  global_step = tf.Variable(
      0, name="global_step", dtype=tf.int64, trainable=False)

  def _model(input, is_training=True):

    # TODO: Use different preprocess for different models
    scaled_input_tensor = tf.scalar_mul((1.0 / 255), input)
    scaled_input_tensor = scaled_input_tensor - 0.5
    scaled_input_tensor = scaled_input_tensor * 2.0
    input = scaled_input_tensor

    if FLAGS.model == "dnn":
      feature_size = FLAGS.image_width * FLAGS.image_height * FLAGS.channels
      logits = dnn_model(input, feature_size, FLAGS.label_size)

    else:
      # Get slim models
      network_fn = nets_factory.get_network_fn(
          FLAGS.model,
          num_classes=FLAGS.label_size,
          #weight_decay=FLAGS.weight_decay,
          is_training=is_training)

      logits, end_points = network_fn(input)
      """
      if 'AuxLogits' in end_points:
        slim.losses.softmax_cross_entropy(
              end_points['AuxLogits'], labels,
              label_smoothing=FLAGS.label_smoothing, weights=0.4,
              scope='aux_loss')
      slim.losses.softmax_cross_entropy(
            logits, labels, label_smoothing=FLAGS.label_smoothing, weights=1.0)
    """

    return logits


  train_numeric_label_op = train_label_op
  validation_numeric_label_op = validation_label_op

  logits = _model(train_features_op)
  loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=train_numeric_label_op))

  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    variables_to_train = []
    if FLAGS.model_trainable_scopes == "":
      variables_to_train = tf.trainable_variables()
    else:
      scopes = [
          scope.strip() for scope in FLAGS.model_trainable_scopes.split(',')
      ]
      for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
        logging.info("Add variables to train: {}".format(variables))

    train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(
        loss, global_step=global_step, var_list=variables_to_train)

  # Define train accuracy op
  train_softmax_op = tf.nn.softmax(logits)
  train_correct_prediction = tf.equal(
      tf.argmax(train_softmax_op, 1), train_numeric_label_op)
  train_accuracy_op = tf.reduce_mean(
      tf.cast(train_correct_prediction, tf.float32))
  """
  # TODO: Compute train accuracy while is_training is False
  train_logits = _model(batch_features_op, is_training=False)
  train_softmax_op = tf.nn.softmax(train_logits)
  train_correct_prediction = tf.equal(
          tf.argmax(train_softmax_op, 1), numeric_label_op)
  train_accuracy_op = tf.reduce_mean(
          tf.cast(train_correct_prediction, tf.float32))
  """

  tf.get_variable_scope().reuse_variables()

  # Define validation accuracy op
  validation_accuracy_logits = _model(
      validation_features_op, is_training=False)
  validation_softmax_op = tf.nn.softmax(validation_accuracy_logits)
  validation_correct_prediction = tf.equal(
      tf.argmax(validation_softmax_op, 1), validation_numeric_label_op)
  validation_accuracy_op = tf.reduce_mean(
      tf.cast(validation_correct_prediction, tf.float32))

  validation_batch_number = int(
      validation_instance_number * FLAGS.validation_dataset_ratio /
      FLAGS.validation_batch_size)
  logging.info("Need to run {} batch times to compute accuracy for validation".
               format(validation_batch_number))

  # Define export model op
  model_features_placeholder = tf.placeholder(
      tf.float32, [None, None, None, FLAGS.channels])
  model_features_input = tf.image.resize_images(
      model_features_placeholder, [FLAGS.image_height, FLAGS.image_width])
  model_logits = _model(model_features_input, is_training=False)
  model_softmax_op = tf.nn.softmax(model_logits)
  model_prediction_op = tf.argmax(model_softmax_op, 1)

  model_base64_placeholder = tf.placeholder(
      shape=[None], dtype=tf.string, name="model_input_b64_images")
  model_base64_string = tf.decode_base64(model_base64_placeholder)
  # TODO: Change to tf.decode_image if supporting tf.resize_images
  model_base64_input = tf.map_fn(lambda x: tf.image.resize_images(tf.image.decode_jpeg(x, channels=FLAGS.channels), [FLAGS.image_height, FLAGS.image_width]), model_base64_string, dtype=tf.float32)
  model_base64_logits = _model(model_base64_input, is_training=False)
  model_base64_softmax_op = tf.nn.softmax(
      model_base64_logits, name="model_output_b64_softmax")
  model_base64_prediction_op = tf.argmax(
      model_base64_softmax_op, 1, name="model_output_b64_prediction")

  # Save dataset iterator to checkpoint if no shuffling, refer to https://github.com/tensorflow/tensorflow/issues/18583
  if FLAGS.train_shuffle_buffer_size <= 0:
    saveable = tf.contrib.data.make_saveable_from_iterator(train_iterator)
    tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)

  saver = tf.train.Saver()
  #tf.summary.scalar("loss", loss)
  #tf.summary.scalar("train_accuracy", train_accuracy_op)
  tf.summary.scalar("validation_accuracy", validation_accuracy_op)
  summary_op = tf.summary.merge_all()

  # Define restorer for checkpoint
  variables_to_restore = {}
  if FLAGS.restore_checkpoint_include_scopes.strip() != "":
    for scope_name in FLAGS.restore_checkpoint_include_scopes.split(","):
      for v in tf.global_variables(scope_name):
        variables_to_restore[v.op.name] = v

  if FLAGS.restore_checkpoint_exclude_scopes.strip() != "":
    for scope_name in FLAGS.restore_checkpoint_exclude_scopes.split(","):
      for v in tf.trainable_variables(scope_name):
        if v.op.name in variables_to_restore:
          variables_to_restore.pop(v.op.name)

  if len(variables_to_restore) == 0:
    restorer = tf.train.Saver()
  else:
    restorer = tf.train.Saver(var_list=variables_to_restore)

  outputs = {
      "softmax": utils.build_tensor_info(model_softmax_op),
      "prediction": utils.build_tensor_info(model_prediction_op),
  }

  outputs_base64 = {
      "softmax": utils.build_tensor_info(model_base64_softmax_op),
      "prediction": utils.build_tensor_info(model_base64_prediction_op),
  }

  model_signature = signature_def_utils.build_signature_def(
      inputs={"images": utils.build_tensor_info(model_features_placeholder)},
      outputs=outputs,
      method_name=signature_constants.PREDICT_METHOD_NAME)

  base64_model_signature = signature_def_utils.build_signature_def(
      inputs={"images": utils.build_tensor_info(model_base64_placeholder)},
      outputs=outputs_base64,
      method_name=signature_constants.PREDICT_METHOD_NAME)

  model_signature_map = {
      signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
      base64_model_signature,
      "serving_tensor":
      model_signature
  }

  def export_model(model_signature_map):
    # export PROPHET_CONFIG='{"inputSlotPaths": ["/autocv/META-INF/mnist_train.json", "/autocv/META-INF/mnist_train.json"], "outputSlotPaths": ["hdfs://172.27.128.31:8020/user/prophet/TensorFlowTraining/test_dag/15/test_node/1"]}'
    prophet_config_json = json.loads(os.environ.get("PROPHET_CONFIG", "{}"))
    if "outputSlotPaths" in prophet_config_json and len(
        prophet_config_json["outputSlotPaths"]) >= 1:
      model_path = prophet_config_json["outputSlotPaths"][0]
    else:
      model_path = FLAGS.model_path

    export_path = os.path.join(model_path, str(FLAGS.model_version))
    logging.info("Try to save the model in: {}".format(export_path))

    graph_file_name = "graph.pb"
    tf.train.write_graph(sess.graph_def, "./", graph_file_name, as_text=False)
    logging.info("Save the graph file: {}".format(graph_file_name))

    try:
      legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
      builder = saved_model_builder.SavedModelBuilder(export_path)
      builder.add_meta_graph_and_variables(
          sess, [tag_constants.SERVING],
          clear_devices=True,
          signature_def_map=model_signature_map,
          legacy_init_op=legacy_init_op)
      builder.save()
      logging.info("Success to save the model in: {}".format(export_path))
    except Exception as e:
      logging.error("Fail to export saved model, exception: {}".format(e))

  with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    sess.run(
        train_iterator.initializer,
        feed_dict={
            train_image_list_placeholder: train_image_list,
            train_label_list_placeholder: train_label_list
        })
    sess.run(
        validation_iterator.initializer,
        feed_dict={
            validation_image_list_placeholder: validation_image_list,
            validation_label_list_placeholder: validation_label_list
        })

    if FLAGS.load_checkpoint:
      logging.info("Restore session from checkpoint: {}".format(
          FLAGS.checkpoint_file_path))
      restorer.restore(sess, FLAGS.checkpoint_file_path)

    if FLAGS.mode == "train":

      try:
        start_train_time = time.time()

        while True:
          # Run training
          _, loss_value, train_accuracy_value, global_step_value = sess.run(
              [train_op, loss, train_accuracy_op, global_step])

          # Save checkpoint periodically
          if FLAGS.steps_to_save_checkpoint > 0 and global_step_value % FLAGS.steps_to_save_checkpoint == 0:
            saver.save(
                sess,
                FLAGS.save_checkpoint_path + "/checkpoint.ckpt",
                global_step=global_step_value)

          # Run validation periodically
          train_steps_total = int(FLAGS.epoch_number * train_instance_number /
                                  FLAGS.train_batch_size)
          steps_to_validate = max(
              min(FLAGS.steps_to_validate, train_steps_total), 1)
          if global_step_value % steps_to_validate == 0:
            start_validation_time = time.time()

            total_validation_accuracy_value = 0.0
            if validation_batch_number <= 0:
              validation_batch_number = 1
            for i in range(validation_batch_number):
              summary_op_value, batch_validation_accuracy_value = sess.run(
                  [summary_op, validation_accuracy_op])
              total_validation_accuracy_value += batch_validation_accuracy_value
            total_validation_accuracy_value = total_validation_accuracy_value / validation_batch_number

            validation_time = time.time() - start_validation_time

            # TODO: Display better progress when epoch_number is negative
            computed_epoch_index = int(
                global_step_value /
                (train_instance_number * 1.0 / FLAGS.train_batch_size))
            computed_progress = global_step_value / (
                train_instance_number * FLAGS.epoch_number * 1.0 / FLAGS.
                train_batch_size)
            logging.info(
                "Epoch: {}, step: {}, progress: {}%, loss: {}, train acc: {}, valid acc: {} ({}s)".
                format(computed_epoch_index, global_step_value,
                       computed_progress * 100, loss_value,
                       train_accuracy_value, total_validation_accuracy_value,
                       validation_time))


      except tf.errors.OutOfRangeError:
        end_train_time = time.time()
        logging.info("End of data, train time: {} seconds".format(
            end_train_time - start_train_time))
        export_model(model_signature_map)

    elif FLAGS.mode == "savedmodel":
      logging.info("Run without training and export the model")
      export_model(model_signature_map)

    elif FLAGS.mode == "test":
      logging.info("Run and inference with test dataset: {}".format(
          FLAGS.validation_csv_file))

      batch_number = int(
          validation_instance_number / FLAGS.validation_batch_size)
      logging.info("Need to run {} batch times".format(batch_number))

      total_accuracy = 0.0
      for i in range(batch_number):
        batch_accuracy = sess.run(validation_accuracy_op)
        total_accuracy += batch_accuracy
      total_accuracy = total_accuracy / batch_number
      logging.info("The accuracy of test dataset: {}".format(total_accuracy))


if __name__ == "__main__":
  main()
