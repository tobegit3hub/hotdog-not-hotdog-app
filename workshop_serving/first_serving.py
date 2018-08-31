#!/usr/bin/env python

import logging
from flask import Flask
import tensorflow as tf


logger = logging.getLogger("first_serving")
logger.setLevel(logging.DEBUG)

app = Flask(__name__)


@app.route("/")
def hello():
  return "Hello World!"


@app.route("/inference")
def inference():

  try:
    model_file_path = "./model/1"

    # Load model
    sess = tf.Session(graph=tf.Graph())
    logger.info("Try to load the model in: {}".format(model_file_path))
    meta_graph = tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING], model_file_path)
    logger.info("Succeed to load model in: {}".format(model_file_path))

    # Run inference
    output_tensor_names = ['Identity:0', 'ArgMax_2:0', 'Softmax_2:0']
    feed_dict_map = {'Placeholder:0': [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]], 'Placeholder_1:0': [[1.0], [2.0]]}
    result_ndarrays = sess.run(output_tensor_names, feed_dict=feed_dict_map)

    # Return result
    return str(result_ndarrays)

  except Exception as e:
    logger.info("Fail to load model and catch exception: {}".format(e))
    return str(e)


def main():
  host = "0.0.0.0"
  port = 8500
  app.run(host=host, port=port)


if __name__ == "__main__":
  main()
