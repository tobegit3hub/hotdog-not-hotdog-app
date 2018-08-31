#!/usr/bin/env python

import os
import random
import pandas as pd
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def main():

  base_data_dir = "./hotdog_Pics"

  filename_label_items = []

  for i in range(2):
    if i == 0:
      folder_name = "Hotdog"
    elif i == 1:
      folder_name = "Not Hotdog"

    image_file_dir = os.path.join(base_data_dir, folder_name)

    for root, dirs, files in os.walk(image_file_dir):
      for filename in files:
        filename_label_item = [
            os.path.join(base_data_dir, folder_name, filename), i
        ]
        #filename_label_item = [os.path.join(base_data_dir, folder_name, filename), folder_name]
        filename_label_items.append(filename_label_item)

  random.shuffle(filename_label_items)

  # 750 in total
  train_csv_filename = "./generated_hotdog_train.csv"
  train_filename_label_items = filename_label_items[0:601]
  pd.DataFrame(train_filename_label_items).to_csv(
      train_csv_filename, header=False, sep=',', index=False)
  logging.info("Generate csv file to: {}".format(train_csv_filename))

  test_csv_filename = "./generated_hotdog_test.csv"
  test_filename_label_items = filename_label_items[601:]
  pd.DataFrame(test_filename_label_items).to_csv(
      test_csv_filename, header=False, sep=',', index=False)
  logging.info("Generate csv file to: {}".format(test_csv_filename))


if __name__ == "__main__":
  main()
