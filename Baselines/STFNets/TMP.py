import numpy as np
import tensorflow as tf


raw_dataset = tf.data.TFRecordDataset('train.tfrecord')
for raw_record in raw_dataset.take(1000):
    a = raw_record.numpy()
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

example.features.feature['label']
example.features.feature['example']

with open("example_me.txt", "w") as external_file:
    print(example, file=external_file)
    external_file.close()