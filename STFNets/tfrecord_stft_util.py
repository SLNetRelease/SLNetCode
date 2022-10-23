import os

import tensorflow as tf
import numpy as np


# FEATURE_DIM = 60
# WIDE = 512
# OUT_DIM = 6

def csv_to_example(fname):
    text = np.loadtxt(fname, delimiter=',')
    features = text[:WIDE*FEATURE_DIM]
    label = text[WIDE*FEATURE_DIM:]

    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=label)),
        'example': tf.train.Feature(float_list=tf.train.FloatList(value=features))
    }))

    return example


def read_and_decode(tfrec_path, wide, feature_dim, out_dim):
    filename_queue = tf.compat.v1.train.string_input_producer([tfrec_path])
    reader = tf.compat.v1.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.io.parse_single_example(serialized=serialized_example,
                                       features={
                                           'label': tf.io.FixedLenFeature([out_dim], tf.float32),
                                           'example': tf.io.FixedLenFeature([wide*feature_dim], tf.float32),
                                       })
    return features['example'], features['label']


def input_pipeline_har(tfrec_path, batch_size, wide, feature_dim, out_dim, shuffle_sample=False, num_epochs=None):
    example, label = read_and_decode(tfrec_path, wide, feature_dim, out_dim)
    example = tf.expand_dims(example, 0)
    example = tf.reshape(example, shape=(wide, feature_dim))
    min_after_dequeue = 1000  # int(0.4*len(csvFileList)) #1000
    capacity = min_after_dequeue + 3 * batch_size
    if shuffle_sample:
        example_batch, label_batch = tf.compat.v1.train.shuffle_batch(
            [example, label], batch_size=batch_size, num_threads=16, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
    else:
        example_batch, label_batch = tf.compat.v1.train.batch(
            [example, label], batch_size=batch_size, num_threads=16)

    return example_batch, label_batch

