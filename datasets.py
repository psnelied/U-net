import tensorflow as tf
import numpy as np
import math
import pandas as pd
import scipy
import scipy.io as sio
import csv
import glob
import cv2
from os.path import exists, splitext
import copy as cp
import json


def read_tensor(path, dtype):
    serialized_tensor = tf.io.read_file(path)
    return tf.io.parse_tensor(serialized_tensor, out_type=tf.float32)


def load_disfa_meta(disfa_path, subsample=1):
    dataset_types = [str(), str()]
    return (tf.data.experimental.CsvDataset(disfa_path,
                                            record_defaults=dataset_types,
                                            header=True)
            .shard(subsample, 0))

def decode_img(img_path):
    img = tf.io.decode_image(tf.io.read_file(img_path), channels=3, dtype=tf.float32)
    img.set_shape((299, 299, 3))
    return img

def decode_label(label_path):
    return read_tensor(label_path, dtype=tf.float32)

def decode_paths(img_path, label_path):
    return decode_img(img_path), decode_label(label_path)

def load_disfa(batchsize,disfa_path, subsample=1):
    meta_dataset = load_disfa_meta(disfa_path, subsample=subsample).cache()
    count = meta_dataset.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1)
    return (meta_dataset
            .shuffle(count)
            .map(decode_paths)
            .batch(batchsize)
            .prefetch(tf.data.AUTOTUNE))            
            