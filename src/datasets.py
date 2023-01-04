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


def load_disfa(batchsize,disfa_path, subsample=1):
    meta_dataset = load_disfa_meta(disfa_path, subsample=subsample).cache()
    count = meta_dataset.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1)
    return (meta_dataset
            .shuffle(count)
            .map(decode_paths)
            .batch(batchsize)
            .prefetch(tf.data.AUTOTUNE))            
            

def gen_300WLP_with_landmarks(csv, batchsize):
    
    meta_dataset = tf.data.experimental.CsvDataset(csv,[str(),str()], header=True)
    
    count = meta_dataset.reduce(tf.constant(0, tf.int64),
                                lambda x, _: x + 1)
    

    loaded_dataset_meta = (meta_dataset
                               .shuffle(count)
                               .map(decode_paths, tf.data.experimental.AUTOTUNE)
                               .batch(batchsize)
                               .prefetch(tf.data.experimental.AUTOTUNE))
    return loaded_dataset_meta
    
def decode_paths(img_path,landmarks_path):
    
    img =tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    landmarks = read_tensor(landmarks_path,dtype = tf.float32)
    
    return img,landmarks


