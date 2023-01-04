import tensorflow as tf
import numpy as np
import math
import pandas as pd
import os


def heatmap_loss(hm1, hm2):
    hm2 = tf.keras.activations.softmax(hm2, axis=(1, 2))
    return tf.reduce_mean(tf.abs(hm1-hm2))

def wing_loss(pred_landmarks, gt_landmarks, w = 10 , epsilon = 2):
    epsilon = tf.dtypes.cast(epsilon, dtype=tf.float32)    
    w = tf.dtypes.cast(w, dtype=tf.float32)

    x = tf.abs(pred_landmarks - gt_landmarks)
        
    c = w - w*tf.math.log(1.0 +w/epsilon)
    
    #mask = tf.where(tf.greater(w,x))
    
    losses = tf.where(
        tf.greater(w, x),
        w * tf.math.log(1.0 + x/epsilon),
        x - c
    )
    loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]), axis=0)
    return loss

def landmark_loss(landmarks1, landmarks2):
    return tf.reduce_mean(tf.abs(landmarks1 - landmarks2))

def gaussian_kernel(x, y, sigma):
    """
    x: (..., 2)
    y: (..., 2)
    """
    norm_xy = tf.norm(x - y, axis=-1)
    return tf.math.exp(-tf.math.pow(norm_xy / sigma, 2))


def laplacian_kernel(sigma):

    def fun(x,y):
        l1_norm = tf.math.reduce_sum(tf.math.abs(x-y),axis=-1)
        return (1/(2*sigma))*tf.math.exp(-l1_norm/sigma)

    return fun


def heatmaps_from_points(points, img_shape, sigma):
    """
    landmarks: (..., 2).
    img_shape: (h, w).
    sigma:
    """
    
    h, w = img_shape[0], img_shape[1]
    H, W = tf.range(h, dtype=tf.float32), tf.range(w, dtype=tf.float32)
    # (..., )
    n_points = tf.shape(points)[:-1]
    # (H, W, 2)
    coord_grid = tf.stack(tf.meshgrid(H, W), axis=-1)
    # (ones_like(...), H, W, 2)
    coord_grid = tf.reshape(coord_grid,
                            tf.concat([tf.ones_like(n_points),
                                       tf.shape(coord_grid)],
                                      axis=0))
    # (..., 1, 1, 2)
    points = tf.reshape(points, tf.concat([n_points, [1, 1, 2]], axis=0))
    # (B, h, w, L)_
    heatmaps = gaussian_kernel(coord_grid, points,sigma)
    return heatmaps

