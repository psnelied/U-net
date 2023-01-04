import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.cm as cm
import random
import math
import tensorflow.keras.layers as tkl
import tensorflow_addons as tfa



# GEOMETRIC TRANSFORMATION
class ImgLandmarksProcessing(tkl.Layer):
    def __init__(self,
                 **kwargs):
        super(ImgLandmarksProcessing, self).__init__()

    def call(self, x, y=None, training=None):
        
        dest_img, dest_landmarks = self.img_landmarks_processing(training=training)

        return x, y

    def img_landmarks_processing(self, img, landmarks, **kwargs):
        return img, landmarks

class ImgLandmarksResize(ImgLandmarksProcessing):
    def __init__(self,
                 im_h,
                 im_w,
                 **kwargs):
        super(ImgLandmarksResize, self).__init__()
        self.im_h = im_h
        self.im_w = im_w

    def img_landmarks_processing(self, img, landmarks, **kwargs):
        """
        img (B, H, W, C)
        landmarks (B, 49, 2)
        """
        src_img_dim = tf.shape(img)[-3:-1]
        dest_img_dim = tf.constant([self.im_h, self.im_w])
        dest_img = tf.image.resize(img, dest_img_dim)

        src_img_dim = tf.dtypes.cast(src_img_dim[::-1],
                                     tf.float32)
        dest_img_dim = tf.dtypes.cast(dest_img_dim[::-1],
                                      tf.float32)
        dest_landmarks = ((dest_img_dim / src_img_dim)[tf.newaxis, tf.newaxis, :]
                          * landmarks)
        return dest_img, dest_landmarks

class ImgLandmarksTrainDataAugmentation(ImgLandmarksProcessing):
    def __init__(self,
                 **kwargs):
        super(ImgLandmarksTrainDataAugmentation, self).__init__(
                                                                **kwargs)

    def img_landmarks_processing(self, img, landmarks, training=None, **kwargs):
        if training:
            return self.augmentation(img=img,
                                     landmarks=landmarks)
        else:
            return img, landmarks
    
    def augmentation(self, img, landmarks, **kwargs):
        return img, landmarks

class ImgLandmarksRandomHorizontalFlip(ImgLandmarksTrainDataAugmentation):
    def __init__(self,
                 flip_mapping,
                 p_flip=0.5,
                 **kwargs):
        super(ImgLandmarksRandomHorizontalFlip, self).__init__(
                                                               **kwargs)
        self.flip_mapping = flip_mapping
        self.p_flip = p_flip

    def augmentation(self, img, landmarks):
        """
        img (B, H, W, C)
        landmarks (B, 49, 2)
        """
        img_shape = tf.shape(img)
        B, W  = img_shape[0], tf.dtypes.cast(img_shape[2],
                                             tf.float32)
        img_dim = tf.dtypes.cast(img_shape[1:-1],
                                 tf.float32)
        u_flips = tf.random.uniform(shape=(B, ),
                                    minval=0,
                                    maxval=1)
        # 1 if it flips 0 else
        # (B, )
        mask_flip = tf.dtypes.cast(u_flips - self.p_flip <= 0 ,
                                   dtype=tf.float32)
        mask_img_flip = mask_flip[:, tf.newaxis, tf.newaxis, tf.newaxis]
        mask_landmarks_flip = mask_flip[:, tf.newaxis, tf.newaxis]

        img_flip = img[:, :, ::-1, :]
        # (B, 49, 2) 
        landmarks_flip = tf.stack([W - landmarks[:, :, 0], landmarks[:, :, 1]],
                                  axis=-1) 
        landmarks_flip = tf.gather(landmarks_flip,
                                   self.flip_mapping,
                                   axis=1)

        r_img_flip = mask_img_flip * img_flip + (1 - mask_img_flip) * img
        r_landmarks_flip = mask_landmarks_flip * landmarks_flip + (1 - mask_landmarks_flip) * landmarks

        return r_img_flip, r_landmarks_flip

class ImgLandmarks68RandomHorizontalFlip(ImgLandmarksRandomHorizontalFlip):
    def __init__(self,
                 p_flip):

        flip_landmark_mapping = tf.constant(np.array([16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                                      26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35, 34, 33,
                                                      32, 31, 45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40, 54, 53, 52,
                                                      51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63, 62, 61, 60, 67, 66, 65]),
                                            dtype=tf.int64)
        super(ImgLandmarks68RandomHorizontalFlip, self).__init__(flip_mapping=flip_landmark_mapping,
                                                               p_flip=p_flip)



def create_trans_vector(start,stop,batch_size,fact=0.1):
    a = tf.constant([[fact*random.randint(start,stop),fact*random.randint(start,stop)]])
    for k in range(batch_size-1):
        translation_x = fact*random.randint(start,stop)
        translation_y = fact*random.randint(start,stop)
        t1 = [[translation_x, translation_y]]
        a = tf.concat([a, t1], 0)
    return a

def create_rot_vector(batch_size):
    min = -0.1*2*math.pi
    max = 0.1*2*math.pi
    angle = random.uniform(min, max)
    a = tf.constant([[angle]])
    for k in range(batch_size-1):
        t1 = [[random.uniform(min, max)]]
        a = tf.concat([a, t1], 0)
    return tf.squeeze(a,axis=-1) 
    


def rotate_landmarks(landmarks, angles, img_shape):
    # (2, )
    offset = tf.dtypes.cast(img_shape[::-1], tf.float32) / 2.0
    offset = offset[tf.newaxis, tf.newaxis, :]

    # (B, )
    sin_angles = tf.math.sin(angles)
    # (B, )
    cos_angles = tf.math.cos(angles)

    # (B, 1, 2, 2)
    rot_matrix = tf.stack(
            [
                tf.stack([cos_angles, sin_angles], axis=1),
                tf.stack([-sin_angles, cos_angles], axis=1),
            ],
            axis=1,
        )[:, tf.newaxis, :, :]
    # (B, 49, 2, 1)
    landmarks_centered = (landmarks - offset)[:, :, :, tf.newaxis]
    # (B, 49, 2)
    rot_landmarks_centered = tf.squeeze(tf.matmul(rot_matrix, landmarks_centered), axis=-1)
    rot_landmarks = rot_landmarks_centered + offset
    return rot_landmarks

def translate_landmarks(landmarks, translations):
    """
    landmarks (B, L, 2)
    random_translations (B, 2)
    """
    random_translations = translations[:, tf.newaxis, :]
    return landmarks + random_translations

def rotate_img(img, angles):
    """
    img (B, H, W, C)
    angles (B, )
    """
    return tfa.image.rotate(images=img, angles=angles)

def translate_img(img, translations, fill_value=0.):
    """
    img : (B, H, W, C)
    translations (B, 2)
    Warning : Coord 1 is x coordinate, coord 2 is y coordinate
    """
    img_shape = tf.shape(img)
    B, H, W = img_shape[0], img_shape[1], img_shape[2]
    range_h, range_w = tf.range(H), tf.range(W)
    range_W, range_H = tf.meshgrid(range_w, range_h)
    # (H, W, 2)
    grid = tf.stack([range_H, range_W], axis=-1)
    # (B, H, W, 2)
    translations = tf.dtypes.cast(tf.math.round(translations), tf.int32)
    translated_grid = grid[tf.newaxis, :, :, :] - translations[:, tf.newaxis, tf.newaxis, ::-1]
    # 1 if point is out of grid 0 else
    # (B, H, W)
    out_of_grid = tf.dtypes.cast(tf.math.logical_or(tf.math.logical_or(translated_grid[:, :, :, 0] < 0,
                                                                       translated_grid[:, :, :, 0] > H - 1),
                                                    tf.math.logical_or(translated_grid[:, :, :, 1] < 0,
                                                                       translated_grid[:, :, :, 1] > W - 1)),
                                 dtype=tf.int32)[:, :, :, tf.newaxis]
    # (B, H, W, 2)
    clipped_translated_grid = (1 - out_of_grid) * translated_grid
    out_of_grid = tf.dtypes.cast(out_of_grid, dtype=tf.float32)
    # (B, H, W, 2)
    translated_img = tf.gather_nd(params=img, indices=clipped_translated_grid, batch_dims=1)
    padded_translated_img = (1 - out_of_grid) * translated_img + out_of_grid * fill_value 
    return padded_translated_img


def flip_img(img,p_flip):
    """
    img (B, H, W, C)
    """
    img_shape = tf.shape(img)
    B, W  = img_shape[0], tf.dtypes.cast(img_shape[2],tf.float32)
    img_dim = tf.dtypes.cast(img_shape[1:-1],tf.float32)
    u_flips = tf.random.uniform(shape=(B, ),minval=0,maxval=1)
    # 1 if it flips 0 else
    # (B, )
    mask_flip = tf.dtypes.cast(u_flips - p_flip <= 0 ,dtype=tf.float32)
    mask_img_flip = mask_flip[:, tf.newaxis, tf.newaxis, tf.newaxis]

    img_flip = img[:, :, ::-1, :]
    # (B, 49, 2) 

    r_img_flip = mask_img_flip * img_flip + (1 - mask_img_flip) * img

    return r_img_flip
