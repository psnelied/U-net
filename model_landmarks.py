#!/usr/bin/env python
# coding: utf-8

#test
# In[ ]:


import tensorflow as tf
import matplotlib.cm as cm
import numpy as np
import math
import pandas as pd
import os
import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl


class DictEfficientNetB0(tkm.Model):
    def __init__(self, weights=None, pooling=None):
        super(DictEfficientNetB0, self).__init__()
        self.efficientnetb0 = tf.keras.applications.EfficientNetB0(include_top=False,
                                                                   weights=weights,
                                                                   pooling=pooling)
        
    
    def build(self, input_shape):
        input_layer = tkl.Input(shape=input_shape[1:])
        output = self.efficientnetb0(input_layer)
        self.efficientb0_intermediary = tkm.Model(inputs=self.efficientnetb0.input,
                                                  outputs={"net1": self.efficientnetb0.get_layer('input_1').output,
                                                           "net2": self.efficientnetb0.get_layer('block2a_expand_activation').output,
                                                           "net3": self.efficientnetb0.get_layer('block3a_expand_activation').output,
                                                           "net4": self.efficientnetb0.get_layer('block4a_expand_activation').output,
                                                           "net5": self.efficientnetb0.get_layer('block6a_expand_activation').output})
 
    def call(self, x, training=None):
        return self.efficientb0_intermediary(x, training=training)

class DecoderBlock(tkm.Model):
    
    def __init__(self, num_filters):
        super(DecoderBlock, self).__init__()
        self.conv2d_tr = tkl.Conv2DTranspose(filters=num_filters,
                                             kernel_size=(2, 2),
                                             strides=2,
                                             padding='same')
        self.conv2d_1 = tkl.Conv2D(filters=num_filters,
                                   kernel_size=(3, 3),
                                   padding='same')
        self.bn_1 = tkl.BatchNormalization()
        self.activation_1 = tkl.Activation('relu')
        self.conv2d_2 = tkl.Conv2D(filters=num_filters,
                                   kernel_size=(3, 3),
                                   padding='same')
        self.bn_2 = tkl.BatchNormalization()
        self.activation_2 = tkl.Activation('relu')
    def call(self, x, skip, training=None, **kwargs):
        x = self.conv2d_tr(x)
        x = tf.concat([x, skip], axis=-1)
        
        x = self.conv2d_1(x)
        x = self.bn_1(x, training=training)
        x = self.activation_1(x)
        x = self.conv2d_2(x)
        x = self.bn_2(x, training=training)
        x = self.activation_2(x)
        return x

class U_net (tkm.Model):
    
    def __init__(self, weights=None, pooling=None):
        super(U_net, self).__init__()
        self.dict_efficientnetb0 = DictEfficientNetB0(weights=weights, pooling=pooling)
        self.decoder_block_1 = DecoderBlock(num_filters=512)
        self.decoder_block_2 = DecoderBlock(num_filters=256)
        self.decoder_block_3 = DecoderBlock(num_filters=128)
        self.decoder_block_4 = DecoderBlock(num_filters=64)
        self.final_conv1D = tkl.Conv2D(filters=68,
                                       activation='linear',
                                       kernel_size=1)
    def call(self, x, training=None, **kwargs):
        x = 255 * x
        output_dict = dict()
        nets = self.dict_efficientnetb0(x, training=training)
        out_1 = self.decoder_block_1(x=nets['net5'],
                                     skip=nets['net4'],
                                     training=training)
        out_2 = self.decoder_block_2(x=out_1,
                                     skip=nets['net3'],
                                     training=training)
        out_3 = self.decoder_block_3(x=out_2,
                                     skip=nets['net2'],
                                     training=training)
        out_4 = self.decoder_block_4(x=out_3,
                                     skip=nets['net1'],
                                     training=training)
        # (B, h, w, 68)
        out_5 = self.final_conv1D(out_4)
        heatmap = tf.keras.activations.softmax(out_5, axis=(1, 2))
        heatmap_shape = tf.shape(heatmap)
        h, w = heatmap_shape[1], heatmap_shape[2]
        # (h, w, 2)
        grid = tf.stack(tf.meshgrid(tf.range(h, dtype=tf.float32),
                                    tf.range(w, dtype=tf.float32)), axis=-1)
        # (B, 68, 2)
        landmarks = tf.math.reduce_sum(heatmap[:, :, :, :, tf.newaxis] *
                                       grid[tf.newaxis, :, :, tf.newaxis, :], 
                                       axis=(1, 2))
        landmarks = tf.stack([landmarks[:, :, 0], tf.dtypes.cast(h, tf.float32) - landmarks[:, :, 1]], axis=-1)
        output_dict['heatmaps'] = heatmap
        output_dict['landmarks'] = landmarks
        return output_dict
        

