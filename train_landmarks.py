import time
import tensorflow as tf
import numpy as np
import math
import pandas as pd
import csv
import glob
import os
import cv2
from os.path import exists, splitext
import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl
from augmentation_landmarks import ImgLandmarks68RandomHorizontalFlip, create_trans_vector, create_rot_vector, rotate_landmarks, translate_landmarks, rotate_img, translate_img
from losses_landmarks import heatmap_loss, landmark_loss, wing_loss, heatmaps_from_points
from test_landmarks import test_nme 

def train_and_checkpoints(epochs, train_dataset,valid_dataset,test_dataset, li_loss_totale, li_loss_valid,nme, model, batch_size , lr=5e-4 , optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4) ):    

    
    for epoch in range (epochs):
        
        
        li_tr = []
        li_vl = []



        print("\nStart of epoch %d" % (epoch,))

        start_time = time.time()

        
        
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_ld_train) in enumerate(train_dataset):
            
            entry_shape = tf.shape(x_batch_train)[2].numpy()
            
            item = ImgLandmarks68RandomHorizontalFlip(p_flip = 0.5)
            x_batch_train, y_ld_train = item.augmentation(x_batch_train, y_ld_train)

            x_batch_train = tf.image.resize(x_batch_train, [224,224] )
          #  x_batch_train = x_batch_train/255
            ld_coeff = 224/entry_shape
            y_ld_train = ld_coeff*y_ld_train
            #if BGR and not RGB
            #x_batch_train = tf.reverse(x_batch_train, axis=[-1])
            
            angles = create_rot_vector(tf.shape(x_batch_train)[0])      
            translations = create_trans_vector(1,224,tf.shape(x_batch_train)[0])

            x_batch_train = rotate_img(x_batch_train,angles)
            y_ld_train = rotate_landmarks(y_ld_train,angles,[224,224])

            x_batch_train = translate_img(x_batch_train,translations)
            y_ld_train = translate_landmarks(y_ld_train, translations)


            with tf.GradientTape() as tape:


                sortie = model(x_batch_train, training=True)
                

                #### heatmap loss ####

                img_shape = tf.shape(x_batch_train)[1:3]
                gt_heatmaps = heatmaps_from_points(sortie['landmarks'],img_shape,5)
                gt_heatmaps = tf.transpose(gt_heatmaps, [0,3,2,1])
                loss_heatmap = heatmap_loss(sortie['heatmaps'],gt_heatmaps)

                #### landmark loss ####
                
                loss_landmark = landmark_loss(y_ld_train,sortie['landmarks'])
                
                #### Wing Loss ####
                
                wg_loss = wing_loss(y_ld_train,sortie['landmarks'])
                
                #### Loss totale ####

                loss = 0.01*loss_heatmap + loss_landmark + 0.001*wg_loss                     

                li_tr.append(loss)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(
                (grad, var) 
                for (grad, var) in zip(grads, model.trainable_variables) 
                if grad is not None
            )
            
            
            # Log every 200 batches.
            if step % 200 == 0:
                
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss))
                )
                print("Seen so far: %d samples" % ((step + 1) * batch_size))
            

        print('===========================')
        print("Epoch {} done".format(epoch))
        print("Time taken: %.2fs" % (time.time() - start_time))
        print('===========================')
        print('STARTING VALIDATION')

        li_loss_totale.append(np.mean(li_tr))


        ## validation at each epoch ##
        
        for step , (x_batch_valid, y_ld_valid) in enumerate(valid_dataset):
            
            entry_shape = tf.shape(x_batch_valid)[2].numpy()
            item = ImgLandmarks68RandomHorizontalFlip(p_flip = 0.5)
            x_batch_valid, y_ld_valid = item.augmentation(x_batch_valid, y_ld_valid)
            x_batch_valid = tf.image.resize(x_batch_valid, [224,224] )
            x_batch_valid = tf.reverse(x_batch_valid, axis=[-1]) 
            ld_coeff = 224/entry_shape
            y_ld_valid = ld_coeff*y_ld_valid
            sortie = model(x_batch_valid, training=False)

            #### heatmap loss ####

            img_shape = tf.shape(x_batch_valid)[1:3]
            gt_heatmaps = heatmaps_from_points(sortie['landmarks'],img_shape,10)
            gt_heatmaps = tf.transpose(gt_heatmaps, [0,3,2,1])
            loss_heatmap = heatmap_loss(sortie['heatmaps'],gt_heatmaps)

            #### landmark loss ####

            loss_landmark = landmark_loss(y_ld_valid,sortie['landmarks'])

            #### Wing 
            wg_loss = wing_loss(y_ld_valid,sortie['landmarks'])

            #### Loss totale ####

            loss = 0.01*loss_heatmap + loss_landmark + 0.001*wg_loss
            
            li_vl.append(loss)

    
                

        print('END VALIDATION')
 
        li_loss_valid.append(np.mean(li_vl))
        
        print('Start NME')
        nme_value = test_nme(test_dataset, model, batch_size)
        nme.append(nme_value)
        print('NME DONE')
        
        ## save the weights for this epoch 
        model.save_weights('results_elie/landmarks_2/full/weights_checkpoint{}.h5'.format(epoch))

                        

