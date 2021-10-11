from numpy.lib.function_base import angle
import tensorflow as tf
from tensorflow import keras
# https://github.com/calmisential/TensorFlow2.0_ResNet/blob/master/models/resnet.py
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import glob

from tensorflow.python.keras.engine import input_layer
from share_model import share_model
from recornition_branch import CNN_LSTM, alphabet
from detection_branch import detection_bra
from rroi_align import Rroi_align
from utis import rois_encoder
import cv2
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
#[cho một cái thì bán nhà ra để mà ở]*F:/de_cuong/project_2/New_folder/segment_line3/5_6.jpg
def main_model():
    input = tf.keras.layers.Input(shape=[None,512,512,3])
    input_label = tf.keras.layers.Input(shape=[None,512,512,3])
    x=share_model()(input)
    detect = detection_bra()(x)
    rois= rois_encoder(detect)
    recognition = Rroi_align()(7,7,1,x,rois)
    recognition = CNN_LSTM(x)
    model = tf.keras.Model(inputs= input, outputs=[detect,recognition])
    return model
model = main_model()
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
with open('F:\project_2\New_folder/label_combine.txt',encoding='UTF-8',mode='r' ) as file:
    for line in file:
        address = line.split('*')[1].split('/')
        content = line.split('*')[0]
        address = 'F:/project_2/New_folder/'+ address[-2]+'/'+ address[-1]
        img= cv2.imread(address)
        img = cv2.resize(img,(512,512))
        img= np.array(img).reshape(1,512,512,-1)
        for i in range(len(content)):
            # print(Y_train[i])
            # tam=np.ones(200,dtype=int)*len(alphabet)
            tam=np.zeros(200,dtype=int)
            for j,char in enumerate (Y_train[i]):
              # print(char)
              num=char_to_int[char]
              tam[j]=num
            integer_encoded.append(tam)
        integer_encoded=np.array(integer_encoded)
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model(img,input_label)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_train, logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %s samples" % ((step + 1) * batch_size))
