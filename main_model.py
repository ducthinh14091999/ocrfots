import tensorflow as tf
from tensorflow.python.keras.engine import input_layer
from share_model import share_model
from recornition_branch import CNN_LSTM, alphabet
from detection_branch import detection_bra
from rroi_align import Rroi_align
from utis import  mapscore_geo, detect
import cv2
import numpy as np
max_word_sample=32
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# @tf.function
def main_model(training=True):
    input = tf.keras.layers.Input(shape=[512,512,3],batch_size=1)
    x= share_model()(input)
    detected = detection_bra()(x)
    # rois= detect(detected[1],detected[0])
    if training==True:
        input_label = tf.keras.layers.Input(shape=[max_word_sample,6],batch_size=1)
        recognition = Rroi_align()(8,8,1,x,input_label)
    else:
        bounding_extract = detect(detected[0],detected[1])
        recognition = Rroi_align()(8,8,1,x,bounding_extract)
    recognition = CNN_LSTM(recognition)
    if training == True:
        model = tf.keras.Model(inputs= [input,input_label], outputs=[detected,recognition])
    else:
        model = tf.keras.Model(inputs=input,outputs=[detected,recognition])
    return model
def roi_gen(posision):
    out= np.zeros([max_word_sample,6])
    for id,posi in enumerate(posision[:max_word_sample]):
        xmin = min(posi[::2])
        xmax = max(posi[::2])
        ymax = max(posi[1::2])
        ymin = min(posi[1::2])
        # theta = 
        out[id,:] = [id,xmin,xmax,ymax,ymin,0]
    return out