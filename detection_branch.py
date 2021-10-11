from numpy.lib.function_base import angle
import tensorflow as tf
from tensorflow import keras
# https://github.com/calmisential/TensorFlow2.0_ResNet/blob/master/models/resnet.py
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

class detection_bra(tf.keras.Model):
    def __init__(self):
        super(detection_bra,self).__init__()
        self.geoconv=tf.keras.layers.Conv2D(4,(1,1),padding='same',activation='sigmoid')
        self.scoreconv=tf.keras.layers.Conv2D(1,(1,1),padding='same',activation='sigmoid')
        self.angleconv=tf.keras.layers.Conv2D(1,(1,1),padding='same',activation='sigmoid')
    def call(self,input):
        geo_map = self.geoconv(input)
        score_map = self.scoreconv(input)
        angle_map = self.angleconv(input)
        geo_result = tf.keras.layers.concatenate([geo_map,angle_map])
        return geo_result,score_map
if __name__ == '__main__':
    input=tf.keras.layers.Input([32,128,128])
    model=detection_bra()(input)
    Model=tf.keras.Model(inputs=input,outputs=model)
    print(Model.summary())        

