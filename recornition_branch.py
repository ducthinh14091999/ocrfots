from re import T
import tensorflow as tf
from tensorflow import keras
# https://github.com/calmisential/TensorFlow2.0_ResNet/blob/master/models/resnet.py
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import time
from tensorflow.python.keras.engine.input_layer import Input
alphabet = " !"+'"'+"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"+'àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ\t_[]{}@\\<>%'
def CNN_LSTM(input):
    input = tf.reshape(input,[32,1,8,8,32])
    num_fil = [ 64,64, 128, 128, 256, 256]
    kernel_size = [ 3, 3, 3, 3, 3,3]
    pools = [[1,1],[1,1],[1,1],[2,1],[2,1],[2,1]]
    stride_distan = [[1,1], [1,1], [1,1], [1,1],[1,1],[1,1]]
    pads =['same','same','same','same','same','same']
    time1= time.time()
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64,(3,3),padding='same',activation=tf.keras.layers.ELU()))(input)
    time2= time.time()
    denta_time = time2-time1
    print('time to config: ',denta_time)
    for pool,pad,fil,size,stride in zip(pools,pads,num_fil,kernel_size,stride_distan):
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(fil,(size,size),stride,activation=tf.keras.layers.ELU(),padding=pad))(x)
        if pool != [1,1]:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=pool,strides=pool))(x)
    x = tf.squeeze(x,axis=1)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512,return_sequences=True)))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(alphabet)+1,activation='softmax'))(x)
    return x
if __name__=='__main__':
    input=tf.keras.layers.Input(shape=(None,8,32))
    output=CNN_LSTM(input)
    Model=tf.keras.Model(inputs=input,outputs=output)
    print(Model.summary())
