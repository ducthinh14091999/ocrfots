from numpy.core.defchararray import endswith
from numpy.lib.function_base import angle
import tensorflow as tf
from tensorflow import keras
# https://github.com/calmisential/TensorFlow2.0_ResNet/blob/master/models/resnet.py
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from utis import detect
from main_model import main_model,int_to_char
import glob
import cv2
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
model= main_model(training=False)
path='F:/project_2/New_folder/data/downloads/'
img_list = glob.glob(path+'*.jpg')
for img_address in img_list:
    img= cv2.imread(img_address)
    img = cv2.resize(img,(512,512))
    img = np.array(img).reshape(1,512,512,3)
    predict= model.predict(img)
    sequence_length= [32,32,32,32,32,32,32,32]
    word_list,_ = tf.nn.ctc_beam_search_decoder(tf.reshape(tf.cast(predict[1],tf.float32),[32,8,-1]), sequence_length, beam_width=100, top_paths=1)
    word=tf.compat.v1.sparse.to_dense(word_list[0])[0] 
    for character in word:
        charr= int_to_char(character)
        print(charr,endswith='')
    print('')
    plt.imshow(img)
    plt.show()