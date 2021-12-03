from typing import TextIO
from numpy.core.numeric import Inf
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
from utis import  mapscore_geo, detect
from loss_fun import loss_fn
from main_model import main_model,roi_gen
import cv2
max_word_sample=32
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
#1-736.jpg*[Lão ta chẳng bị thương gì sất, lão bị xung huyết đay mà. Còn có the làm gì được nữa đay, tôi da canhbaos]*21*8*620*6*623*23*19*22

model = main_model()
print(model.summary(line_length=140, positions=None, print_fn=None))
if os.path.exists('model_paragram.h5'):
    model.load_weights('model_paragram.h5')
optimizer = keras.optimizers.Adam(learning_rate=1e-4)
step = 0
best_loss=Inf
count =0
while(1):

    img_file= glob.glob('F:/project_2/New_folder/data/downloads/*.jpg')+glob.glob('F:/project_2/New_folder/data/downloads/*.png')
    for Img_name in img_file:
        contents=[]
        text_regions=[]
        txt= Img_name[:-3]+'txt'
        file =  open(txt,encoding='UTF-8',mode='r' )
        para = file.readlines()
        last_name= para[0].split('*')[0]
        name= last_name
        for line in para:
                content = line.split('*')[0]
                text_region = line.split('*')[1:]
                text_region[-1]=text_region[-1][:-1]
                text_region = np.array(text_region).astype(int)
                text_region = (text_region/max(text_region)*127).astype(int)
                text_regions.append(text_region)
                tam=np.zeros(8,dtype=int)
                # encoded_label= np.zeros([200,len(alphabet)])
                for j,char in enumerate (content[:8]):
                # print(char)
                    num=char_to_int[char]
                    tam[j]=num
                    contents.append(tam)
        address = Img_name    
        img = cv2.imread(address)
        img = cv2.resize(img,(512,512))
        img = np.array(img).reshape(1,512,512,-1)
        roi_label= roi_gen(text_regions)
        text_regions= np.array(text_regions).reshape(-1,8)
        input_label = mapscore_geo(text_regions)
        contents=np.array(contents[:max_word_sample])    
        #training config
        # for i in range(5):
        #     plt.imshow(input_label[1][i,:,:,0])
        #     plt.show()
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded6
            # on the GradientTape.
            roi_label = roi_label.reshape([1,max_word_sample,6])
            logits = model([img,roi_label])  # Logits for this minibatch
            # for i in range(5):
            #     plt.imshow((logits[0][0][0,:,:,i]).numpy())
            #     plt.show()
            # Compute the loss value for this minibatch.
            loss_value,reg_loss,dectect_loss = loss_fn([input_label,contents.reshape([-1,8])], logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient([dectect_loss,reg_loss], model.trainable_weights)
        # for id,i in enumerate(model.trainable_weights):
        #     if (grads[id]==0).numpy().all():
        #         print(id,i.name,'all zero')
        #     else:
        #         print(id,i.name,'have gradients')
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        if step % 20 == 0:
            print(
                "Training loss (for one batch) at step {}  have loss {} regconition loss {} detection loss {}".format(step, float(loss_value),reg_loss,dectect_loss)
            )
            print("Seen so far: %s samples" % ((step + 1) * 1))
        step+=1
        if loss_value< best_loss:
            best_loss=loss_value
            model.save_weights('model_paragram.h5')
            count=0
        else:
            count+=1
            if count == 200:
                optimizer.learning_rate = optimizer.learning_rate* 0.4
                print('new learning rate: {}'.format(optimizer.learning_rate.numpy()))
                count=0
