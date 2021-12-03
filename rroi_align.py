from numpy.lib.function_base import angle
import tensorflow as tf
from tensorflow import keras
# https://github.com/calmisential/TensorFlow2.0_ResNet/blob/master/models/resnet.py
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

from tensorflow.python.ops.gen_array_ops import zeros_like
# @tf.function
class Rroi_align(tf.keras.Model):
    def __init__(self):
        super(Rroi_align,self).__init__()
        self.spatial_scale=1
    def call(self, pooled_height,pooled_width, spatial_scale, features, rois):
        channel=features.shape[3]
        width = features.shape[1]
        height = features.shape[2]
        numroi = 32
        x_axis= tf.linspace(0.0,tf.cast(pooled_height-1,tf.float32), pooled_height)
        y_axis = tf.linspace(0.0, tf.cast(pooled_width-1,tf.float32), pooled_width)
        x_t, y_t = tf.meshgrid(x_axis, y_axis) #position follow x axis, y axis
        x_t = tf.tile(tf.expand_dims(tf.expand_dims(x_t,-1),0),[numroi,1,1,channel])
        y_t = tf.tile(tf.expand_dims(tf.expand_dims(y_t,-1),0),[numroi,1,1,channel])
        # M_matrix_0 = tf.stack(tf.ones([pooled_height, pooled_width],tf.int32), tf.range(channel, delta=1, dtype=tf.int32, name='range'))
        M_matrix_1 = tf.expand_dims(tf.ones([pooled_height, pooled_width,channel],tf.float32)*tf.cast(rois[0,0,1],tf.float32),0) # rois with left
        M_matrix_2 = tf.expand_dims(tf.ones([pooled_height, pooled_width,channel],tf.float32)*tf.cast(rois[0,0,2],tf.float32),0) # rois with right
        M_matrix_3 = tf.expand_dims(tf.ones([pooled_height, pooled_width,channel],tf.float32)*tf.cast(rois[0,0,3],tf.float32),0) # rois with top
        M_matrix_4 = tf.expand_dims(tf.ones([pooled_height, pooled_width,channel],tf.float32)*tf.cast(rois[0,0,4],tf.float32),0) # róis with bottom
        M_matrix_5 = tf.expand_dims(tf.ones([pooled_height, pooled_width,channel],tf.float32)*180.0*3.1415926535*tf.cast(rois[0,0,5],tf.float32),0) # rois with angle
        for i in range(31):
            M_matrix_1 = tf.concat([M_matrix_1,tf.expand_dims(tf.ones([pooled_height, pooled_width,channel],tf.float32)*tf.cast(rois[0,i,1],tf.float32),0)],0)
            M_matrix_2 = tf.concat([M_matrix_2,tf.expand_dims(tf.ones([pooled_height, pooled_width,channel],tf.float32)*tf.cast(rois[0,i,2],tf.float32),0)],0)
            M_matrix_3 = tf.concat([M_matrix_3,tf.expand_dims(tf.ones([pooled_height, pooled_width,channel],tf.float32)*tf.cast(rois[0,i,3],tf.float32),0)],0)
            M_matrix_4 = tf.concat([M_matrix_4,tf.expand_dims(tf.ones([pooled_height, pooled_width,channel],tf.float32)*tf.cast(rois[0,i,4],tf.float32),0)],0)
            M_matrix_5 = tf.concat([M_matrix_5,tf.expand_dims(tf.ones([pooled_height, pooled_width,channel],tf.float32)*tf.cast(rois[0,i,5],tf.float32)*180.0*3.1415926535,0)],0)
        roi_pooled_width = tf.math.truediv(M_matrix_4, M_matrix_3, name=None)*pooled_width
        dx =tf.cast(-roi_pooled_width/2,tf.float32)
        dy =tf.cast(-pooled_height,tf.float32)/2.0
        Sx = tf.cast(tf.math.truediv(tf.cast(M_matrix_4,dtype=tf.float32), roi_pooled_width, name=None)*spatial_scale,tf.float32)
        Sy = M_matrix_3/tf.cast(pooled_height*spatial_scale,tf.float32)
        Alpha = tf.math.cos(M_matrix_5)
        Beta = tf.math.sin(M_matrix_5)
        Dx = M_matrix_1*spatial_scale
        Dy = M_matrix_2*spatial_scale
        M_0_0 = tf.math.multiply(Alpha,tf.cast(Sx,tf.float32))
        M_0_1 =  tf.math.multiply(Beta,tf.cast(Sy,tf.float32))
        M_0_2 = M_0_0*tf.cast(dx,tf.float32) + M_0_1*tf.cast(dy,tf.float32) + tf.cast(Dx,tf.float32)
        M_1_0 = -tf.math.multiply(Beta,Sx)
        M_1_1 = tf.math.multiply(Alpha,Sy)
        M_1_2 = M_1_0*dx + M_1_1*dy + tf.cast(Dy,tf.float32)
        P0 = M_0_0*x_t+M_0_1*y_t + M_0_2
        P1 = M_1_0*x_t+M_1_1*y_t + M_1_2
        P2 = M_0_0*x_t+M_0_1*(y_t+1) + M_0_2
        P3 = M_1_0*x_t+M_1_1*(y_t+1) + M_1_2
        P4 = M_0_0*(x_t+1)+M_0_1*y_t + M_0_2
        P5 = M_1_0*(x_t+1)+M_1_1*y_t + M_1_2
        P6 = M_0_0*(x_t+1)+M_0_1*(y_t+1) + M_0_2
        P7 = M_1_0*(x_t+1)+M_1_1*(y_t+1) + M_1_2
        leftMost = tf.math.maximum(tf.math.round(tf.math.minimum(tf.math.minimum(P0,P2),tf.math.minimum(P4,P6))),0)
        rightMost = tf.math.minimum(tf.math.round(tf.math.maximum(tf.math.maximum(P0,P2),tf.math.maximum(P4,P6))),width-1)
        topMost = tf.math.maximum(tf.math.round(tf.math.minimum(tf.math.minimum(P1,P3),tf.math.minimum(P5,P7))),0)
        bottomMost = tf.math.minimum(tf.math.round(tf.math.maximum(tf.math.maximum(P1,P3),tf.math.maximum(P5,P7))),height-1)
        bin_cx = (leftMost + rightMost)/2
        bin_cy = (topMost + bottomMost)/2
        bin_l = tf.math.floor(bin_cx) #posi of this point in new map in x axis
        bin_r = tf.math.ceil(bin_cx) #posi of this point in new map in x axis
        bin_t = tf.math.floor(bin_cy) #posi of this point in new map in y axis
        bin_b = tf.math.ceil(bin_cy) #posi of this point in new map in y axis
        rx = bin_cx - tf.math.floor(bin_cx)
        ry = bin_cy - tf.math.floor(bin_cy)
        wlt = (1-rx)*(1-ry)
        wrt = rx*(1-ry)
        wrb = rx*ry
        wlb = (1-rx)* ry
        con_idx_x = tf.floor(bin_cx)
        con_idx_y = tf.floor(bin_cy)
        # lt_value = features[0,tf.cast(bin_t,tf.int32),tf.cast(bin_l,tf.int32),0]
        lt_value = tf.cast(tf.expand_dims(tf.gather_nd(features[0],tf.cast(tf.stack([tf.zeros_like(bin_t[0]),bin_l[0],bin_t[0]],axis=3),tf.int32)),0),tf.int32)
        rt_value = tf.cast(tf.expand_dims(tf.gather_nd(features[0],tf.cast(tf.stack([tf.zeros_like(bin_t[0]),bin_r[0],bin_t[0]],axis=3),tf.int32)),0),tf.int32)
        lb_value = tf.cast(tf.expand_dims(tf.gather_nd(features[0],tf.cast(tf.stack([tf.zeros_like(bin_r[0]),bin_l[0],bin_b[0]],axis=3),tf.int32)),0),tf.int32)
        rb_value = tf.cast(tf.expand_dims(tf.gather_nd(features[0],tf.cast(tf.stack([tf.zeros_like(bin_r[0]),bin_r[0],bin_b[0]],axis=3),tf.int32)),0),tf.int32)

        for i in range(1,32):
            lt_value_ = tf.cast(tf.gather_nd(features[0],tf.expand_dims(tf.cast(tf.stack([tf.zeros_like(bin_t[i]),bin_l[i],bin_t[i]],axis=3),tf.int32),0)),tf.int32)
            rt_value_ = tf.cast(tf.gather_nd(features[0],tf.expand_dims(tf.cast(tf.stack([tf.zeros_like(bin_t[i]),bin_r[i],bin_t[i]],axis=3),tf.int32),0)),tf.int32)
            lb_value_ = tf.cast(tf.gather_nd(features[0],tf.expand_dims(tf.cast(tf.stack([tf.zeros_like(bin_r[i]),bin_l[i],bin_b[i]],axis=3),tf.int32),0)),tf.int32)
            rb_value_ = tf.cast(tf.gather_nd(features[0],tf.expand_dims(tf.cast(tf.stack([tf.zeros_like(bin_r[i]),bin_r[i],bin_b[i]],axis=3),tf.int32),0)),tf.int32)
            lt_value = tf.concat([lt_value,lt_value_],0)
            rt_value = tf.concat([rt_value,rt_value_],0)
            lb_value = tf.concat([lb_value,lb_value_],0)
            rb_value = tf.concat([rb_value,rb_value_],0)
        lt_value =tf.cast(lt_value,tf.float32)
        rt_value =tf.cast(rt_value,tf.float32)
        lb_value =tf.cast(lb_value,tf.float32)
        rb_value =tf.cast(rb_value,tf.float32)
        # rt_value = features[0,tf.cast(bin_t,tf.int32),tf.cast(bin_r,tf.int32),0]
        # lb_value = features[0,tf.cast(bin_b,tf.int32),tf.cast(bin_l,tf.int32),0]
        # rb_value = features[0,tf.cast(bin_b,tf.int32),tf.cast(bin_r,tf.int32),0]
        output = lt_value * wlt + rt_value * wrt + rb_value * wrb + lb_value * wlb
        return output
if __name__=="__main__":
    feature = np.random.random((1,128,128,32))
    rois=np.array([[0,1,3,4,6,0.2],[1,2,4,3,8,0.3],[2,3,6,3,8,0.5]]).reshape(1,3,6)
    x= Rroi_align()(8,8, 1, feature, rois)
    print(x[0].shape,x[1].shape,x[2].shape)