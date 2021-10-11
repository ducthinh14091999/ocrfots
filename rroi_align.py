from numpy.lib.function_base import angle
import tensorflow as tf
from tensorflow import keras
# https://github.com/calmisential/TensorFlow2.0_ResNet/blob/master/models/resnet.py
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

class Rroi_align(tf.keras.Model):
    def __init__(self):
        super (Rroi_align,self).__init_()
        self.spatial_scale=1
    def call(self, pooled_height,pooled_width, spatial_scale, features, rois):
        channel=features.shape[3]
        width = features.shape[1]
        height = features.shape[2]
        numroi = rois.shape[0]
        x_axis= tf.linspace(0, pooled_height, pooled_height)
        y_axis = tf.linspace(0, pooled_width, pooled_width)
        x_t, y_t = tf.meshgrid(x_axis, y_axis)
        xy=tf.stack([x_t,y_t])
        output=tf.tile(tf.zeros([pooled_height, pooled_width]), tf.ones([numroi,channel]))
        M_matrix_0 = tf.tile(tf.ones([pooled_height, pooled_width]), tf.range(channel, delta=1, dtype=None, name='range'))
        M_matrix_1 = tf.tile(tf.ones([pooled_height, pooled_width]), rois[:,1])
        M_matrix_2 = tf.tile(tf.ones([pooled_height, pooled_width]), rois[:,2])
        M_matrix_3 = tf.tile(tf.ones([pooled_height, pooled_width]), rois[:,3])
        M_matrix_4 = tf.tile(tf.ones([pooled_height, pooled_width]), rois[:,4])
        M_matrix_5 = tf.tile(tf.ones([pooled_height, pooled_width]), rois[:,5]*180.0*3.1415926535)
        roi_pooled_width = tf.math.truediv(M_matrix_4, M_matrix_3, name=None)*pooled_width
        dx =-roi_pooled_width/2
        dy =-pooled_height/2.0
        Sx = tf.math.truediv(M_matrix_4, roi_pooled_width, name=None)*spatial_scale
        Sy = M_matrix_3/pooled_height*spatial_scale
        Alpha = tf.math.cos(M_matrix_5)
        Beta = tf.math.sin(M_matrix_5)
        Dx = M_matrix_1*spatial_scale
        Dy = M_matrix_2*spatial_scale
        M_0_0 = tf.math.multiply(Alpha,Sx)
        M_0_1 =  tf.math.multiply(Beta,Sy)
        M_0_2 = M_0_0*dx + M_0_1*dy + Dx
        M_1_0 = -tf.math.multiply(Beta,Sx)
        M_1_1 = tf.math.multiply(Alpha,Sy)
        M_1_2 = M_1_0*dx + M_1_1*dy + Dy
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
        bin_l = tf.math.floor(bin_cx)
        bin_r = tf.math.ceil(bin_cx)
        bin_t = tf.math.floor(bin_cy)
        bin_b = tf.math.ceil(bin_cy)
        posi_bin_t= bin_t>0 and bin_t<height
        posi_bin_l = bin_l>0 and bin_l<width
        posi_bin_b = bin_b>0 and bin_b< height
        posi_bin_r = bin_r>0 and bin_r < width
        lt_value = features(posi_bin_t*posi_bin_l)
        rt_value = features(posi_bin_t*posi_bin_l)
        lb_value = features(posi_bin_b*posi_bin_l)
        rb_value = features(posi_bin_r*posi_bin_b)
        rx = bin_cx - tf.math.floor(bin_cx)
        ry = bin_cy - tf.math.floor(bin_cy)
        wlt = (1-rx)*(1-ry)
        wrt = rx*(1-ry)
        wrb = rx*ry
        wlb = (1-rx)* ry
        output = lt_value * wlt + rt_value * wrt + rb_value * wrb + lb_value * wlb
        con_idx_x = tf.floor(bin_cx)
        con_idx_y = tf.floor(bin_cy)
        return output, con_idx_x, con_idx_y