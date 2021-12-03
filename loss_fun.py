import tensorflow as tf
from tensorflow import keras
# https://github.com/calmisential/TensorFlow2.0_ResNet/blob/master/models/resnet.py
# Helper libraries
import numpy as np
def loss_fn(label,predict):
    #label 1 is  recognition label 2 is detect
    alpha=1
    beta = 1

    ctc_loss = tf.reduce_sum(ctc(label[1],predict[1]))
    detect_loss = detect_loss_fun(label[0],predict[0])
    loss = alpha*ctc_loss+ tf.cast(detect_loss,tf.float32)*beta
    return loss, ctc_loss, detect_loss
def ctc(label,pre):
    pre =tf.squeeze(pre,axis=1)
    label_length = tf.constant(np.ones((32)), dtype=tf.int32)
    logit_length = tf.constant(np.ones((32)),  dtype=tf.int32)
    # loss= tf.nn.ctc_loss(label[:8], pre, label_length, logit_length, logits_time_major=True)
    loss= tf.nn.ctc_loss(label[:32], tf.transpose(pre,perm=[1,0,2]), label_length, logit_length, logits_time_major=True)
    return loss
def detect_loss_fun(label,pre):
    score_loss = iou_loss(label[0],pre[1])
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=tf.cast(label[1],tf.double), num_or_size_splits=5, axis=0)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=tf.cast(pre[0][0,:,:,:],tf.double), num_or_size_splits=5, axis=2)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    L_AABB = -tf.math.log((area_intersect + 1.0)/(area_union + 1.0))
    L_theta = 1 - tf.math.cos(theta_pred - theta_gt)
    L_g = L_AABB + 20 * L_theta
    return tf.cast(tf.reduce_mean(L_g),tf.float32)+score_loss
def iou_loss(label,pre):
    pre = tf.squeeze(pre,axis=0)
    # pre = pre > 0.6
    intersection = tf.math.minimum(tf.expand_dims(pre[:,:,0],-1), tf.constant(label,dtype = tf.float32))
    union = tf.math.maximum(tf.expand_dims(pre[:,:,0],-1), tf.constant(label,dtype = tf.float32))
    area_intersection = tf.reduce_sum(intersection[:,:,0],axis=1)
    area_union = tf.reduce_sum(union[:,:,0],axis= 1)
    IOU_loss = area_intersection/(area_union+ 1e-8)
    return tf.reduce_sum(1-IOU_loss)
def dice_loss(overly_small_text_region_training_mask, text_region_boundary_training_mask, loss_weight, small_text_weight):
    def loss(y_true, y_pred):
        eps = 1e-5
        _training_mask = tf.minimum(overly_small_text_region_training_mask + small_text_weight, 1) * text_region_boundary_training_mask
        intersection = tf.reduce_sum(y_true * y_pred * _training_mask)
        union = tf.reduce_sum(y_true * _training_mask) + tf.reduce_sum(y_pred * _training_mask) + eps
        loss = 1. - (2. * intersection / union)
        return loss * loss_weight
    return loss

def rbox_loss(overly_small_text_region_training_mask, text_region_boundary_training_mask, small_text_weight, target_score_map):
    def loss(y_true, y_pred):
        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred, num_or_size_splits=5, axis=3)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
        h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -tf.math.log((area_intersect + 1.0)/(area_union + 1.0))
        L_theta = 1 - tf.math.cos(theta_pred - theta_gt)
        L_g = L_AABB + 20 * L_theta
        _training_mask = tf.minimum(overly_small_text_region_training_mask + small_text_weight, 1) * text_region_boundary_training_mask
        return tf.reduce_mean(L_g * target_score_map * _training_mask)
    return loss
