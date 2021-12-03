from re import L
from numpy import linalg
from numpy.lib.function_base import angle
import tensorflow as tf
from tensorflow import keras
# https://github.com/calmisential/TensorFlow2.0_ResNet/blob/master/models/resnet.py
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from skimage.draw import polygon
import numpy as np
from shapely.geometry import Polygon
import cv2
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_math_ops import betainc
def distance2point(point,p1,p2):
    return np.linalg.norm(np.cross(p2 - p1, p1 - point)) / np.linalg.norm(p2 - p1)
def mapscore_geo(inputs):
    inputs=inputs.reshape(-1,8)
    map_score=np.zeros([1,128,128,1])
    dis_left = np.zeros([1,128,128,1])
    dis_right = np.zeros([1,128,128,1])
    dis_top = np.zeros([1,128,128,1])
    dis_bottom = np.zeros([1,128,128,1])
    dis_angle = np.zeros([1,128,128,1])
    for input in inputs:
        rr, cc = polygon(input[1::2],input[::2])
        input = input.reshape(4,2)
        
        map_score[0,list(rr),list(cc),0]=1
        for i in zip(rr,cc):
            i= np.array(i)
            dis_left[0,i[0],i[1],0]=distance2point(i,input[0,::-1],input[3,::-1])
            dis_right[0,i[0],i[1],0]=distance2point(i,input[1,::-1],input[2,::-1])
            dis_top[0,i[0],i[1],0]= distance2point(i,input[0,::-1],input[1,::-1])
            dis_bottom[0,i[0],i[1],0]= distance2point(i,input[2,::-1],input[3,::-1])
        right_bot=[input[2][0],max(input[2][1],input[1][1])]
        dis_angle[0,rr,cc,0]=np.arccos(
            np.cross(input[2]-input[3],right_bot-input[3])/
        (np.linalg.norm(input[2]-input[3])*np.linalg.norm( right_bot -input[3])))
        
        
        geo_map = np.concatenate([dis_right,dis_left,dis_top,dis_bottom,dis_angle])
    return map_score, geo_map

def intersection(g, p):
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union


def weighted_merge(g, p):
    g[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])
    g[8] = (g[8] + p[8])
    return g


def standard_nms(S, thres):
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds+1]

    return S[keep]
# @tf.function
def nms_local_tf(poly):
    return tf.numpy_function(nms_locality,[poly],tf.float32)
def nms_locality(polys, thres=0.3):
    '''
    locality aware nms
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    '''
    S = []
    p = None
    for g in polys:
        if p is not None and intersection(g, p) > thres:
            p = weighted_merge(g, p)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)

    if len(S) == 0:
        return np.array([])
    return standard_nms(np.array(S), thres)
# @tf.function
# def restore_rectangle_rbox(origin, geometry):
#     d = geometry[:, :4]
#     angle = geometry[:, 4]
#     # for anglIncompatible shapes: [1,128,128,1] vs. [1,512,512,1]e > 0
#     origin_0 = origin[angle >= 0]
#     d_0 = d[angle >= 0]
#     angle_0 = angle[angle >= 0]
#     if origin_0.shape[0] > 0:
#         p = np.array([tf.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
#                       d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
#                       d_0[:, 1] + d_0[:, 3], tf.zeros(d_0.shape[0]),
#                       tf.zeros(d_0.shape[0]), tf.zeros(d_0.shape[0]),
#                       d_0[:, 3], -d_0[:, 2]])
#         p = tf.reshape(tf.transpose(p,(1, 0)),(-1, 5, 2))  # N*5*2

#         rotate_matrix_x = tf.transpose(tf.stack([tf.cos(angle_0), tf.sin(angle_0)],0),(1, 0))
#         rotate_matrix_x = tf.transpose(tf.reshape(tf.repeat(rotate_matrix_x, 5, axis=1),(-1, 2, 5)),(0, 2, 1))  # N*5*2

#         rotate_matrix_y = tf.transpose(tf.stack([-tf.sin(angle_0), tf.cos(angle_0)],0),(1, 0))
#         rotate_matrix_y = tf.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

#         p_rotate_x = tf.reduce_sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
#         p_rotate_y = tf.reduce_sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

#         p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

#         p3_in_origin = origin_0 - p_rotate[:, 4, :]
#         new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
#         new_p1 = p_rotate[:, 1, :] + p3_in_origin
#         new_p2 = p_rotate[:, 2, :] + p3_in_origin
#         new_p3 = p_rotate[:, 3, :] + p3_in_origin

#         new_p_0 =tf.concat([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
#                                   new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
#     else:
#         new_p_0 = np.zeros((0, 4, 2))
#     # for angle < 0
#     origin_1 = origin[angle < 0]
#     d_1 = d[angle < 0]
#     angle_1 = angle[angle < 0]
#     if origin_1.shape[0] > 0:
#         p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
#                       np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
#                       np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
#                       -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
#                       -d_1[:, 1], -d_1[:, 2]])
#         p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

#         rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
#         rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

#         rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
#         rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

#         p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
#         p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

#         p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

#         p3_in_origin = origin_1 - p_rotate[:, 4, :]
#         new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
#         new_p1 = p_rotate[:, 1, :] + p3_in_origin
#         new_p2 = p_rotate[:, 2, :] + p3_in_origin
#         new_p3 = p_rotate[:, 3, :] + p3_in_origin

#         new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
#                                   new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
#     else:
#         new_p_1 = np.zeros((0, 4, 2))
#     return np.concatenate([new_p_0, new_p_1])



def restore_rectangle(origin, geometry):
    return restore_rectangle_rbox(origin, geometry)
# @tf.function
def detect(score_map, geo_map, score_map_thresh=0.2, box_thresh=0.1, nms_thres=0.2):
    '''
    input:
    score_map:(1,128,128,1)
    geo_map:(1,128,128,5)
    output:
    bounding_box:(num_roi,6)
    param:
    score_map_thresh
    box_thresh
    nms_thres
    process:
    step1:
    threshold score_map with thresh score map
    step2:
    Find bounding box of each word follow geo map
    step3:
    compute the angle of bounding box.
    step4:
    return result 
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = tf.where(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = tf.gather(xy_text,tf.sort(xy_text[:,0]))
    indice=tf.stack([xy_text[:, 0],xy_text[:, 1],tf.ones_like(xy_text[:,1])*0],axis=-1)
    geo_map_selec=tf.reshape(tf.gather_nd(geo_map,indice),(-1,1))
    # restore
    for i in range(1,5,1):
        indice=tf.stack([xy_text[:, 0],xy_text[:, 1],tf.ones_like(xy_text[:,1])*i],axis=-1)
        geo_map_selec=tf.concat([geo_map_selec,tf.reshape(tf.gather_nd(geo_map,indice),(-1,1))],1)
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map_selec ) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = tf.concat([tf.reshape(text_box_restored,(-1, 8)),tf.reshape(tf.cast(tf.gather_nd(score_map,xy_text),tf.int64),(-1,1))],1)
    # boxes[:, :8] = tf.reshape(text_box_restored,(-1, 8))
    # boxes[:, 8] = tf.gather_nd(score_map,xy_text)
    # nms part
    # boxes = nms_local_tf(boxes)
    boxes = tf.stack([boxes[:,0],boxes[:,1],boxes[:,4],boxes[:,5]],axis=1)
    index = tf.image.non_max_suppression(tf.cast(boxes,tf.float32),tf.ones_like(tf.reduce_sum(boxes,axis=1),dtype=tf.float32)*0.6,iou_threshold=0.5,max_output_size=32)
    boxes = tf.gather(boxes, index)
    # boxes = merge_quadrangle_n9(boxes.astype('float32'), nms_thres)

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    # for i, box in enumerate(boxes):
    #     mask = tf.zeros_like(score_map, dtype=tf.uint8)
    #     cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
    #     boxes[i, 8] = cv2.mean(score_map, mask)[0]
    # boxes = boxes[boxes[:, 8] > box_thresh]
    boxes = tf.expand_dims(tf.stack([tf.cast(tf.ones_like(boxes[:,0]),tf.int64),boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3],tf.zeros_like(boxes[:,0])],1),axis=0)
    # plt.imshow(score_map)
    # plt.figure(figsize=(16,14))
    # plt.show
    return boxes
# @tf.function
def restore_rectangle_rbox(origin, geometry):
    d = geometry[:, :4]
    angle = geometry[:, 4]
    # for anglIncompatible shapes: [1,128,128,1] vs. [1,512,512,1]e > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]

    p = tf.stack([tf.zeros_like(d_0[:,0]), -d_0[:, 0] - d_0[:, 2],
                    d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                    d_0[:, 1] + d_0[:, 3], tf.zeros_like(d_0[:,0]),
                    tf.zeros_like(d_0[:,0]), tf.zeros_like(d_0[:,0]),
                    d_0[:, 3], -d_0[:, 2]],-1)
    p = tf.reshape(tf.transpose(p,(1, 0)),(-1, 5, 2))  # N*5*2

    rotate_matrix_x = tf.transpose(tf.stack([tf.math.cos(angle_0),tf.math.sin(angle_0)],0),(1, 0))
    rotate_matrix_x = tf.transpose(tf.reshape(tf.repeat(rotate_matrix_x, 5, axis=1),(-1, 2, 5)),(0, 2, 1))  # N*5*2

    rotate_matrix_y = tf.transpose(tf.stack([-tf.math.sin(angle_0), tf.math.cos(angle_0)]),(1, 0))
    rotate_matrix_y = tf.transpose(tf.reshape(tf.repeat(rotate_matrix_y, 5, axis=1),(-1, 2, 5)),(0, 2, 1))

    p_rotate_x = tf.reduce_sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
    p_rotate_y = tf.reduce_sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

    p_rotate = tf.concat([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

    p3_in_origin = origin_0 - tf.cast(p_rotate[:, 4, :],tf.int64)
    new_p0 = tf.cast(p_rotate[:, 0, :],tf.int64) + p3_in_origin  # N*2
    new_p1 = tf.cast(p_rotate[:, 1, :],tf.int64) + p3_in_origin
    new_p2 = tf.cast(p_rotate[:, 2, :],tf.int64) + p3_in_origin
    new_p3 = tf.cast(p_rotate[:, 3, :],tf.int64) + p3_in_origin

    new_p_0 = tf.concat([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    # else:
    #     new_p_0 = np.zeros((0, 4, 2))
    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    # if origin_1.shape[0] > 0:
    p = tf.stack([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                    tf.zeros_like(d_1[:,0]), -d_1[:, 0] - d_1[:, 2],
                    tf.zeros_like(d_1[:,0]), tf.zeros_like(d_1[:,0]),
                    -d_1[:, 1] - d_1[:, 3], tf.zeros_like(d_1[:,0]),
                    -d_1[:, 1], -d_1[:, 2]],-1)
    p = tf.reshape(tf.transpose(p,(1, 0)),(-1, 5, 2))  # N*5*2

    rotate_matrix_x = tf.transpose(tf.stack([tf.math.cos(-angle_1), -tf.math.sin(-angle_1)],0),(1, 0))
    rotate_matrix_x = tf.transpose(tf.reshape(tf.repeat(rotate_matrix_x, 5, axis=1),(-1, 2, 5)),(0, 2, 1))  # N*5*2

    rotate_matrix_y = tf.transpose(tf.stack([tf.math.sin(-angle_1), tf.math.cos(-angle_1)],0),(1, 0))
    rotate_matrix_y = tf.transpose(tf.reshape(tf.repeat(rotate_matrix_y, 5, axis=1),(-1, 2, 5)),(0, 2, 1))

    p_rotate_x = tf.reduce_sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
    p_rotate_y = tf.reduce_sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

    p_rotate = tf.concat([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

    p3_in_origin = origin_1 - tf.cast(p_rotate[:, 4, :],tf.int64)
    new_p0 = tf.cast(p_rotate[:, 0, :],tf.int64) + p3_in_origin  # N*2
    new_p1 = tf.cast(p_rotate[:, 1, :],tf.int64) + p3_in_origin
    new_p2 = tf.cast(p_rotate[:, 2, :],tf.int64) + p3_in_origin
    new_p3 = tf.cast(p_rotate[:, 3, :],tf.int64) + p3_in_origin

    new_p_1 = tf.concat([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    # else:
    #     new_p_1 = np.zeros((0, 4, 2))
    return tf.concat([new_p_0, new_p_1],0)

if __name__ == '__main__':
    # 343,350,448,135,474,143,369,359
    print(Polygon(np.array([[343, 350], [448, 135],
                            [474, 143], [369, 359]])).area)
    inputs = np.array([[[21,8],[60,6],[60,23],[19,22]],
            [[70,64],[120,64],[120,90],[70,90]]])
    score_map,geo_map=mapscore_geo(inputs)
    geo_map =geo_map.transpose(-1,1,2,0)
    geo_map= tf.constant(geo_map,tf.float32)
    score_map = tf.constant(score_map,tf.float32)
    bb=detect(score_map,geo_map)
    print(bb)
    plt.imshow(score_map[0,:,:,0])
    plt.show()
