# Copyright UCL Business plc 2017. Patent Pending. All rights reserved. 
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence, 
# please contact info@uclb.com

"""Fully convolutional model for monocular depth estimation
    by Clement Godard, Oisin Mac Aodha and Gabriel J. Brostow
    http://visual.cs.ucl.ac.uk/pubs/monoDepth/
"""

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import scipy as sp
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

search_rng = 33
left_shift_ = int((search_rng-1)/2)

tfcv_parameters = namedtuple('parameters', 
                        'encoder, '
                        'height, width, '
                        'height_o, width_o, '                        
                        'batch_size, '
                        'num_threads, '
                        'num_epochs, '
                        'do_stereo, '
                        'wrap_mode, '
                        'use_deconv, '
                        'post_proc, '
                        'alpha_image_loss, '
                        'disp_gradient_loss_weight, '
                        'lr_loss_weight, '
                        'full_summary')

class deep3dModel(object):
    """deep3d model"""

    def __init__(self, params, mode, left, right, reuse_variables=None, model_index=0):
        self.params = params
        self.mode = mode
        self.left = left
        self.right = right
        self.model_collection = ['model_' + str(model_index)]

        self.reuse_variables = reuse_variables

        self.build_model()
        self.build_outputs()
        if self.mode == 'single':
            return 

        self.build_losses()
        if self.mode == 'test':
            return        
        self.build_summaries()     

    ## Evauation functions
    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def LSSIM(self, x, y):
        SSIM = tf.image.ssim(x, y, max_val = 1.0) 
        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)
        _, height, width, _ = x.shape
        lssim_stack = []
        for i in range(height-11):
            for j in range(width-11):
                x_ = x[:,i:(i+11),j:(j+11),:]
                y_ = y[:,i:(i+11),j:(j+11),:]
                lssim = tf.image.ssim(x_, y_, max_val = 1.0) 
                lssim = tf.clip_by_value((1 - lssim) / 2, 0, 1)
                lssim_stack.append(lssim)

        return(tf.reduce_mean(lssim_stack))

    ## Post processing function
    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def gradient_x(self, img):
        gx = img[:,:,:-1] - img[:,:,1:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:] - img[:,1:,:]
        return gy

    def get_smoothness(self, disp, img):     
        disp_x = tf.pad(disp, [[0,0],[0,0],[0,1]], "SYMMETRIC")
        disp_y = tf.pad(disp, [[0,0],[0,1],[0,0]], "SYMMETRIC")
        disp_gradients_x = self.gradient_x(disp_x)
        disp_gradients_y = self.gradient_y(disp_y)        

        img_x = tf.pad(img, [[0,0],[0,0],[0,1],[0,0]], "SYMMETRIC")
        img_y = tf.pad(img, [[0,0],[0,1],[0,0],[0,0]], "SYMMETRIC")
        image_gradients_x = self.gradient_x(img_x)
        image_gradients_y = self.gradient_y(img_y) 

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))
        smoothness_x = disp_gradients_x * weights_x[:,:,:,0]
        smoothness_y = disp_gradients_y * weights_y[:,:,:,0]       
        return smoothness_x + smoothness_y

    ## CNN functions
    def conv(self, x, num_out_layers, kernel_size, stride=1, activation_fn=tf.nn.relu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x,     num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 1)
        return conv2

    def fully_conn(self, x, num_out_layers, l2_penalty=1e-8, activation_fn=tf.nn.relu):
        net = slim.flatten(x)
        output = slim.fully_connected(net, num_out_layers, activation_fn=activation_fn, 
            weights_regularizer=slim.l2_regularizer(l2_penalty))
        return output

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, stride, pad=1):
        pad = int(pad)
        p_x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, stride, 'SAME')      
        if(pad):
            pad_ = stride*pad
            return tf.nn.relu(conv[:,pad_:-pad_,pad_:-pad_,:])
        else:
            return tf.nn.relu(conv)

    def pred_deconv(self, x, num_out_layers, scale):
        bn_pool = slim.batch_norm(x)
        pred_conv = self.conv(bn_pool, num_out_layers, 3, 1)
        pred_deconv = self.deconv(pred_conv, num_out_layers=num_out_layers, kernel_size=2*scale, stride=scale, pad=scale/2)        
        return pred_deconv

    def build_vgg(self):
        batchNum = self.params.batch_size        
        layerNum = search_rng
        with tf.variable_scope('encoder'):
            conv1 = self.conv_block(self.model_input,   64, 3)
            pool1 = self.maxpool(conv1,                     2)   # H/2
            
            conv2 = self.conv_block(pool1,             128, 3)
            pool2 = self.maxpool(conv2,                     2)   # H/4

            conv3_1 = self.conv_block(pool2,           256, 3)   # H/4
            conv3_2 = self.conv_block(conv3_1,         256, 3)   # H/4
            pool3 = self.maxpool(conv3_2,                   2)   # H/8

            conv4_1 = self.conv_block(pool3,           512, 3)   # H/8
            conv4_2 = self.conv_block(conv4_1,         512, 3)   # H/8
            pool4 = self.maxpool(conv4_2,                   2)   # H/16

            conv5_1 = self.conv_block(pool4,           512, 3)   # H/16
            conv5_2 = self.conv_block(conv5_1,         512, 3)   # H/16
            pool5 = self.maxpool(conv5_2,                   2)   # H/32
            if(1):
                #fc8m_0  = self.conv_block(pool5, layerNum*4, 3)   # H/32
                fc8m  = self.conv_block(pool5,    layerNum, 3)   # H/32
            ##///// fully connected
            else:
                fc6   = self.fully_conn(pool5,            512)   # 512
                drop6 = slim.dropout(fc6,     keep_prob = 0.5)

                fc7   = self.fully_conn(drop6,            512)   # 512
                drop7 = slim.dropout(fc7,     keep_prob = 0.5)

                fc8   = self.fully_conn(drop7,        6*12*layerNum)   # 4*12*33
                fc8m  = tf.reshape(fc8, [batchNum, 6, 12, layerNum])   # 4, 12, 33


        with tf.variable_scope('decoder'):
            '''
            scale = 1
            bn_pool1 = slim.batch_norm(pool1)
            pred1 = self.conv(bn_pool1,          layerNum, 3, 1)
            pred1 = self.deconv(pred1, layerNum, scale, scale,0)
            '''
            pred1 = self.pred_deconv(pool1,       layerNum, 1)                       
            pred2 = self.pred_deconv(pool2,       layerNum, 2)           
            pred3 = self.pred_deconv(pool3,       layerNum, 4)
            pred4 = self.pred_deconv(pool4,       layerNum, 8)            
            pred5 = self.deconv(fc8m,     layerNum, 16, 16, 0)            

        with tf.variable_scope('featureMask'):           
            feat = pred1 + pred2 + pred3 + pred4 + pred5
            # feat_act = tf.nn.relu(feat)

            # up    = self.pred_deconv(feat_act, num_out_layers=layerNum, scale=2)
            up    = self.deconv(feat,       layerNum, 2, 2, 0)            
            up_out = self.conv(up,             layerNum, 3, 1)

            self.feat_masks = tf.nn.softmax(up_out)

    def depthDot(self, masks, left_image, left_shift=left_shift_, name="depthDot"):
        '''
        inputs:
            masks,          shape: N, H, W, S
            left_image,     shape: N, H, W, C
        returns
            right_image,    shape: N, H, W, C
        ref: https://github.com/JustinTTL/Deep3D_TF/blob/master/selection.py    
        '''
        left_shift = left_shift*2
        _, H, W, S = masks.get_shape().as_list()
        with tf.variable_scope(name):
            padded = tf.pad(left_image, [[0,0],[0,0],[0, left_shift],[0,0]], mode='REFLECT')          
    
            # padded is the image padded whatever the left_shift variable is on either side
            layers = []
            for s in np.arange(S):
                layers.append(tf.slice(padded, [0,0,s,0], [-1,H,W,-1]))
            
            slices = tf.stack(layers, axis=4)
            disparity_image = tf.multiply(slices, tf.expand_dims(masks, axis=3))
            return tf.reduce_sum(disparity_image, axis=4)

    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.relu):
            with tf.variable_scope('model', reuse=self.reuse_variables):
                #self.left_pyramid  = self.scale_pyramid(self.left,  5)
                #self.right_pyramid = self.scale_pyramid(self.right, 5)

                self.model_input = self.left
                #build model
                if self.params.encoder == 'vgg':
                    self.build_vgg()
                elif self.params.encoder == 'resnet50':
                    self.build_resnet50()
                else:
                    return None


    def generate_disp_est(self, feat_masks):
        layers = []
        for s in np.arange(search_rng):
            layers.append(feat_masks[None,:,:,s]*s)

        return tf.reduce_sum(layers, axis=3)

    def generate_image_right(self, feat_masks, left_image):
        return self.depthDot(feat_masks, left_image)        



    def build_outputs(self):
        # STORE DISPARITIES
        with tf.variable_scope('disparities'):
            self.disp_est = self.generate_disp_est(self.feat_masks) 
            self.disp_est_c3 = tf.expand_dims(self.disp_est, 3)            
            self.disp_feat = self.feat_masks 
        # GENERATE IMAGES
        with tf.variable_scope('images'):
            self.right_est = self.generate_image_right(self.feat_masks, self.model_input)


    def build_losses(self):
        # DISPARITY SMOOTHNESS
        with tf.variable_scope('smoothness'):
            #g_x = self.left[None,:,:-1,:] - self.left[None,:,1:,:] 
            #print('===>self.left: ', self.left.shape)            
            #self.disp_right_smoothness = self.get_smoothness(self.disp_est, self.right)
            pass

        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # IMAGE RECONSTRUCTION
            # L1
            self.l1_right = tf.abs(self.right_est - self.right)
            self.l1_reconstruction_loss_right = tf.reduce_mean(self.l1_right)
             
            # SSIM
            self.ssim_right = self.LSSIM(self.right_est, self.right)
            self.ssim_loss_right = tf.reduce_mean(self.ssim_right)

            # WEIGTHED SUM
            self.image_loss_right = self.params.alpha_image_loss * self.ssim_loss_right + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_right
            self.image_loss = self.image_loss_right#tf.add_n(self.image_loss_right)

            # DISPARITY SMOOTHNESS
            #self.disp_left_loss  = tf.reduce_mean(tf.abs(self.disp_right_smoothness))
            #self.disp_gradient_loss = self.disp_left_loss #tf.add_n(self.disp_right_loss)
            
            #print('===> smoothness_x: ', smoothness_x.dtype)
            # TOTAL LOSS
            self.total_loss = self.image_loss #+ self.params.disp_gradient_loss_weight * self.disp_gradient_loss# + self.params.lr_loss_weight * self.lr_loss

            # GET MAE
            self.mae = tf.reduce_mean(self.l1_right)*255

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            i = 0
            tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_right, collections=self.model_collection)
            tf.summary.scalar('l1_loss_' + str(i), self.l1_reconstruction_loss_right, collections=self.model_collection)
            tf.summary.scalar('image_loss_' + str(i), self.image_loss_right, collections=self.model_collection)
            tf.summary.image('disp_right_est' + str(i), self.disp_est_c3, max_outputs=4, collections=self.model_collection)

            if self.params.full_summary:
                tf.summary.image('left_est_' + str(i), self.left_est, max_outputs=4, collections=self.model_collection)
                tf.summary.image('right_est_' + str(i), self.right_est, max_outputs=4, collections=self.model_collection)
                tf.summary.image('ssim_right_' + str(i), self.ssim_right, max_outputs=4, collections=self.model_collection)
                tf.summary.image('l1_right_' + str(i), self.l1_right, max_outputs=4, collections=self.model_collection)

            if self.params.full_summary:
                tf.summary.image('right', self.right,  max_outputs=4, collections=self.model_collection)


