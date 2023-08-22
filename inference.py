#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 21:42:09 2021

@author: jake drysdale
"""
#imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


#local imports
import pylib as py
import tf2lib as tl
import tf2gan as gan
import module_eval as module


def get_args():
    #dataset arguments
    py.arg('-d', '--dataset', type=str, default='/home/jake/Documents/data/thesis_eval_data_v2/aug_oneshots_16k/train')
    py.arg('-p', '--preproc', type=bool, default=False)
    py.arg('-cl', '--n_classes', type=int, default=3)
    py.arg('--sr', type=int, default=16000)
    py.arg('-len','--audio_length', type=int, default=8192) #16384
    
    #training arguments
    py.arg('-z', '--z_dim', type=int, default=100)
    py.arg('-ch', '--channels', type=int, default=1)
    py.arg('-b', '--batch_size', type=int, default=64)
    py.arg('-e', '--epochs', type=int, default=10000)
    py.arg('--lr', type=float, default=0.0002)
    py.arg('--beta_1', type=float, default=0.5)
    py.arg('--n_d', type=int, default=3)
    
    #model arguments
    py.arg('--e_dim', type=int, default=50)
    py.arg('--num_res', type=int, default=9)
    py.arg('--scale_base', type=int, default=2)
    py.arg('--n_filters', type=int, default=512) #1024
    py.arg('--n_mapping', type=int, default=6)
    py.arg('--adversarial_loss_mode', default='wgan', choices=['wgan'])
    py.arg('--gradient_penalty_mode', default='wgan-gp', choices=['wgan-gp'])
    py.arg('--gradient_penalty_weight', type=float, default=10.0)
    py.arg('--experiment_name', default='style_cond_mapping_z64_scale2_thesis_16k_8912_z100_tomatchstylecond_this')

    return(py.args()) 

args = get_args()

def load_model():
    
    
    # output_dir
    if args.experiment_name == 'none':
        args.experiment_name = '%s_%s' % (args.experiment_name, 
                                          args.adversarial_loss_mode)
        if args.gradient_penalty_mode != 'none':
            args.experiment_name += '_%s' % args.gradient_penalty_mode
    output_dir = py.join('output', args.experiment_name)
    py.mkdir(output_dir)
    
    # save settings
    py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)
    
    
    # =========================================================================
    # model building
    # =========================================================================
    networks = module.Networks(latent_size=args.z_dim, 
                               num_res=args.num_res, 
                               n_classes=args.n_classes,
                               n_chan=args.channels,
                               n_filters=args.n_filters,
                               scale_base=args.scale_base,
                               embedding_dim=50,
                               mapping_size=args.n_mapping)
    
    G = networks._G
    D = networks._D
    G_s = networks._G_synth
    S = networks._style_net
    # G.summary()
    # D.summary()
    
    
    # loss functions
    d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
    
    G_optimizer = keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1)
    D_optimizer = keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1)
    


    
    # epoch counter
    ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
    
    
    # checkpoint
    checkpoint = tl.Checkpoint(dict(G=G,
                                    D=D,
                                    S=S,
                                    G_s=G_s,
                                    G_optimizer=G_optimizer,
                                    D_optimizer=D_optimizer,
                                    ep_cnt=ep_cnt),
                               py.join(output_dir, 'checkpoints'),
                               max_to_keep=5)
    try:  # restore checkpoint including the epoch counter
        checkpoint.restore().expect_partial()
        # .assert_existing_objects_matched()
    except Exception as e:
        print(e)
    
    return G, S

#G, S = load_model()

def layer_noise(num):
    threshold = np.int32(np.random.uniform(0.0, 5, size = [num]))
    n1 = tf.random.normal(shape=(num, args.z_dim))
    n2 = tf.random.normal(shape=(num, args.z_dim))
    
    n = []
    for i in range(args.num_res-1):
        n.append([])
        for j in range(num):
            if i < threshold[j]:
                n[i].append(n1[j])
            else:
                n[i].append(n2[j])
        n[i] = tf.convert_to_tensor(n[i])
    return n



def generate_noise(condition, stylenet, noise):
    labels_save=np.array([condition])
    
    all_labels_save=[]
    for i in range(args.num_res-1):
        all_labels_save.append(labels_save)
    
    # n_save = layer_noise(1)
    n_save = noise
    w_noise=[]
    for i in range(len(n_save)):
        w_noise.append(stylenet([n_save[i], all_labels_save[i]]))
    return w_noise


def feature_slider(w, directions, component, amount):

    w_hat = np.clip(w+(directions[component]*amount), 0,1000)

    return w_hat



@tf.function
def sample(G, styles):
    
    
    ones = np.ones((1, 1), dtype=np.float32)
    inc_noise = np.random.uniform(0.0, 1.0,
                                   size = [1, 
                                           args.audio_length, args.channels])

    
    
    return G(styles + [ones, inc_noise], training=False)

def load_models():
    G, S = load_model()
    return G, S

def generate(cond, generator, stylenet):
    latent = generate_noise(cond, stylenet)
    return sample(generator,latent)[0]