#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 15:36:52 2023

@author: jake
"""
import umap
import umap.plot
import numpy as np
import os


w_dir = '/media/jake/ubuntu_storage/thesis_final_evaluation/neuraldrumsynth/neuraldrumsynth_gui/umap_files'
w_paths = [os.path.join(w_dir, x) for x in os.listdir(w_dir) if x.endswith('.npy')]
w_noise = [np.load(x) for x in w_paths]

kick_w = w_noise[0]
snare_w =  w_noise[1]
cymbal_w =  w_noise[2]




kick_embedding = umap.UMAP(n_neighbors=20,
                      min_dist=1,
                      metric='euclidean').fit_transform(kick_w)

kick_mapper = umap.UMAP(n_neighbors=20,
                      min_dist=1,
                      metric='euclidean').fit(kick_w)
# umap.plot.points(mapper)



snare_embedding = umap.UMAP(n_neighbors=20,
                      min_dist=1,
                      metric='euclidean').fit_transform(snare_w)

snare_mapper = umap.UMAP(n_neighbors=20,
                      min_dist=1,
                      metric='euclidean').fit(snare_w)

cymbal_embedding = umap.UMAP(n_neighbors=20,
                             n_components=2,
                             min_dist=1,
                             metric='euclidean').fit_transform(snare_w)

cymbal_mapper = umap.UMAP(n_neighbors=20,
                          n_components = 2,
                          min_dist=1,
                          metric='euclidean').fit(cymbal_w)

embedding = snare_embedding
mapper = snare_mapper





#Samplable points for 2D mapping - creates the corner vlaues of a 2d plot/box
corners = np.array([
    [-5, 10],  # 1
    [-5, -0.8],  # 7
    [15, 10],  # 2
    [15, -0.8],  # 0
])




# create possible values in 2D space, 10x10, change the 10 for more resolution
test_pts = np.array([
    (corners[0]*(1-x) + corners[1]*x)*(1-y) +
    (corners[2]*(1-x) + corners[3]*x)*y
    for y in np.linspace(0, 1, 10)
    for x in np.linspace(0, 1, 10)
])




# invert back to z^D space
inv_kick_points = kick_mapper.inverse_transform(test_pts)
inv_snr_points = snare_mapper.inverse_transform(test_pts)
inv_cymb_points = cymbal_mapper.inverse_transform(test_pts)

# save them
w_dir = '/media/jake/ubuntu_storage/thesis_final_evaluation/neuraldrumsynth/neuraldrumsynth_gui/latent_values/umap_values'
np.save(w_dir+'/kick_umap.npy',inv_kick_points)
np.save(w_dir+'/snare_umap.npy',inv_snr_points)
np.save(w_dir+'/cymbal_umap.npy',inv_cymb_points)
