import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage.io import imread, imshow
from skimage.feature import blob_dog, blob_log, blob_doh
from scipy.ndimage import gaussian_filter
import cv2
import os
from tqdm import tqdm
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import distance, distance_matrix
from cProfile import Profile
import scipy.ndimage as nd
from scipy.signal import find_peaks,peak_prominences
from scipy.optimize import curve_fit
from multiprocessing import Process
from scipy import fftpack
from skimage import measure
from scipy.interpolate import interp2d
from scipy.spatial.distance import euclidean
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max
import networkx as nx
from scipy import spatial
from copy import deepcopy
from scipy.stats import binned_statistic

def get_displacement_field(cur_cors, pre_cors, upper_lim):
    tree = spatial.cKDTree(pre_cors)
    distances, ndx = tree.query(cur_cors)
    vectors = np.zeros([cur_cors.shape[0], 2])
    for i in range(cur_cors.shape[0]):
        vector = cur_cors[i] - pre_cors[ndx[i]]
        if np.linalg.norm(vector) > upper_lim: continue;
        vectors[i] = vector
    return vectors

def disp_stat(atom_loc, atom_disp, bins = 20, dist_direction = 'x', disp_direction = 'y'):
    atom_loc_idx = 0 if dist_direction == 'x' else 1
    disp_direction_idx = 0 if disp_direction == 'x' else 1
    mean_stat = binned_statistic(atom_loc[:,atom_loc_idx], atom_disp[:,disp_direction_idx], 
                                 statistic='mean', 
                                 bins=bins)
    std_stat = binned_statistic(atom_loc[:,atom_loc_idx], atom_disp[:,disp_direction_idx], 
                                 statistic='std', 
                                 bins=bins)
    fig,ax = plt.subplots(figsize=[6,4])
    plt.plot(mean_stat.bin_edges[:-1],mean_stat.statistic,'o')
    plt.errorbar(mean_stat.bin_edges[:-1],mean_stat.statistic,yerr=std_stat.statistic,linestyle = 'None')
    disp_x = mean_stat.bin_edges[:-1]
    disp_y = mean_stat.statistic
    disp_y = np.nan_to_num(disp_y)
    disp_yerr = np.nan_to_num(std_stat.statistic)
    plt.xlabel('Distance '+dist_direction+'/ pixel',fontsize=17)
    plt.ylabel('Displacement '+u'$\Delta$'+disp_direction+'/ pixel',fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    return disp_x, disp_y, disp_yerr