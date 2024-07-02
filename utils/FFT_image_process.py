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


def read_image(path, frame):
    cap = cv2.VideoCapture(path)
    frames = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT) ))
    width=int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_WIDTH)))-1
    height=int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_HEIGHT)))-1
    frame_number=frame
    cap.set(1, frame_number-1)
    res, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def radial_profile(data, center):
    y,x = np.indices((data.shape)) # first determine radii of all pixels
    r = np.sqrt((x-center[0])**2+(y-center[1])**2)
    ind = np.argsort(r.flat) # get sorted indices
    sr = r.flat[ind] # sorted radii
    sim = data.flat[ind] # image values sorted by radii
    ri = sr.astype(np.int32) # integer part of radii (bin size = 1)
    # determining distance between changes
    deltar = ri[1:] - ri[:-1] # assume all radii represented
    rind = np.where(deltar)[0] # location of changed radius
    nr = rind[1:] - rind[:-1] # number in radius bin
    csim = np.cumsum(sim, dtype=np.float64) # cumulative sum to figure out sums for each radii bin
    tbin = csim[rind[1:]] - csim[rind[:-1]] # sum for image values in radius bins
    radialprofile = tbin/nr # the answer
    return radialprofile

def get_peak_position(image, prominence):
    F1 = fftpack.fft2(image.astype(float))
    F2 = fftpack.fftshift(F1)
    Profile = radial_profile(abs(F2),np.array(F2.shape)//2)
    peaks, _ = find_peaks(Profile, prominence=prominence)  
    Peaks = np.transpose([peaks,peak_prominences(Profile,peaks)[0]])
    Peak_position = Peaks[Peaks[:, 1].argsort()][-1][0]
    return Peak_position

def fft_filtered_image(image, Peak_start, Peak_end):
    F1 = fftpack.fft2((image).astype(float))
    F2 = fftpack.fftshift(F1)
    mask = np.zeros(F2.shape[:2], dtype="uint8")
    cv2.circle(mask, (F2.shape[0]//2,F2.shape[1]//2), Peak_end, 255, -1)
    cv2.circle(mask, (F2.shape[0]//2,F2.shape[1]//2), Peak_start, 0, cv2.FILLED)
    F2 = F2*mask
    im_filtered = fftpack.ifft2(fftpack.ifftshift(F2)).real
    return im_filtered

def find_diff_spots(image_filtered, threshold, direction = 'y'):
    F1 = fftpack.fft2((image_filtered).astype(float))
    F2 = abs(fftpack.fftshift(F1))
    F2 /= F2[F2!=0].min()
    F2 = rescale_intensity(F2, out_range=(0,255))
    F2 = gaussian_filter(F2,2)
    if direction == 'y':
        blobs_log = blob_log(F2[:,:np.array(F2.shape[0])//2], min_sigma = 2, max_sigma=6, threshold=threshold)  
    else:
        blobs_log = blob_log(F2[:np.array(F2.shape[0])//2,:], min_sigma = 2, max_sigma=6, threshold=threshold)  
    return F2, blobs_log

def get_grain_from_binary_mapping(blobs, size_thres):
    object_crystal_list = np.zeros_like(blobs)
    all_labels, num_labels = measure.label(blobs,return_num=True)
    num_pixel = np.bincount(all_labels[all_labels!=0].flatten())
    num_grain = (num_pixel>size_thres).sum()
    object_labels = num_pixel.argsort()[::-1][:num_grain]
    for object_label in object_labels:
        object_crystal = all_labels == object_label
        object_crystal_list += object_crystal
    return object_crystal_list

def group_diff_spots(image, blobs_log, size_thres, grain_mask_thres, R_diffspot, overlap_thres):
    F1 = fftpack.fft2((image).astype(float))
    F2 = fftpack.fftshift(F1)
    object_crystal_areas = []
    diffraction_spot_list = []
    num_diffraction_spot = len(blobs_log)
    for i in range(num_diffraction_spot):
        mask = np.zeros(F2.shape[:2], dtype="uint8")
        cv2.circle(mask, blobs_log[i,:2].astype(int)[::-1], R_diffspot, 255, -1)
        im1 = fftpack.ifft2(fftpack.ifftshift(F2*mask)).real
        im = abs(im1)
        im1 = gaussian_filter(im,5)   #################################################### filter
        im1 = rescale_intensity(im1, out_range=(0,255)).astype(np.uint8)
        # blobs = im1 > 0.2 * im1.max()   #################################################### threshold
        otsu_threshold, image_result = cv2.threshold(
            im1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
        blobs = im1 > otsu_threshold*grain_mask_thres
        object_crystal = get_grain_from_binary_mapping(blobs, size_thres)
        object_crystal_areas.append(object_crystal)
        diffraction_spot_list.append(blobs_log[i,:2].astype(int)[::-1])
        
    grains = []          # contains a list of grains [[[grain area],[grain diffraction spot]],....], Duplicate to be remove
    duplicate = []
    num_grain = len(object_crystal_areas)
    for i in range(num_grain):
        if i in duplicate: continue; 
        overlap = object_crystal_areas[i]
        grain_spot = [i]
        for j in range(num_grain):
            if i == j: continue;
            new_pattern = overlap*object_crystal_areas[j]
            if new_pattern.sum() > overlap_thres*object_crystal_areas[j].sum() or new_pattern.sum() > overlap_thres*overlap.sum(): ## same grain if 60% overlap
                overlap = new_pattern
                grain_spot.append(j)
                duplicate.append(j)
        grains.append([overlap, grain_spot])
    grains = np.array(grains)
    
    diffraction_spots, grain_masks = [], []
    for j, k in enumerate(grains[:,1]):
        diffraction_spot = []
        for i in k: 
            diffraction_spot.append(diffraction_spot_list[i])
        if len(diffraction_spot) < 2: continue;
        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.dilate(grains[j,0].astype(np.uint8), kernel, iterations=15)
        grain_masks.append(img_dilation.astype(bool))
        diffraction_spots.append(diffraction_spot)
        print(diffraction_spot)
    return diffraction_spots,grain_masks


def find_grain_atoms(diff_spots, grain_masks, im_filtered, R_diffspot, min_distance, threshold_rel):
    grain_atoms = []
    F1 = fftpack.fft2((im_filtered).astype(float))
    F2 = fftpack.fftshift(F1)
    for j, k in enumerate(tqdm(diff_spots)):
        mask = np.zeros(F2.shape[:2], dtype="uint8")
        for i in k:
            x, y = i
            cv2.circle(mask, [x,y], R_diffspot, j+1, -1)
            cv2.circle(mask, [F2.shape[1]-x, F2.shape[0]-y], R_diffspot, j+1, -1)
        im1 = fftpack.ifft2(fftpack.ifftshift(F2*mask)).real
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(abs(F2*mask))
        # plt.xlim(FFT.shape[1]//2 - Peak_end, FFT.shape[1]//2 + Peak_end)
        # plt.ylim(FFT.shape[1]//2 - Peak_end, FFT.shape[1]//2 + Peak_end)
        # plt.subplot(1,2,2)
        # plt.imshow(im1)
        coordinates = peak_local_max(im1*grain_masks[j], min_distance=min_distance, threshold_rel=threshold_rel,exclude_border=5)
#         plt.scatter(coordinates[:, 1], coordinates[:, 0],s=1.5, color = 'r')
        grain_atoms.append(coordinates)
    coordinates_all = peak_local_max(im_filtered, min_distance=min_distance,exclude_border=5)
    return grain_atoms, coordinates_all
