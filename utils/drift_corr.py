import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
import os
from tqdm import tqdm
import tifffile
from skimage.registration import phase_cross_correlation
from skimage.morphology import disk
from skimage.filters import median


def select_ROI(img):
    showCrosshair = False
    fromCenter = False
    r = cv2.selectROI("Image", img, fromCenter, showCrosshair)
    cv2.destroyAllWindows()
    return r

def median_filter(img, size):
    med = median(img, disk(size))
    return med


def shift_img(img,tx,ty,output_size):
    M = np.array([[1.,0,ty],[0,1.,tx]])
    img_shift = cv2.warpAffine(img,M,output_size)
    return img_shift


def drift_vectors_CC(video_loc, roi, RANGE = (0,-1,1), overlap_ratio = 0.8, upsample_factor=20):
    cap = cv2.VideoCapture(video_loc)
    frames = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT)))
    width=int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_WIDTH)))-1
    height=int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_HEIGHT)))-1
    print('Num of Frames = ', frames, ', width = ',width, ', height = ',height)
    drift_vectors = [[0,0,0,0]]
    frames = frames if RANGE[1] == -1 else RANGE[1]
    for frame_number in tqdm(range(max(RANGE[2],RANGE[0]), frames-1, RANGE[2])):
        left, right, bottom, top = roi[1], roi[1]+roi[3], roi[0], roi[0]+roi[2]
        cap.set(1, frame_number)
        res, image = cap.read()
        img_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[left:right, bottom:top]
        reference_frame = frame_number - RANGE[2]
        rep = 0
        while rep < 3:
            rep += 1
            cap.set(1, reference_frame)
            res, image = cap.read()
            img_2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[left:right, bottom:top]
            shifts,error,phasediff = phase_cross_correlation(img_2,img_1,overlap_ratio=overlap_ratio, upsample_factor=20, normalization="phase")
            if abs(phasediff) < 1: break;
            reference_frame -= RANGE[2]*5
            reference_frame = RANGE[0]
            print(reference_frame)
        else:
            print('Fail to track on frame', frame_number)
            shifts,error,phasediff = [0,0], 0, 0
        drift_vectors.append([frame_number,shifts[0],shifts[1],phasediff])
        
    drift_vectors = np.array(drift_vectors)
    for i in range(1, len(drift_vectors)):
        drift_vectors[i][1:3] += drift_vectors[i-1][1:3]
    return drift_vectors


def drift_corr_tiff(video_loc, target_loc, RANGE, vectors):
    cap = cv2.VideoCapture(video_loc)
    frames = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT)))
    width=int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_WIDTH)))-1
    height=int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_HEIGHT)))-1
    print('Num of Frames = ', frames)
    vectors[:,1:3] += vectors[:,1:3].min()
    x_size, y_size = height+vectors[:,2].max().astype(int), width+vectors[:,1].max().astype(int)
    ##########################
    with tifffile.TiffWriter(target_loc) as stack:
        for frame_number,loc in enumerate(vectors):
            x_correct,y_correct = loc[1:3]
            cap.set(cv2.CAP_PROP_POS_FRAMES, RANGE[0]+frame_number*RANGE[2])
            res, img_1 = cap.read()
            img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            img_shift = shift_img(img_1, x_correct, y_correct, [x_size, y_size])
            stack.save(img_shift, photometric='minisblack')   