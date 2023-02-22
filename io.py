import numpy as np
import cv2
import os

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