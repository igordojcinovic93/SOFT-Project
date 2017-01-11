# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 19:22:50 2017

@author: igot
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.morphology import opening, closing,disk,square,diamond, erosion
from skimage.exposure import histogram
from skimage.measure import label,regionprops

cap = cv2.VideoCapture('video-0.avi')

#funkcija za iscrtavanje regiona
def draw_border_region(region,img_size):
    img_r = np.ndarray((img_size[0], img_size[1]), dtype = 'float32')
    coords = region.coords
    for coord in coords:
        img_r[coord[0], coord[1]] = 1
    return img_r

#funkcija za iscrtavanje regiona brojeva
def draw_number_regions(regions, img_size):
    img_r = np.ndarray((img_size[0], img_size[1]), dtype = 'float32')
    for reg in regions:
        coords = reg.coords
        for coord in coords:
            img_r[coord[0], coord[1]] = 1
    return img_r
    
#funkcija za izvlacenja kose crte
def border_region(regions):
    for region in regions:
        for prop in region:
            if(region.area > 100):
                print(prop, region[prop])
                reg = region
    return reg
    
#funkcija za izvlacenje regiona brojeva
def number_regions(regions):
    for region in regions:
        num_regs = []
        for prop in region:
            if(region.area > 50):
                print("sucess")
                num_regs.append(region)
    return num_regs
#funkcija za izvlacenje svih regiona
def get_regions(img):
    labeled_img = label(img)
    regions = regionprops(labeled_img)
    return regions
    
#test
str_elem = disk(1.5)
#img = [cap.read()[1] for i in xrange(5)]
img_arr = [cap.read()[1] for i in xrange(20)]
img = img_arr[15]
#plt.imshow(img) 
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,img_tresh = cv2.threshold(img_gray,25,250,cv2.THRESH_BINARY)

img_erode = erosion(img_tresh,selem = square(3))

img_tresh_open = opening(img_tresh,selem = square(5))


#izdvajanje regiona

#labeled_img = label(img_tresh)
#regions = regionprops(labeled_img)

#plt.imshow(img_tresh,'gray')
#plt.imshow(labeled_img)
#print('# Regiona: {}'.format(len(get_regions(img_tresh)))
plt.imshow(draw_border_region(border_region(get_regions(img_tresh)),img_tresh.shape))
plt.imshow(draw_number_regions(number_regions(get_regions(img_tresh)),img_tresh.shape))
#cv2.imwrite('thres_img.png',img_tresh)
cv2.imwrite('gray_img.png',img_gray)
cv2.imwrite('treshold_img.png',img_tresh)
cv2.imwrite('treshold_open.png',img_tresh_open)
#plt.imshow(img_tresh,'gray')
#plt.imshow(gray,'gray')
#gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in img]
    
#gray = [np.float64(i) for i in gray]
        
#noise = np.random.randn(*gray[1].shape)*10
#noisy = [i+noise for i in gray]
#noisy = [np.uint8(np.clip(i,0,255)) for i in noisy]
#dst = cv2.fastNlMeansDenoisingMulti(noisy,2,5,None,4,7,35)

#plt.imshow(dst)

