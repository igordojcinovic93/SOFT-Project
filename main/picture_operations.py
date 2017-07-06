'''
Modul pripremljen za rad nad slikama

Igor Dojcinovic
'''

import cv2
from skimage.feature import hog
import numpy as np

#osnovne funkcije za rad sa slikom

def read_image(imgName):
    return cv2.imread(imgName)

def trans_to_grey(img):
    im_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_grey = cv2.GaussianBlur(im_grey, (5, 5), 0)
    return im_grey


def apply_thresh(img):
    im_grey = trans_to_grey(img)
    ret, im_th = cv2.threshold(im_grey, 100, 255, cv2.THRESH_BINARY)
    im_th = cv2.erode(im_th,(3,3))
    im_th = cv2.dilate(im_th,(5,5))
    return ret, im_th


#funkcija za izdvajanje regiona u obliku kvadrata
def extract_rectangle_contours(threshold_img):
    image, ctrs, hier = cv2.findContours(threshold_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    return rects, ctrs

'''
#funkcija za crtaje kontura
def draw_conturs(image_original, rectangles, clasifier):
    sumonum = 0
    for rectangle in rectangles:
        cv2.rectangle(image_original, (rectangle[0], rectangle[1]),(rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (0, 255, 0), 3)
        leng = int(rectangle[3] * 1.6)
        pt1 = int(rectangle[1] + rectangle[3] // 2 - leng // 2)
        pt2 = int(rectangle[0] + rectangle[2] // 2 - leng // 2)
        ret, im_th = apply_thresh(image_original)
        cv2.imshow("test",im_th)
        roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        roi_hod_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        nbr = clasifier.predict(np.array([roi_hod_fd], 'float64'))
        cv2.putText(image_original, str(int(nbr[0])), (rectangle[0], rectangle[1]),
                                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        #sumonum = sumonum + nbr
    print sumonum
'''