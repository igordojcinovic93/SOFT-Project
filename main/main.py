# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 18:59:48 2017

@author: igot
"""

import numpy as np
import cv2

"Inicijalizacija video snjimka"

cap = cv2.VideoCapture('video-0.avi')
print cap.isOpened()
"Do poslednje video-frame-a"
while(cap.isOpened()):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()

