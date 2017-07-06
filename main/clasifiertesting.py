from bcolz.tests.test_ndcarray import plainCompoundTest
from keras.models import Sequential
from matplotlib import pyplot as plt
import cv2

from sklearn.externals import joblib
import os
from skimage.feature import hog
import numpy as np
import picture_operations as po
import keras.models

from keras.layers.core import Dense

from vector import pnt2line

import uuid


import numpy
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import  Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, Convolution2D
import keras.layers
from keras.layers.core import Dense
from keras.utils import np_utils
from keras.optimizers import  SGD
from sklearn.externals import joblib

def neuralnet():
    model = Sequential()
    batch_size = 128
    num_classes = 10
    epochs = 100
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    model = Sequential()
    '''
    model.add(Dense(30, activation='relu',input_dim=784))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(0.2))
    '''
    model.add(Dense(70, activation='relu', input_dim=784))

    model.add(Dense(50, activation='relu'))
    model.add(Dense(10,activation='softmax'))
    sgd = SGD(lr=0.1, decay=0.001, momentum=0.7)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    if(os.path.isfile("model3.h5")):
        model.load_weights("model3.h5")
    else:
        model.fit(x_train,y_train,batch_size=batch_size,nb_epoch=epochs, verbose=1, validation_data=(x_test, y_test))
        model.save_weights("model3.h5")
    return model

def lineregion(img):
    gray_img = np.ndarray((img.shape[0], img.shape[1]))
    for i in np.arange(0,img.shape[0]):
        for j in np.arange(0, img.shape[1]):
            if img[i,j,2] < 50 and img[i,j,1] <50 and not img[i,j,0] <150:
                gray_img[i,j] = 255
            else:
                gray_img[i,j] = 0
    gray_img = gray_img.astype('uint8')
    gray_img_d1 = cv2.dilate(gray_img, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=2)
    return gray_img

def number_region(num_img):
    num_img = cv2.morphologyEx(num_img, cv2.MORPH_DILATE, (3,3))

    for i in range(0,num_img.shape[0]):
        for j in range(0,num_img.shape[1]):
            if num_img[i,j] < 150:
                num_img[i,j] = 0
            else:
                num_img[i,j] = 255

    return num_img



for idx in np.arange(4,10):
    clf = neuralnet()
    cap = cv2.VideoCapture('input/video-' + str(idx) + '.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output/output-' + str(idx) + '.avi', fourcc, 20.0, (640,480))
    sum = 0
    appendsum = []
    i = 0
    c_line_x = 0
    c_line_y = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret==True:
            im = frame
            if(c_line_x == 0):
                line = lineregion(im)
                _,ctrline, _ = cv2.findContours(line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rect_line = cv2.boundingRect(ctrline[0])
                ctr = ctrline[0]
                l_m = cv2.moments(ctr)    #momenti za liniju
                c_line_x = int(l_m["m10"] / l_m["m00"]) #centar x kordinate
                c_line_y = int(l_m['m01'] / l_m['m00']) #centar y kordinate

                h_line = rect_line[3]
                w_line = rect_line[2]
                line_pts = [(c_line_x-w_line/2, c_line_y+h_line/2), (c_line_x+w_line/2, c_line_y-h_line/2)]

            im[:,:,1]=0
            blue_min = np.array([0,0,200], np.uint8)
            blue_max = np.array([50,50,255], np.uint8)
            dst = cv2.inRange(im,blue_min,blue_max)

            print dst
            no_blue = cv2.countNonZero(dst)


            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ret, im_th = cv2.threshold(im_gray, 80, 255, cv2.THRESH_BINARY)
            cv2.imshow("test", im_th)

            _, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            rects = [cv2.boundingRect(ctr) for ctr in ctrs]

            if im_th is not None:
                for ctr in ctrs:
                    rect = cv2.boundingRect(ctr)
                    c_m = cv2.moments(ctr)
                    if(c_m['m00'] != 0):
                        c_num_x = int(c_m['m10']/c_m['m00'])    #centralne kordinate kontura brojeva
                        c_num_y = int(c_m['m01'] / c_m['m00'])
                        #cv2.rectangle(im, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 3)
                        leng = int(rect[3]*1.6)
                        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
                        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
                        x_c = rect[0] + rect[2] /2
                        y_c = 480 - (rect[1] + rect[3]/2)
                        rect_center = (c_num_x,c_num_y)
                        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
                        height = np.size(roi, 0)
                        width = np.size(roi, 1)


                        if height > 15 and width > 15:
                            roi = cv2.resize(roi, (28,28), interpolation=cv2.INTER_AREA)
                            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14,14), cells_per_block=(1,1), visualise=False)
                            dist, pnt, r = pnt2line(rect_center, line_pts[0], line_pts[1])
                            roi = number_region(roi)

                            roid_test = roi
                            roid_test = roid_test / 255
                        #    roid_test = cv2.morphologyEx(roid_test,cv2.MORPH_DILATE, (2,2))
                            roid_test = roi.reshape(1,-1)


                            nbr = clf.predict_classes(roid_test.astype('float64'), verbose=1)
                            cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
                            cv2.imshow("t2", im)
                            if(dist < 0.8):
                                print int(nbr)
                                sum += int(nbr)
                                appendsum.append(int(nbr))
                        out.write(im)
                    if(cv2.waitKey(1) & 0xFF == ord('q')):
                        break
        else:
            break
    print sum
    print appendsum
    output_file = open('result/res-'+str(idx)+'.txt','a')
    output_file.write('Elements: '+ str(appendsum)+ ' Sum: '+ str(sum))
    output_file.close()
    cap.release()
    out.release()

