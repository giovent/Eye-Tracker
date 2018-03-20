#Ignore warnings from Tensorflow
from utils import *
from model_builder import *

face_cascade = cv2.CascadeClassifier('Haar-Cascades/haarcascade_frontalface_alt.xml');

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import cv2
from model_builder import *

TRAINED_MODEL_PATH = 'Trained Models/model.ckpt'

face_cascade = cv2.CascadeClassifier('Haar-Cascades/haarcascade_frontalface_alt.xml');

cap = cv2.VideoCapture(0)

hi = 480
wi = 640

sess = tf.Session()

### Load the trained model
saver = tf.train.Saver()
saver.restore(sess, TRAINED_MODEL_PATH)
###

i=0
filter_len = 3
ps = list(np.zeros([filter_len,2]))
dist = []
while(True):
    i+=1

    ret,img = cap.read()
    if(not ret):
        print 'Error, no camera device found'
        break
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for fn,(xx,y,w,h) in enumerate(faces):
        ROI = gray[y+h/6:y+h/2,xx+w/6:xx+w/6+int((h/2-h/6)/15.*36.)]
        dist.append([h/2-h/6,int((h/2-h/6)/15.*36.)])
        cv2.imshow("ROI",ROI)
        ROI = cv2.resize(ROI,(72,32))
        cv2.imshow("ROIs",ROI)
        # Normalize the input from 0 to 1 (as it has been done the data in training)
        ROI = ROI/255.
        p = np.asarray(sess.run(output, feed_dict={x:np.reshape(ROI,(-1,2*2*16*36)).astype(float),keep_prob:1.0}))
        ps.append(p[0])
        ps = ps[1:]
        f = np.mean(ps,axis=0)
        #print p
        print f
        #cv2.rectangle(img,(wi/2+wi/(2*15)*int(p[0][1]),hi/2-hi/(2*4)*int(p[0][0])),(wi/2+wi/(2*15)*int(p[0][1])+5,hi/2-hi/(2*4)*int(p[0][0])+5),(0,0,255,255),3)
        point1 = (max(2,min(wi-2,wi/2+wi*int(f[1])/(2*15))),max(2,min(hi-2,hi/2-hi*int(f[0])/(2*4))))
        point2 = (point1[0]+5,point1[1]+5)
        cv2.rectangle(img,point1,point2,(0,255,0,255),3)


    cv2.rectangle(img,(0,0),(wi,hi),(0,0,255,255),3)
    cv2.rectangle(img,(0,0),(wi/3,hi),(0,0,255,255),3)
    cv2.rectangle(img,(0,0),(wi,hi/3),(0,0,255,255),3)
    cv2.rectangle(img,(0,0),(wi*2/3,hi),(0,0,255,255),3)
    cv2.rectangle(img,(0,0),(wi,hi*2/3),(0,0,255,255),3)

    cv2.imshow("Original",img)
    cv2.waitKey(1)

sess.close()
