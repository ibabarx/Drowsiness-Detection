import cv2 as cv
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf


model = keras.models.load_model("./MobileNet.h5")
webcam = cv.VideoCapture(0)

while True:
    isRecording,Frame= webcam.read()
    black_and_white = cv.cvtColor(Frame, cv.COLOR_BGR2GRAY)
    rpred = 1
    lpred = 1
    for (x, y, w, h) in cv.CascadeClassifier('haarcascade_righteye_2splits.xml').detectMultiScale(black_and_white,1.1,4):
        cv.rectangle(Frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        r_eye=black_and_white[y:y+h,x:x+w]
        r_eye = cv.resize(r_eye, (224, 224))
        r_eye = np.expand_dims(r_eye, axis=-1)
        r_eye = np.repeat(r_eye, 3, axis=-1)
        r_eye = np.expand_dims(r_eye, axis=0)
        r_eye= r_eye/255
        rpred = np.argmax(model.predict(r_eye))
        if(rpred==1):
            lbl='Open'
        if(rpred==0):
            lbl='Closed'
        break
        
    for (x2, y2, w2, h2) in cv.CascadeClassifier('haarcascade_lefteye_2splits.xml').detectMultiScale(black_and_white, 1.1, 4):
        cv.rectangle(Frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
        l_eye= black_and_white[y2:y2+h2,x2:x2+w2]
        l_eye = cv.resize(l_eye, (224, 224))
        l_eye = np.expand_dims(l_eye, axis=-1)
        l_eye = np.repeat(l_eye, 3, axis=-1)
        l_eye = np.expand_dims(l_eye, axis=0)
        l_eye= l_eye/255
        lpred = np.argmax(model.predict(l_eye))
        if(lpred==1):
            lbl='Open'
        if(lpred==0):
            lbl='Closed'
        break

    if (rpred == 0) and (lpred == 0):
        cv.putText(Frame, 'Closed',(50,50),cv.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255))
        print('/a')
    else:
        cv.putText(Frame, 'Open',(50,50),cv.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0))

    cv.imshow('live', Frame)
    if cv.waitKey(20) & 0xff==ord('d'):
        break

webcam.release()
cv.destroyAllWindows()