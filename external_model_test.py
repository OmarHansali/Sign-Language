# python version 3.10.0

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from cvzone.ClassificationModule import Classifier

# pretrained model

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)

# labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
model=Classifier("model/Test.h5","model/model-labels.txt")

offset=20
imgSize=300

c=0


while True:
    success,img=cap.read()
    imgoutput = img.copy()
    hands,img=detector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']

        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop= img[y-offset:y+h+offset,x-offset:x+w+offset]
       
        aspectRatio = h/w
        if aspectRatio>1:
            k=imgSize/h
            wcal=math.ceil(k*w)
            imgResize=cv2.resize(imgCrop,(wcal,imgSize))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((300-wcal)/2)
            imgWhite[:,wGap:wcal+wGap]=imgResize
            prediction, index= model.getPrediction(imgWhite)
            
        else:
            k=imgSize/w
            hcal=math.ceil(k*h)
            imgResize=cv2.resize(imgCrop,(imgSize,hcal))
            imgResizeShape=imgResize.shape
            hGap=math.ceil((300-hcal)/2)
            imgWhite[hGap:hcal+hGap,:]=imgResize
            prediction, index= model.getPrediction(imgWhite)
            
           

        cv2.putText(imgoutput, model.list_labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)

    cv2.imshow("Image",imgoutput)
   
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()