import cv2
import time
import autopy
import numpy as np
import HandTrackingModule as htm
wCam,hCam=640,480
cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
PRTime=0
detector=htm.handDetector(detectionCon=0.05,maxHands=2)
wScr,hScr=autopy.screen.size()
print(wScr,hScr)
while True:
    success,img=cap.read()
    img=detector.findHands(img)
    lmlist,bbox= detector.findPosition(img)
    if len(lmlist)!=0:
        x1,y1=lmlist[8][1:]
        x2,y2=lmlist[12][1:]
        print(x1,y1,x2,y2)
        fingers=detector.fingersUp()
        #print(fingers)
        if fingers[1]== 1 and fingers[2]==0 :
            x3=np.interp(x1,(0,wCam), (0,wScr))
            y3=np.interp(y1,(0,hCam), (0,hScr))
            autopy.mouse.move(x3,y3)
        CURTime=time.time()
        fps=1/(CURTime-PRTime)
        PRTime=CURTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow("image",img)
        cv2.waitKey(1) 
           