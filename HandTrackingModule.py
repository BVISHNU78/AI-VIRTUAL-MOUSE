import cv2
import mediapipe as mp
import time
import bbox
import math
class handDetector():
    def __init__(self,mode=False,maxHands=1,modelC=1,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.modelC=modelC
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        self.mpHands =mp.solutions.hands
        self.hands =self.mpHands.Hands(self.mode,self.maxHands,self.modelC,self.detectionCon,self.trackCon)
        self.mpDraw =mp.solutions.drawing_utils
        self.tipIds=[4,8,12,16,20]
    def findHands(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results= self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self,img,handNo=0,draw=True):
        xList=[]
        yList=[]
        bbox=[]
        self.lmlist=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                height,width,channel=img.shape
                cx,cy=int(lm.x*width),int(lm.y*height)
                #print([cx,cy])
                xList.append(cx)
                yList.append(cy)
                self.lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
            xmin,xmax=min(xList),max(xList)
            ymin,ymax=min(yList),max(yList)
            bbox=xmin,ymin,xmax,ymax
            if draw:
                cv2.rectangle(img,(bbox[0]-20,bbox[1]-20),(bbox[2]+20,bbox[3]+20),(0,255,0),2)
        return self.lmlist,bbox
    def fingersUp(self):
        fingers = []
        if self.lmlist[self.tipIds[0]][1] >self.lmlist[self.tipIds[0] -1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):#checking tip point is below tippoint-2 (only in Y direction)
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    def findDistance(self, p1, p2, img, draw=True,r=15,t=3):# finding distance between two points p1 & p2
        x1, y1 = self.lmlist[p1][1],self.lmlist[p1][2]#getting x,y of p1
        x2, y2 = self.lmlist[p2][1],self.lmlist[p2][2]#getting x,y of p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]
    
def main():
    PRTime=0
    CURTime=0
    cap=cv2.VideoCapture(0)
    detector=handDetector() 
    while True:
        success,img = cap.read()
        img=detector.findHands(img)
        lmList,bbox=detector.findPosition(img)
        if len(lmList)!= 0:
            print(lmList[4])
        CURTime=time.time()
        fps=1/(CURTime-PRTime)
        PRTime=CURTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow("image",img)
        cv2.waitKey(1) 
if __name__=="__main__":
    main()