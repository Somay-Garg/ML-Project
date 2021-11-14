import cv2
import os
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
path = 'dataset'
names = ['None']
name = os.listdir(path)
for i in name:
    names.append(i.split('.')[0])
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray,scaleFactor = 1.32,minNeighbors = 5,minSize = (int(minW), int(minH)),)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        cv2.putText(img,str(id),(x+5,y-5),font,1,(255,255,255),2)
    cv2.imshow('camera',img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
cam.release()
cv2.destroyAllWindows()
