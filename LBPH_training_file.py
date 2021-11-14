import cv2
import numpy as np
from PIL import Image
import os
# Path for face image database
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
# function to get the images and label data
def train_images(path):
    img = [os.path.join(path,f) for f in os.listdir(path)]
    imagePaths=[]
    faces=[]
    ids = []
    for i in img:
        for j in os.listdir(i):
            imagePaths.append(os.path.join(i,j))
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[0].split('.')[-1])
        #print(id)
        face = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in face:
            faces.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faces,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = train_images(path)
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.write('trainer.yml')
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
