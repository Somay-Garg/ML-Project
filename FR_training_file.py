import face_recognition
from sklearn import svm
import os
import cv2
import joblib
encodings = []
names = []
path='dataset'
imagePaths=[]
train_dir = [os.path.join(path,i)for i in os.listdir(path)]
print("Training Images\n")
for i in train_dir:
    for j in os.listdir(i):
        imagePaths.append(os.path.join(i,j))
for person_img in imagePaths:
    face = face_recognition.load_image_file(person_img)
    face_bounding_boxes = face_recognition.face_locations(face)
    if len(face_bounding_boxes) == 1:
        face_enc = face_recognition.face_encodings(face)[0]
        encodings.append(face_enc)
        name = os.path.split(person_img)[0].split('\\')[-1].split('.')[0]
        names.append(name)
    else:
        print(person_img + " image can't be used for training")
clf = svm.SVC(gamma='scale')
clf.fit(encodings,names)
joblib.dump(clf, 'SVM_classifier_model')
print("\nClassifier Trained.\n")
    