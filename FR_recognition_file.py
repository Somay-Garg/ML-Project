import joblib
import cv2
import face_recognition#this library file can only be used if dlib library is pre-installed and dlib can be installed by installing visual studio
clf = joblib.load('SVM_classifier_model')#loading the pre-trained model
#for real time face recognition
cam = cv2.VideoCapture(0)
cam.set(3, 640)# set video widht
cam.set(4, 480)# set video height
while True:
    success, image = cam.read()
    img = cv2.resize(image, (0,0), None, 0.25, 0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_loc = face_recognition.face_locations(img)
    no = len(face_loc)
    if no > 0:#if face found
        for i in range(no):#if there are multiple faces in the video
            face_end = face_recognition.face_encodings(img, face_loc)[i]
            print("\nMaking Predcitions\n")
            name = clf.predict([face_end])
            y1,x2,y2,x1 = face_loc[0]
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(image,str(name),(x1+5,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    else:#if no face found
        print("\nNo Face found.\n")
    cv2.imshow('camera', image)
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
cam.release()
cv2.destroyAllWindows()
