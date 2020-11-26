import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing import image

faceCascade = cv2.CascadeClassifier("Resources/haarcascade_lefteye_2splits.xml")

model = load_model('Mask_model')
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

cap = cv2.VideoCapture(0)
while(True):
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgCropped = img.copy()
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x-130, y-50), (x + w + 40, y + h + 120), (255, 0, 0), 2)
        imgCropped = img[y-50:y+h+120, x-130:x+w+40]
        if len(imgCropped) == 0  :
            continue
        else:
            imgCropped = cv2.resize(imgCropped,(64,64))
            imgCropped = cv2.cvtColor(imgCropped,cv2.COLOR_BGR2RGB)
            test_image = image.img_to_array(imgCropped)
            test_image = np.expand_dims(test_image, axis=0)
            result = model.predict(test_image)
            print(int(result[0][0]))
            if int(result[0][0]) == 0:
                cv2.putText(img,"With Mask",(x-130, y-50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
            else:
                cv2.putText(img, "Without Mask", (x - 130, y - 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

    cv2.imshow("result",img)
    cv2.waitKey(1)