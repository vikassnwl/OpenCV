import numpy as np
import cv2
import matplotlib.pyplot as plt

haar_cascade = cv2.CascadeClassifier('haar_face.xml')

people = ['Ben Affleck', 'Elton John', 'Jerry Seinfeld', 'Madonna', 'Mindy Kaling']

# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv2.imread(r'train/Mindy Kaling/mindy-kaling-bj-novak-removebg.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for x, y, w, h in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with confidence of {confidence}')

    cv2.putText(img, str(people[label]), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# cv2.imshow('Detected Face', img)

# cv2.waitKey(0)

# plt.subplot(121);plt.imshow(gray, cmap='gray')
# plt.subplot(122);plt.imshow(img[:, :, ::-1])
plt.imshow(img[:, :, ::-1])
plt.show()