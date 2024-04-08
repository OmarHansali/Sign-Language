# python version 3.10.0

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf
import pickle


#### test the model on real time prediction ####

# Connect to camera port 0
cap = cv2.VideoCapture(0)

# create the hand detector from mediapipe
detector=HandDetector(maxHands=1)

# import the model labels
with open('model/model-labels.pkl', 'rb') as file:
    # Read the contents of the file into a variable
    labels = pickle.load(file)

print(labels, labels[0], labels[1], labels[2], labels[3])

# Load the pre-trained TensorFlow CNN model
classifier = tf.keras.models.load_model('model/model-weights.h5')

# For real-time sign prediction
offset = 40
imgSize = 128
while True:
    success, img = cap.read()

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop= img[y-offset:y+h+offset,x-offset:x+w+offset]

        # Check if the cropped image is not empty
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
            imgResize = cv2.resize(imgGray, (imgSize, imgSize))
            imgInput = np.expand_dims(imgResize, axis=0)
            imgInput = np.expand_dims(imgInput, axis=-1)  # Add channel dimension
            imgInput = imgInput.astype(np.float32) / 255.0  # Normalize pixel values

            # Preprocess the image for the CNN model
            prediction = classifier.predict(imgInput)
            print(prediction)
            index = np.argmax(prediction)
            sign_label = labels[index]

            # Display the predicted sign
            cv2.putText(img, sign_label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-time Sign Prediction", img)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()