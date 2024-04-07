# python version 3.10.0
import cv2
from cvzone.HandTrackingModule import HandDetector
import time

# Connect to camera port 0
cap = cv2.VideoCapture(0)

# Create the hand detector from mediapipe
detector=HandDetector(maxHands=1)

# For real-time sign prediction
offset = 20
imgSize = 128

# Specify the folder to save the images
folder = "data/A"

# Initialize a counter
c=0

# Start an infinite loop
while True:
    # Read a frame from the camera
    success, img = cap.read()

    # Detect hands in the frame
    hands, img = detector.findHands(img)

    # If hands are detected
    if hands:
        # Get the bounding box of the first hand detected
        x, y, w, h = hands[0]['bbox']

        # Crop the image around the hand
        imgCrop= img[y-offset:y+h+offset,x-offset:x+w+offset]

        # Check if the cropped image is not empty
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            # Convert the cropped image to grayscale
            imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)

            # Resize the grayscale image
            imgResize = cv2.resize(imgGray, (imgSize, imgSize))

        # # Display the cropped image
        # cv2.imshow("Imagecrop",imgCrop)

        # Display the input image
        cv2.imshow("Imagewhite",imgResize)

    # Display the original frame
    cv2.imshow("Image",img)

    # Wait for a key press
    key=cv2.waitKey(1)

    # If 's' key is pressed
    if key== ord("s"): 
        # Increment the counter
        c+=1
        # Save the input image
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgResize)
        # Print the counter
        print(c)

    # If 'q' key is pressed
    if key == ord("q"):
        break

# Release the camera
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()