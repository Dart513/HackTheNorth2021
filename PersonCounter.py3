import numpy as np
import cv2

from collections import namedtuple

#make a person object
MyStruct = namedtuple("Person", "id x y")


#initiate the HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

#open webcam video capture stream
cap = cv2.VideoCapture(0)


#list of people and entities



while(True):
    #read the frame
    ret, frame = cap.read()

    #resize and grayscale frame for faster detection
    frame = cv2.resize(frame, (680, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #ret,frame = cv2.threshold(frame,80,255,cv2.THRESH_BINARY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()

cv2.destroyAllWindows()
# the following is necessary on the mac,
# maybe not on other platforms:
cv2.waitKey(1)
